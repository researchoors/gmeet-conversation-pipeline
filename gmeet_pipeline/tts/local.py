"""Local TTS backend — Kokoro text-to-speech with optional RVC voice conversion.

Ported from meeting_agent_rvc.py (lines 258-348, 492-523).

The heavy Kokoro KPipeline and RVC VC modules are loaded lazily on first use
so that importing this module never crashes when the optional ML dependencies
are not installed.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import uuid
from pathlib import Path
from typing import Optional, Union

from .base import BaseTTS

logger = logging.getLogger("gmeet_pipeline.tts.local")


class LocalTTS(BaseTTS):
    """Generate speech locally using Kokoro TTS and optional RVC voice
    conversion.

    The Kokoro KPipeline and RVC inference modules are loaded lazily via
    :meth:`_ensure_init` on the first call to :meth:`generate`.  This keeps
    the import cheap when the heavy ML dependencies are not available.
    """

    def __init__(
        self,
        rvc_model_path: str,
        rvc_exp_dir: str,
        rvc_repo_dir: str,
        rvc_f0_method: str = "rmvpe",
        rvc_f0_up_key: int = 0,
        rvc_index_rate: float = 0.0,
        kokoro_voice: str = "af_heart",
        audio_dir: Union[str, Path] = "",
    ) -> None:
        self.rvc_model_path = rvc_model_path
        self.rvc_exp_dir = rvc_exp_dir
        self.rvc_repo_dir = rvc_repo_dir
        self.rvc_f0_method = rvc_f0_method
        self.rvc_f0_up_key = rvc_f0_up_key
        self.rvc_index_rate = rvc_index_rate
        self.kokoro_voice = kokoro_voice
        self.audio_dir = Path(audio_dir) if audio_dir else Path.home() / ".hermes" / "audio_cache" / "meeting_tts"

        # Lazy-loaded heavy objects (populated by _ensure_init)
        self._kokoro_pipeline = None  # kokoro.KPipeline
        self._vc = None               # RVC VC module
        self._tgt_sr: int = 0
        self._net_g = None
        self._version: str = ""
        self._hubert_model = None
        self._initialized: bool = False

        # Ensure the audio directory exists
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _ensure_init(self) -> None:
        """Load Kokoro KPipeline and RVC VC module (runs once)."""
        if self._initialized:
            return

        # --- espeak-ng lib + data path (macOS brew) ---
        try:
            import os
            espeak_data = "/opt/homebrew/Cellar/espeak-ng/1.52.0/share/espeak-ng-data"
            if os.path.isdir(espeak_data):
                os.environ.setdefault("ESPEAK_DATA_PATH", espeak_data)
            espeak_lib = "/opt/homebrew/lib/libespeak-ng.dylib"
            if os.path.exists(espeak_lib):
                import ctypes
                ctypes.cdll.LoadLibrary(espeak_lib)
        except Exception:
            pass

        # --- Kokoro (pre-init spacy to avoid circular import) ---
        try:
            import spacy
            import spacy.util   # fully init spacy first (circular import workaround)
            from kokoro import KPipeline  # type: ignore[import-untyped]

            self._kokoro_pipeline = KPipeline(lang_code="a")
            logger.info("Kokoro KPipeline loaded")
        except Exception as exc:
            logger.error(f"Failed to load Kokoro KPipeline: {exc}")
            self._kokoro_pipeline = None

        # --- RVC ---
        try:
            import sys

            rvc_repo = self.rvc_repo_dir
            if rvc_repo and rvc_repo not in sys.path:
                sys.path.insert(0, rvc_repo)

            from infer.lib.infer_pack.models import (
                SynthesizerTrnMs256NSFsid,
                SynthesizerTrnMs256NSFsid_nono,
                SynthesizerTrnMs768NSFsid,
                SynthesizerTrnMs768NSFsid_nono,
            )
            from infer.lib.infer_pack.models_onnx import (
                SynthesizerTrnMsNSFsidM,
                SynthesizerTrnMsNSFsidNonoM,
            )
            from infer.inference import load_hubert, vc_single  # type: ignore[import-untyped]

            import torch  # type: ignore[import-untyped]

            # Load RVC model
            cpt = torch.load(self.rvc_model_path, map_location="cpu")
            self._version = cpt.get("version", "v1")

            # Determine model class based on version / architecture
            if self._version == "v2":
                if cpt.get("f0", 1) == 0:
                    net_g_cls = SynthesizerTrnMs768NSFsid_nono
                else:
                    net_g_cls = SynthesizerTrnMs768NSFsid
            else:
                if cpt.get("f0", 1) == 0:
                    net_g_cls = SynthesizerTrnMs256NSFsid_nono
                else:
                    net_g_cls = SynthesizerTrnMs256NSFsid

            net_g = net_g_cls(
                *cpt["config"],
                is_half=False,
                version=self._version,
            )
            del net_g.enc_q  # noqa: attribute exists at runtime

            net_g.load_state_dict(cpt["weight"], strict=False)
            net_g.eval()
            for p in net_g.parameters():
                p.requires_grad = False

            self._tgt_sr = net_g.target_sr
            self._net_g = net_g

            # Load HuBERT
            self._hubert_model = load_hubert("cpu", False)
            self._vc = vc_single  # function reference

            logger.info(
                f"RVC model loaded (version={self._version}, "
                f"tgt_sr={self._tgt_sr})"
            )

        except Exception as exc:
            logger.error(f"Failed to load RVC: {exc}")
            self._vc = None
            self._net_g = None
            self._hubert_model = None

        self._initialized = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def generate(self, text: str, bot_id: str) -> Optional[str]:
        """Generate TTS audio locally using Kokoro + RVC.

        Runs Kokoro + RVC inference in a thread pool to avoid blocking the
        event loop.  Returns the WAV filename on success, or ``None`` on
        failure.
        """
        self._ensure_init()

        if self._kokoro_pipeline is None:
            logger.error("Kokoro pipeline not available — cannot generate TTS")
            return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_sync, text, bot_id)

    # ------------------------------------------------------------------
    # Synchronous generation (runs in thread pool)
    # ------------------------------------------------------------------

    def _generate_sync(self, text: str, bot_id: str) -> Optional[str]:
        """Blocking Kokoro → (optional RVC) → WAV save."""
        import numpy as np

        try:
            # --- Kokoro TTS ---
            audio_segments: list[np.ndarray] = []
            for _, _, audio in self._kokoro_pipeline(text, voice=self.kokoro_voice):
                if audio is not None:
                    audio_np = (
                        audio.cpu().numpy()
                        if hasattr(audio, "cpu")
                        else np.array(audio, dtype=np.float32)
                    )
                    audio_segments.append(audio_np)

            if not audio_segments:
                logger.warning("Kokoro produced no audio")
                return None

            audio_np = np.concatenate(audio_segments)

            # --- Optional RVC voice conversion ---
            if self._vc is not None and self._net_g is not None and self._hubert_model is not None:
                audio_np = self._apply_rvc(audio_np)

            # --- Save WAV ---
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = self.audio_dir / filename
            self._save_wav(audio_np, filepath, sample_rate=22050)

            logger.info(
                f"Local TTS: {filename} ({len(audio_np)} samples) for bot {bot_id}"
            )
            return filename

        except Exception as exc:
            logger.error(f"Local TTS generation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # RVC voice conversion
    # ------------------------------------------------------------------

    def _apply_rvc(self, audio_np) -> "np.ndarray":
        """Apply RVC voice conversion to a float32 numpy audio array."""
        import numpy as np
        import torch  # type: ignore[import-untyped]

        # RVC expects int16 input
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)

        # Try to load index file for feature retrieval
        index_path = ""
        try:
            import glob

            index_files = glob.glob(f"{self.rvc_exp_dir}/*.index")
            if index_files:
                index_path = index_files[0]
        except Exception:
            pass

        # vc_single signature:
        #   vc_single(sid, audio, f0_up_key, f0_method, index_path, index_rate,
        #             tgt_sr, net_g, hubert_model, version)
        result = self._vc(
            0,  # sid
            audio_int16,
            self.rvc_f0_up_key,
            self.rvc_f0_method,
            index_path,
            self.rvc_index_rate,
            self._tgt_sr,
            self._net_g,
            self._hubert_model,
            self._version,
        )

        # result is (tgt_sr, numpy_audio)
        if isinstance(result, tuple) and len(result) == 2:
            return result[1].astype(np.float32) / 32768.0

        return audio_np

    # ------------------------------------------------------------------
    # WAV writer
    # ------------------------------------------------------------------

    @staticmethod
    def _save_wav(
        audio_np, filepath: Path, sample_rate: int = 22050
    ) -> None:
        """Write a float32 numpy array as a 16-bit PCM WAV file."""
        import numpy as np

        # Clip and convert to int16
        audio_int16 = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
        raw = audio_int16.tobytes()

        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(raw)

        with open(filepath, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write(struct.pack("<I", 36 + data_size))
            f.write(b"WAVE")
            # fmt chunk
            f.write(b"fmt ")
            f.write(struct.pack("<I", 16))  # chunk size
            f.write(struct.pack("<H", 1))   # PCM format
            f.write(struct.pack("<H", num_channels))
            f.write(struct.pack("<I", sample_rate))
            f.write(struct.pack("<I", byte_rate))
            f.write(struct.pack("<H", block_align))
            f.write(struct.pack("<H", bits_per_sample))
            # data chunk
            f.write(b"data")
            f.write(struct.pack("<I", data_size))
            f.write(raw)
