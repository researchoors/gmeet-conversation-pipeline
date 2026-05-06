"""Local TTS backend — Kokoro text-to-speech with optional RVC voice conversion.

The heavy Kokoro KPipeline and RVC VC modules are loaded lazily on first use
so that importing this module never crashes when the optional ML dependencies
are not installed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct
import tempfile
import uuid
from pathlib import Path
from typing import Optional, Union

from .base import BaseTTS

logger = logging.getLogger("gmeet_pipeline.tts.local")


class LocalTTS(BaseTTS):
    """Generate speech locally using Kokoro TTS and optional RVC voice
    conversion.

    The Kokoro KPipeline and RVC VC modules are loaded lazily via
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
        rvc_filter_radius: int = 3,
        rvc_rms_mix_rate: float = 0.25,
        rvc_protect: float = 0.33,
        kokoro_voice: str = "af_heart",
        audio_dir: Union[str, Path] = "",
    ) -> None:
        self.rvc_model_path = rvc_model_path
        self.rvc_exp_dir = rvc_exp_dir
        self.rvc_repo_dir = rvc_repo_dir
        self.rvc_f0_method = rvc_f0_method
        self.rvc_f0_up_key = rvc_f0_up_key
        self.rvc_index_rate = rvc_index_rate
        self.rvc_filter_radius = rvc_filter_radius
        self.rvc_rms_mix_rate = rvc_rms_mix_rate
        self.rvc_protect = rvc_protect
        self.kokoro_voice = kokoro_voice
        self.audio_dir = Path(audio_dir) if audio_dir else Path.home() / ".hermes" / "audio_cache" / "meeting_tts"

        # Lazy-loaded heavy objects (populated by _ensure_init)
        self._kokoro_pipeline = None  # kokoro.KPipeline
        self._vc = None               # RVC VC class instance
        self._rvc_loaded: bool = False
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

            # RVC needs to run from its repo dir for configs/ and assets/
            original_cwd = os.getcwd()
            os.chdir(rvc_repo)

            # Set env vars RVC's Config expects
            rvc_model_dir = str(Path(self.rvc_model_path).parent)
            os.environ.setdefault("weight_root", rvc_model_dir)
            os.environ.setdefault("index_root", rvc_model_dir)
            os.environ.setdefault("outside_index_root", rvc_model_dir)
            # rmvpe_root points to the RMVPE model inside the RVC repo
            os.environ.setdefault("rmvpe_root", os.path.join(rvc_repo, "assets", "rmvpe"))

            # MPS fallback for Apple Silicon
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            from configs.config import Config
            from infer.modules.vc.modules import VC

            config = Config()
            vc = VC(config)

            # get_vc loads the model — expects filename relative to weight_root
            model_filename = Path(self.rvc_model_path).name
            vc.get_vc(model_filename, 0.5, self.rvc_protect)

            self._vc = vc
            self._rvc_loaded = True

            # Restore working directory — note: HuBERT and RMVPE are lazy-loaded
            # by vc_single on first call. _apply_rvc sets cwd to the RVC repo
            # before calling vc_single, which ensures relative paths resolve.
            os.chdir(original_cwd)

            logger.info(
                f"RVC model loaded (version={vc.version}, "
                f"tgt_sr={vc.tgt_sr}, model={model_filename})"
            )

        except Exception as exc:
            logger.error(f"Failed to load RVC: {exc}")
            self._vc = None
            self._rvc_loaded = False
            # Try to restore cwd even on failure
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

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
            sample_rate = 24000  # Kokoro outputs at 24kHz

            # --- Optional RVC voice conversion ---
            if self._rvc_loaded and self._vc is not None:
                audio_np, sample_rate = self._apply_rvc(audio_np, sample_rate)

            # --- Save WAV ---
            filename = f"tts_{uuid.uuid4().hex[:8]}.wav"
            filepath = self.audio_dir / filename
            self._save_wav(audio_np, filepath, sample_rate=sample_rate)

            logger.info(
                f"Local TTS: {filename} ({len(audio_np)} samples, "
                f"{sample_rate}Hz) for bot {bot_id}"
            )
            return filename

        except Exception as exc:
            logger.error(f"Local TTS generation failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # RVC voice conversion
    # ------------------------------------------------------------------

    def _apply_rvc(self, audio_np, input_sr: int = 24000) -> tuple:
        """Apply RVC voice conversion to a float32 numpy audio array.

        Returns (audio_float32, output_sample_rate).
        """
        import numpy as np

        # vc_single may lazy-load HuBERT which uses relative paths —
        # ensure cwd is the RVC repo during the call.
        original_cwd = os.getcwd()
        if self.rvc_repo_dir:
            os.chdir(self.rvc_repo_dir)

        # Write Kokoro output to a temp WAV file — vc_single expects a file path
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._save_wav(audio_np, Path(tmp_path), sample_rate=input_sr)

            # Find index file for feature retrieval
            index_path = ""
            try:
                import glob
                index_files = glob.glob(f"{self.rvc_exp_dir}/*.index")
                if index_files:
                    index_path = index_files[0]
            except Exception:
                pass

            # vc_single signature:
            #   vc_single(sid, input_audio_path, f0_up_key, f0_file,
            #             f0_method, file_index, file_index2, index_rate,
            #             filter_radius, resample_sr, rms_mix_rate, protect)
            result = self._vc.vc_single(
                0,                              # sid
                tmp_path,                       # input_audio_path
                self.rvc_f0_up_key,             # f0_up_key
                "",                             # f0_file
                self.rvc_f0_method,             # f0_method
                index_path,                     # file_index
                index_path,                     # file_index2
                self.rvc_index_rate,            # index_rate
                self.rvc_filter_radius,         # filter_radius
                0,                              # resample_sr (0 = use model's native sr)
                self.rvc_rms_mix_rate,          # rms_mix_rate
                self.rvc_protect,               # protect
            )

            # result is (info_string, (sample_rate, numpy_audio_int16))
            if isinstance(result, tuple) and len(result) == 2:
                info, audio_data = result
                if isinstance(audio_data, tuple) and len(audio_data) == 2:
                    out_sr, audio_out = audio_data
                    if audio_out is not None:
                        audio_float = audio_out.flatten().astype(np.float32) / 32768.0
                        logger.info(f"RVC: {info}")
                        return audio_float, out_sr

            logger.warning(f"RVC returned unexpected result, using Kokoro output: {result}")
            return audio_np, input_sr

        except Exception as exc:
            logger.error(f"RVC voice conversion failed: {exc}")
            return audio_np, input_sr

        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            # Restore working directory
            try:
                os.chdir(original_cwd)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # WAV writer
    # ------------------------------------------------------------------

    @staticmethod
    def _save_wav(
        audio_np, filepath: Path, sample_rate: int = 24000
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
