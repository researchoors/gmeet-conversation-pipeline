"""Comprehensive tests for gmeet_pipeline.tts.local.LocalTTS."""

import asyncio
import os
import struct
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gmeet_pipeline.tts.local import LocalTTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts(tmp_path, **overrides):
    """Create a LocalTTS with sensible test defaults."""
    defaults = dict(
        rvc_model_path=str(tmp_path / "models" / "hank.pth"),
        rvc_exp_dir=str(tmp_path / "exp"),
        rvc_repo_dir=str(tmp_path / "rvc_repo"),
        audio_dir=str(tmp_path / "audio"),
    )
    defaults.update(overrides)
    return LocalTTS(**defaults)


def _sine_wave(freq=440, sr=24000, duration=0.05):
    """Return a float32 numpy sine wave."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t)


def _read_wav_header(filepath):
    """Parse a minimal WAV header and return (sample_rate, num_channels, bits_per_sample, data_size, raw_data)."""
    with open(filepath, "rb") as f:
        riff = f.read(4)
        assert riff == b"RIFF"
        file_size = struct.unpack("<I", f.read(4))[0]
        wave = f.read(4)
        assert wave == b"WAVE"
        # fmt chunk
        fmt_id = f.read(4)
        assert fmt_id == b"fmt "
        fmt_size = struct.unpack("<I", f.read(4))[0]
        fmt_data = f.read(fmt_size)
        audio_format = struct.unpack("<H", fmt_data[0:2])[0]
        num_channels = struct.unpack("<H", fmt_data[2:4])[0]
        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
        byte_rate = struct.unpack("<I", fmt_data[8:12])[0]
        block_align = struct.unpack("<H", fmt_data[12:14])[0]
        bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
        # data chunk
        data_id = f.read(4)
        assert data_id == b"data"
        data_size = struct.unpack("<I", f.read(4))[0]
        raw_data = f.read(data_size)
    return sample_rate, num_channels, bits_per_sample, data_size, raw_data


# ===========================================================================
# 1. Constructor tests
# ===========================================================================


class TestConstructor:
    """Test LocalTTS constructor defaults and parameter assignment."""

    def test_default_params(self, tmp_path):
        tts = _make_tts(tmp_path)
        assert tts.rvc_f0_method == "rmvpe"
        assert tts.rvc_f0_up_key == 0
        assert tts.rvc_index_rate == 0.0
        assert tts.rvc_filter_radius == 3
        assert tts.rvc_rms_mix_rate == 0.25
        assert tts.rvc_protect == 0.33
        assert tts.kokoro_voice == "af_heart"

    def test_custom_params(self, tmp_path):
        tts = _make_tts(
            tmp_path,
            rvc_f0_method="pm",
            rvc_f0_up_key=5,
            rvc_index_rate=0.75,
            rvc_filter_radius=5,
            rvc_rms_mix_rate=0.5,
            rvc_protect=0.5,
            kokoro_voice="bf_emma",
        )
        assert tts.rvc_f0_method == "pm"
        assert tts.rvc_f0_up_key == 5
        assert tts.rvc_index_rate == 0.75
        assert tts.rvc_filter_radius == 5
        assert tts.rvc_rms_mix_rate == 0.5
        assert tts.rvc_protect == 0.5
        assert tts.kokoro_voice == "bf_emma"

    def test_rvc_paths_assigned(self, tmp_path):
        model_path = str(tmp_path / "models" / "hank.pth")
        exp_dir = str(tmp_path / "exp")
        repo_dir = str(tmp_path / "rvc_repo")
        tts = _make_tts(tmp_path)
        assert tts.rvc_model_path == model_path
        assert tts.rvc_exp_dir == exp_dir
        assert tts.rvc_repo_dir == repo_dir

    def test_lazy_fields_none(self, tmp_path):
        tts = _make_tts(tmp_path)
        assert tts._kokoro_pipeline is None
        assert tts._vc is None
        assert tts._rvc_loaded is False
        assert tts._initialized is False

    def test_audio_dir_defaults_to_home(self, tmp_path):
        tts = LocalTTS(
            rvc_model_path="",
            rvc_exp_dir="",
            rvc_repo_dir="",
            audio_dir="",  # empty string → default
        )
        expected = Path.home() / ".hermes" / "audio_cache" / "meeting_tts"
        assert tts.audio_dir == expected

    def test_audio_dir_custom(self, tmp_path):
        custom = tmp_path / "custom_audio"
        tts = _make_tts(tmp_path, audio_dir=str(custom))
        assert tts.audio_dir == custom


# ===========================================================================
# 12. audio_dir is created on init
# ===========================================================================


class TestAudioDirCreation:
    """Test that audio_dir is created on construction."""

    def test_audio_dir_created(self, tmp_path):
        audio_dir = tmp_path / "nested" / "audio"
        assert not audio_dir.exists()
        tts = _make_tts(tmp_path, audio_dir=str(audio_dir))
        assert audio_dir.exists()
        assert audio_dir.is_dir()


# ===========================================================================
# 2 & 3. _save_wav tests
# ===========================================================================


class TestSaveWav:
    """Test LocalTTS._save_wav static method."""

    def test_save_wav_known_audio(self, tmp_path):
        """Write a known sine wave and verify WAV header and data length."""
        audio = _sine_wave(freq=440, sr=24000, duration=0.05)
        filepath = tmp_path / "test.wav"
        LocalTTS._save_wav(audio, filepath, sample_rate=24000)

        sr, nch, bps, data_size, raw_data = _read_wav_header(filepath)
        assert sr == 24000
        assert nch == 1
        assert bps == 16
        # data_size should match: num_samples * num_channels * bytes_per_sample
        assert data_size == len(audio) * 2  # 16-bit = 2 bytes per sample
        assert len(raw_data) == data_size

    def test_save_wav_clipping_values_above_one(self, tmp_path):
        """Values > 1.0 should be clipped to max int16 (32767)."""
        audio = np.array([2.0, 1.5, 1.0, 0.5, 0.0], dtype=np.float32)
        filepath = tmp_path / "clip_high.wav"
        LocalTTS._save_wav(audio, filepath, sample_rate=24000)

        _, _, _, _, raw_data = _read_wav_header(filepath)
        # Parse int16 samples
        samples = np.frombuffer(raw_data, dtype=np.int16)
        # Clipped values should be 32767
        assert samples[0] == 32767
        assert samples[1] == 32767
        # 1.0 → 32767 (rounding of 32767.0)
        assert samples[2] == 32767
        # 0.5 → ~16383
        assert abs(samples[3] - 16383) <= 1
        # 0.0 → 0
        assert samples[4] == 0

    def test_save_wav_clipping_values_below_minus_one(self, tmp_path):
        """Values < -1.0 should be clipped to min int16 (-32768)."""
        audio = np.array([-2.0, -1.5, -1.0, -0.5, 0.0], dtype=np.float32)
        filepath = tmp_path / "clip_low.wav"
        LocalTTS._save_wav(audio, filepath, sample_rate=24000)

        _, _, _, _, raw_data = _read_wav_header(filepath)
        samples = np.frombuffer(raw_data, dtype=np.int16)
        # Clipped values should be -32768
        assert samples[0] == -32768
        assert samples[1] == -32768
        # -1.0 → -32768 (rounding of -32768.0, but np.clip(-1.0*32767, ...) = -32767)
        # Actually np.clip(-1.0 * 32767, -32768, 32767) = -32767
        assert samples[2] == -32767
        # -0.5 → ~-16383
        assert abs(samples[3] - (-16383)) <= 1
        # 0.0 → 0
        assert samples[4] == 0

    def test_save_wav_custom_sample_rate(self, tmp_path):
        """Test _save_wav with a non-default sample rate."""
        audio = _sine_wave(freq=440, sr=16000, duration=0.05)
        filepath = tmp_path / "test_16k.wav"
        LocalTTS._save_wav(audio, filepath, sample_rate=16000)

        sr, _, _, _, _ = _read_wav_header(filepath)
        assert sr == 16000


# ===========================================================================
# 4 & 5. _ensure_init with mocked import failures
# ===========================================================================


class TestEnsureInit:
    """Test LocalTTS._ensure_init() lazy loading and failure handling."""

    def test_ensure_init_not_initialized_by_default(self, tmp_path):
        tts = _make_tts(tmp_path)
        assert tts._initialized is False

    def test_ensure_init_sets_initialized(self, tmp_path):
        """Even when both Kokoro and RVC fail, _initialized should be True."""
        tts = _make_tts(tmp_path)
        with patch.dict("sys.modules", {}):
            tts._ensure_init()
        assert tts._initialized is True

    def test_ensure_init_kokoro_import_failure(self, tmp_path):
        """When Kokoro fails to import, _kokoro_pipeline should stay None."""
        tts = _make_tts(tmp_path)
        with patch.dict("sys.modules", {}):
            # Force kokoro to not exist
            tts._ensure_init()
        assert tts._kokoro_pipeline is None

    def test_ensure_init_kokoro_load_failure(self, tmp_path):
        """When Kokoro KPipeline() raises, _kokoro_pipeline should be None."""
        tts = _make_tts(tmp_path)
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline.side_effect = RuntimeError("Kokoro init failed")

        with patch.dict("sys.modules", {"kokoro": mock_kokoro, "spacy": MagicMock(), "spacy.util": MagicMock()}):
            tts._ensure_init()

        assert tts._kokoro_pipeline is None
        assert tts._initialized is True

    def test_ensure_init_rvc_import_failure(self, tmp_path):
        """When RVC modules fail to import, _vc should be None and _rvc_loaded False."""
        tts = _make_tts(tmp_path)
        mock_kokoro = MagicMock()
        mock_pipeline = MagicMock()
        mock_kokoro.KPipeline.return_value = mock_pipeline

        # Make configs.config import fail
        with patch.dict("sys.modules", {
            "kokoro": mock_kokoro,
            "spacy": MagicMock(),
            "spacy.util": MagicMock(),
            "configs": MagicMock(),
        }):
            # configs.config import will fail because Config doesn't exist
            def raise_import_error(name, *args):
                if name == "configs.config":
                    raise ImportError("No configs")
                return MagicMock()
            with patch("builtins.__import__", side_effect=raise_import_error):
                tts._ensure_init()

        assert tts._vc is None
        assert tts._rvc_loaded is False
        assert tts._initialized is True

    def test_ensure_init_idempotent(self, tmp_path):
        """_ensure_init should only run once even if called multiple times."""
        tts = _make_tts(tmp_path)
        tts._initialized = True
        tts._ensure_init()  # Should be a no-op
        # Still True, no crash
        assert tts._initialized is True

    def test_ensure_init_restores_cwd_on_rvc_failure(self, tmp_path):
        """If RVC init fails, cwd should still be restored."""
        tts = _make_tts(tmp_path)
        original_cwd = os.getcwd()
        mock_kokoro = MagicMock()
        mock_kokoro.KPipeline.return_value = MagicMock()

        # Make chdir happen then RVC fail
        with patch.dict("sys.modules", {
            "kokoro": mock_kokoro,
            "spacy": MagicMock(),
            "spacy.util": MagicMock(),
        }):
            # Force RVC import to fail after chdir
            with patch("builtins.__import__", side_effect=lambda *a, **kw: (_ for _ in ()).throw(ImportError("RVC missing")) if "configs" in str(a) else MagicMock()):
                tts._ensure_init()

        assert os.getcwd() == original_cwd


# ===========================================================================
# 6. generate returns None when _kokoro_pipeline is None
# ===========================================================================


class TestGenerateNoPipeline:
    """Test generate() when kokoro pipeline is unavailable."""

    async def test_generate_returns_none_when_no_pipeline(self, tmp_path):
        tts = _make_tts(tmp_path)
        tts._initialized = True
        tts._kokoro_pipeline = None

        result = await tts.generate("Hello", "bot1")
        assert result is None


# ===========================================================================
# 7. _apply_rvc returns original audio when _vc is None
# ===========================================================================


class TestApplyRvcNoVc:
    """Test _apply_rvc when RVC VC is not loaded."""

    def test_apply_rvc_returns_original_when_vc_none(self, tmp_path):
        tts = _make_tts(tmp_path)
        tts._vc = None
        # Ensure rvc_repo_dir exists so chdir doesn't fail
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)
        audio = _sine_wave()
        result_audio, result_sr = tts._apply_rvc(audio, 24000)
        # Should return the original audio unchanged
        assert result_sr == 24000
        np.testing.assert_array_equal(result_audio, audio)


# ===========================================================================
# 8. _apply_rvc cwd is restored even on exception
# ===========================================================================


class TestApplyRvcCwdRestore:
    """Test that _apply_rvc restores cwd even on exception."""

    def test_cwd_restored_on_exception(self, tmp_path):
        tts = _make_tts(tmp_path)
        # Create a mock VC that raises
        mock_vc = MagicMock()
        mock_vc.vc_single.side_effect = RuntimeError("RVC crashed")
        tts._vc = mock_vc
        tts._rvc_loaded = True
        tts.rvc_repo_dir = str(tmp_path / "rvc_repo")

        # Ensure rvc_repo_dir exists so chdir works
        (tmp_path / "rvc_repo").mkdir(parents=True, exist_ok=True)

        original_cwd = os.getcwd()
        audio = _sine_wave()
        result_audio, result_sr = tts._apply_rvc(audio, 24000)

        # cwd should be restored
        assert os.getcwd() == original_cwd
        # Should return original audio on failure
        np.testing.assert_array_equal(result_audio, audio)
        assert result_sr == 24000


# ===========================================================================
# 9. _apply_rvc calls vc.vc_single with correct params
# ===========================================================================


class TestApplyRvcParams:
    """Test _apply_rvc calls vc.vc_single with correct parameters."""

    def test_vc_single_called_with_correct_params(self, tmp_path):
        tts = _make_tts(
            tmp_path,
            rvc_f0_method="pm",
            rvc_f0_up_key=3,
            rvc_index_rate=0.5,
            rvc_filter_radius=5,
            rvc_rms_mix_rate=0.4,
            rvc_protect=0.25,
        )
        # Ensure rvc_repo_dir exists so chdir works
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)

        # Create a mock VC
        mock_vc = MagicMock()
        # vc_single returns (info_string, (sample_rate, audio_int16))
        out_audio = np.zeros(100, dtype=np.int16)
        mock_vc.vc_single.return_value = ("success", (40000, out_audio))
        tts._vc = mock_vc
        tts._rvc_loaded = True

        audio = _sine_wave()
        tts._apply_rvc(audio, 24000)

        # Check vc_single was called
        assert mock_vc.vc_single.called
        call_args = mock_vc.vc_single.call_args
        args = call_args[0]

        # vc_single(sid, input_audio_path, f0_up_key, f0_file, f0_method,
        #           file_index, file_index2, index_rate, filter_radius,
        #           resample_sr, rms_mix_rate, protect)
        assert args[0] == 0                    # sid
        assert args[2] == 3                    # f0_up_key
        assert args[3] == ""                   # f0_file
        assert args[4] == "pm"                 # f0_method
        assert args[7] == 0.5                  # index_rate
        assert args[8] == 5                    # filter_radius
        assert args[9] == 0                    # resample_sr
        assert args[10] == 0.4                 # rms_mix_rate
        assert args[11] == 0.25                # protect

    def test_apply_rvc_converts_result_to_float32(self, tmp_path):
        """Test that _apply_rvc converts int16 output to float32."""
        tts = _make_tts(tmp_path)
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)
        mock_vc = MagicMock()
        # Simulate RVC output: int16 audio at 40kHz
        out_int16 = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        mock_vc.vc_single.return_value = ("RVC ok", (40000, out_int16))
        tts._vc = mock_vc
        tts._rvc_loaded = True

        audio = _sine_wave()
        result_audio, result_sr = tts._apply_rvc(audio, 24000)

        assert result_sr == 40000
        # Should be float32
        assert result_audio.dtype == np.float32
        # Check conversion: int16 / 32768.0
        expected = out_int16.flatten().astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result_audio, expected)

    def test_apply_rvc_handles_unexpected_result_format(self, tmp_path):
        """When RVC returns unexpected format, fall back to original audio."""
        tts = _make_tts(tmp_path)
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)
        mock_vc = MagicMock()
        mock_vc.vc_single.return_value = "unexpected"
        tts._vc = mock_vc
        tts._rvc_loaded = True

        audio = _sine_wave()
        result_audio, result_sr = tts._apply_rvc(audio, 24000)

        # Should fall back to original audio
        assert result_sr == 24000
        np.testing.assert_array_equal(result_audio, audio)

    def test_apply_rvc_cleans_up_temp_file(self, tmp_path):
        """_apply_rvc should clean up its temp WAV file."""
        tts = _make_tts(tmp_path)
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)
        mock_vc = MagicMock()
        out_audio = np.zeros(100, dtype=np.int16)
        mock_vc.vc_single.return_value = ("ok", (40000, out_audio))
        tts._vc = mock_vc
        tts._rvc_loaded = True

        audio = _sine_wave()

        # Track temp files before and after
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_tempdir = tempfile.gettempdir()
            tempfile.tempdir = tmpdir
            try:
                tts._apply_rvc(audio, 24000)
                # Temp dir should be empty (temp wav cleaned up)
                remaining = list(Path(tmpdir).glob("*.wav"))
                assert len(remaining) == 0
            finally:
                tempfile.tempdir = orig_tempdir


# ===========================================================================
# 10. _generate_sync concatenates Kokoro segments and saves WAV
# ===========================================================================


class TestGenerateSync:
    """Test _generate_sync method."""

    def test_generate_sync_concatenates_segments(self, tmp_path):
        """Test that _generate_sync concatenates multiple Kokoro audio segments."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        # Mock kokoro pipeline to yield multiple segments
        seg1 = np.ones(100, dtype=np.float32) * 0.1
        seg2 = np.ones(200, dtype=np.float32) * 0.2

        # Kokoro pipeline yields (graph, phonemes, audio) tuples
        # Audio objects need .cpu().numpy()
        mock_audio1 = MagicMock()
        mock_audio1.cpu.return_value.numpy.return_value = seg1
        mock_audio2 = MagicMock()
        mock_audio2.cpu.return_value.numpy.return_value = seg2

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            (MagicMock(), "hello", mock_audio1),
            (MagicMock(), "world", mock_audio2),
        ]
        tts._kokoro_pipeline = mock_pipeline
        tts._rvc_loaded = False  # Skip RVC

        result = tts._generate_sync("Hello world", "bot1")
        assert result is not None
        assert result.endswith(".wav")
        assert result.startswith("tts_")

        # Verify the WAV file was created
        wav_path = tts.audio_dir / result
        assert wav_path.exists()

        # Read back and verify concatenated length
        sr, nch, bps, data_size, raw_data = _read_wav_header(wav_path)
        assert sr == 24000
        expected_samples = len(seg1) + len(seg2)
        assert data_size == expected_samples * 2  # 16-bit

    def test_generate_sync_skips_none_audio(self, tmp_path):
        """Segments with None audio should be skipped."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        seg1 = np.ones(100, dtype=np.float32) * 0.1
        mock_audio1 = MagicMock()
        mock_audio1.cpu.return_value.numpy.return_value = seg1

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            (MagicMock(), "hello", mock_audio1),
            (MagicMock(), "world", None),  # None audio — should be skipped
        ]
        tts._kokoro_pipeline = mock_pipeline
        tts._rvc_loaded = False

        result = tts._generate_sync("Hello world", "bot1")
        assert result is not None

        wav_path = tts.audio_dir / result
        sr, nch, bps, data_size, raw_data = _read_wav_header(wav_path)
        # Only seg1 should be in the output
        assert data_size == len(seg1) * 2

    def test_generate_sync_handles_plain_array_audio(self, tmp_path):
        """Audio without .cpu() attribute should be handled as plain array."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        seg1 = np.ones(100, dtype=np.float32) * 0.1
        # No .cpu() attribute — plain numpy array
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            (MagicMock(), "hello", seg1),
        ]
        tts._kokoro_pipeline = mock_pipeline
        tts._rvc_loaded = False

        result = tts._generate_sync("Hello", "bot1")
        assert result is not None

        wav_path = tts.audio_dir / result
        assert wav_path.exists()


# ===========================================================================
# 11. _generate_sync returns None when Kokoro produces no audio
# ===========================================================================


class TestGenerateSyncNoAudio:
    """Test _generate_sync when Kokoro produces no audio."""

    def test_returns_none_when_no_audio_segments(self, tmp_path):
        """If Kokoro pipeline yields no audio segments, return None."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = []  # No segments at all
        tts._kokoro_pipeline = mock_pipeline

        result = tts._generate_sync("Hello", "bot1")
        assert result is None

    def test_returns_none_when_all_segments_none(self, tmp_path):
        """If all segments have None audio, return None."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            (MagicMock(), "hello", None),
            (MagicMock(), "world", None),
        ]
        tts._kokoro_pipeline = mock_pipeline

        result = tts._generate_sync("Hello world", "bot1")
        assert result is None

    def test_returns_none_on_exception(self, tmp_path):
        """If Kokoro pipeline raises, _generate_sync returns None."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("Kokoro crashed")
        tts._kokoro_pipeline = mock_pipeline

        result = tts._generate_sync("Hello", "bot1")
        assert result is None


# ===========================================================================
# Integration-style: generate() calls _generate_sync via thread pool
# ===========================================================================


class TestGenerateIntegration:
    """Test generate() end-to-end with mocked pipeline."""

    async def test_generate_returns_filename(self, tmp_path):
        """generate() should return a WAV filename on success."""
        tts = _make_tts(tmp_path)
        tts._initialized = True

        seg = np.ones(100, dtype=np.float32) * 0.1
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [(MagicMock(), "hi", seg)]
        tts._kokoro_pipeline = mock_pipeline
        tts._rvc_loaded = False

        result = await tts.generate("Hello", "bot1")
        assert result is not None
        assert result.endswith(".wav")
        assert (tts.audio_dir / result).exists()

    async def test_generate_with_rvc(self, tmp_path):
        """generate() with RVC loaded should call _apply_rvc."""
        tts = _make_tts(tmp_path)
        tts._initialized = True
        Path(tts.rvc_repo_dir).mkdir(parents=True, exist_ok=True)

        seg = np.ones(100, dtype=np.float32) * 0.1
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [(MagicMock(), "hi", seg)]
        tts._kokoro_pipeline = mock_pipeline
        tts._rvc_loaded = True

        mock_vc = MagicMock()
        out_audio = np.zeros(100, dtype=np.int16)
        mock_vc.vc_single.return_value = ("ok", (40000, out_audio))
        tts._vc = mock_vc

        result = await tts.generate("Hello", "bot1")
        assert result is not None
        assert result.endswith(".wav")
