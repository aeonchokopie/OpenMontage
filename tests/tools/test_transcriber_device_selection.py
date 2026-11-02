import sys
from types import SimpleNamespace

from tools.analysis.transcriber import Transcriber


class _Info:
    language = "en"
    duration = 1.0


def test_transcriber_uses_ctranslate2_cuda_without_torch(monkeypatch, tmp_path) -> None:
    devices = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            devices.append((device, compute_type))

        def transcribe(self, *args, **kwargs):
            return iter(()), _Info()

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(
            get_cuda_device_count=lambda: 1,
            get_supported_compute_types=lambda device: {"float16", "float32"},
        ),
    )
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"fake")

    result = Transcriber().execute({"input_path": str(input_path), "output_dir": str(tmp_path)})

    assert result.success, result.error
    assert devices == [("cuda", "float16")]
    assert result.data["device"] == "cuda"


def test_transcriber_falls_back_when_cuda_fails_during_iteration(monkeypatch, tmp_path) -> None:
    devices = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            self.device = device
            devices.append((device, compute_type))

        def transcribe(self, *args, **kwargs):
            if self.device == "cuda":
                def broken_iterator():
                    raise RuntimeError("cublas64_12.dll not found")
                    yield

                return broken_iterator(), _Info()
            return iter(()), _Info()

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(
            get_cuda_device_count=lambda: 1,
            get_supported_compute_types=lambda device: {"float16"},
        ),
    )
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"fake")

    result = Transcriber().execute({"input_path": str(input_path), "output_dir": str(tmp_path)})

    assert result.success, result.error
    assert devices == [("cuda", "float16"), ("cpu", "int8")]
    assert result.data["device"] == "cpu"
    assert "cublas64_12.dll" in result.data["gpu_fallback_reason"]


def _no_cuda(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ctranslate2",
        SimpleNamespace(get_cuda_device_count=lambda: 0, get_supported_compute_types=lambda device: set()),
    )


def test_transcriber_passes_ffmpeg_decoded_array_not_path(monkeypatch, tmp_path) -> None:
    """faster-whisper's PyAV decoder breaks on PyAV 15+ (metadata_errors kwarg);
    a decodable file must reach the model as a numpy array decoded by ffmpeg."""
    import shutil
    import subprocess

    import numpy as np
    import pytest

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    received = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            pass

        def transcribe(self, audio, **kwargs):
            received.append(audio)
            return iter(()), _Info()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    _no_cuda(monkeypatch)
    input_path = tmp_path / "tone.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(input_path)],
        check=True,
    )

    result = Transcriber().execute({"input_path": str(input_path), "output_dir": str(tmp_path)})

    assert result.success, result.error
    assert isinstance(received[0], np.ndarray)
    assert received[0].dtype == np.float32
    assert abs(len(received[0]) - 16000) < 400  # ~1 s at 16 kHz
    assert result.data["audio_decoder"] == "ffmpeg"


def test_transcriber_falls_back_to_path_when_ffmpeg_cannot_decode(monkeypatch, tmp_path) -> None:
    received = []

    class FakeWhisperModel:
        def __init__(self, model_size, *, device, compute_type):
            pass

        def transcribe(self, audio, **kwargs):
            received.append(audio)
            return iter(()), _Info()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    _no_cuda(monkeypatch)
    input_path = tmp_path / "audio.wav"
    input_path.write_bytes(b"not audio")

    result = Transcriber().execute({"input_path": str(input_path), "output_dir": str(tmp_path)})

    assert result.success, result.error
    assert received == [str(input_path)]
    assert result.data["audio_decoder"] == "faster_whisper"
