"""Hardware-free tests for low-latency audio stream lifecycle."""

from types import SimpleNamespace

import pytest

from pickhero.audio import input as audio_input
from pickhero.config import Config


class _FakePortAudioError(Exception):
    pass


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.active = False
        self.latency = 0.012
        self.start_calls = 0
        self.stop_calls = 0
        self.close_calls = 0

    def start(self):
        self.start_calls += 1
        self.active = True

    def stop(self):
        self.stop_calls += 1
        self.active = False

    def close(self):
        self.close_calls += 1


class _FakeSoundDevice:
    PortAudioError = _FakePortAudioError

    def __init__(self):
        self.streams = []

    def query_devices(self, device=None, kind=None):
        return {
            "default_samplerate": 48_000.0,
            "max_input_channels": 1,
        }

    def InputStream(self, **kwargs):
        stream = _FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


@pytest.fixture
def fake_sd(monkeypatch):
    fake = _FakeSoundDevice()
    monkeypatch.setattr(audio_input, "sd", fake)
    return fake


def test_active_monitor_stream_is_rearmed_without_reopening(fake_sd):
    capture = audio_input.AudioCapture(Config())

    capture.start()
    first_stream = capture._stream
    capture.start()

    assert capture._stream is first_stream
    assert len(fake_sd.streams) == 1
    assert first_stream.start_calls == 1
    assert first_stream.stop_calls == 0
    assert first_stream.close_calls == 0


def test_low_latency_stream_and_diagnostics(fake_sd):
    capture = audio_input.AudioCapture(Config())
    capture.start()

    stream = fake_sd.streams[0]
    assert stream.kwargs["latency"] == "low"
    assert stream.kwargs["blocksize"] == 256

    diagnostics = capture.get_latency_diagnostics()
    assert diagnostics["sample_rate"] == 48_000
    assert diagnostics["block_size"] == 256
    assert diagnostics["device_ms"] == pytest.approx(12.0)
    assert diagnostics["block_ms"] == pytest.approx(1000 * 256 / 48_000)
    assert diagnostics["detector_ms"] == pytest.approx(1000 * 2048 / 48_000)

