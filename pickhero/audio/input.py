"""Audio input capture using sounddevice.

Runs a sounddevice InputStream that feeds audio buffers to the pitch detector.
Detected notes are pushed to a thread-safe queue for consumption by the main thread.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import sounddevice as sd
except OSError:
    # Importing sounddevice fails when the native PortAudio runtime is absent
    # (for example in CI). Keep pure matching/tab features importable and give
    # the user a focused error only when audio capture is actually requested.
    sd = None

from pickhero.audio.detector import PitchDetector, DetectedNote
from pickhero.config import Config


@dataclass
class TimestampedNote:
    """A detected note with a timestamp (ms from session start)."""
    note: DetectedNote
    timestamp_ms: float


class AudioCapture:
    """Captures audio from an input device and runs pitch detection.

    Detected notes are pushed to `note_queue` for consumption by other threads.
    The sounddevice callback runs in a separate thread automatically.
    """

    def __init__(self, config: Config | None = None):
        if config is None:
            config = Config()
        self.config = config
        ac = config.audio

        calibration = getattr(config, 'calibration', None) or None
        # Built-in laptop microphones commonly expose 48 kHz only.  Asking
        # PortAudio for the old hard-coded 44.1 kHz rate can either fail or
        # produce silence on Windows, so use the selected device's native
        # rate whenever it is available.
        sample_rate = int(round(ac.sample_rate))
        if sd is not None:
            try:
                device_info = sd.query_devices(ac.device_index, "input")
                native_rate = int(round(float(device_info["default_samplerate"])))
                if native_rate > 0:
                    sample_rate = native_rate
            except (sd.PortAudioError, KeyError, TypeError, ValueError):
                pass
        self._sample_rate = sample_rate

        self.detector = PitchDetector(
            buf_size=ac.buf_size,
            hop_size=ac.hop_size,
            sample_rate=sample_rate,
            confidence_threshold=ac.confidence_threshold,
            onset_threshold=ac.onset_threshold,
            noise_gate_db=ac.noise_gate_db,
            calibration=calibration if calibration else None,
        )
        self.note_queue: queue.Queue[TimestampedNote] = queue.Queue()
        self._detector_lock = threading.Lock()
        self._session_generation: int = 0
        self._stream: Any | None = None
        self._start_time: float = 0.0
        self._signal_db: float = -120.0
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self._last_callback_time: float = 0.0
        self._callback_interval_ms: float = 0.0
        self._callback_blocksize: int = ac.hop_size
        self._overflow_count: int = 0
        self._reported_input_latency_ms: float = 0.0
        # YIN works on a rolling analysis window. Its pitch describes roughly
        # the centre of that window, not the instant when Python receives the
        # result, so remove half a window from musical timing timestamps.
        self._analysis_latency_ms = (ac.buf_size / sample_rate) * 500.0

    def _capture_timestamp_ms(
        self,
        callback_elapsed_ms: float,
        frames: int,
        chunk_end_frame: int,
        input_latency_ms: float,
    ) -> float:
        """Estimate when the analysed sound reached the microphone.

        Callback wall time includes the device buffer, unprocessed samples at
        the tail of a block, and the detector's rolling window. Removing those
        sources avoids systematically labelling on-time playing as late.
        """
        tail_frames = max(0, frames - chunk_end_frame)
        tail_ms = (tail_frames / self._sample_rate) * 1000.0
        timestamp = (
            callback_elapsed_ms
            - max(0.0, input_latency_ms)
            - tail_ms
            - self._analysis_latency_ms
        )
        return max(0.0, timestamp)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Sounddevice callback — runs in audio thread."""
        # PortAudio may flag a recoverable overflow. The buffer still contains
        # useful microphone data, so process it instead of making the meter
        # appear permanently dead.
        callback_now = time.perf_counter()
        with self._detector_lock:
            session_generation = self._session_generation
            session_start = self._start_time
        if self._last_callback_time > 0:
            interval_ms = (callback_now - self._last_callback_time) * 1000.0
            if self._callback_interval_ms <= 0:
                self._callback_interval_ms = interval_ms
            else:
                self._callback_interval_ms = (
                    self._callback_interval_ms * 0.85 + interval_ms * 0.15
                )
        self._last_callback_time = callback_now
        if status:
            self._overflow_count += 1
        callback_elapsed_ms = (callback_now - session_start) * 1000.0

        # PortAudio provides the actual ADC-to-callback delay for this block.
        # Some host APIs omit it, in which case use the stream's reported
        # latency captured at startup.
        input_latency_ms = self._reported_input_latency_ms
        try:
            adc_time = float(time_info.inputBufferAdcTime)
            current_time = float(time_info.currentTime)
            measured_ms = (current_time - adc_time) * 1000.0
            if 0.0 <= measured_ms <= 500.0:
                input_latency_ms = measured_ms
        except (AttributeError, TypeError, ValueError):
            pass

        # indata shape: (frames, channels) — take first channel
        mono = indata[:, 0].copy()

        # Process in hop_size chunks
        hop = self.detector.hop_size
        for i in range(0, len(mono) - hop + 1, hop):
            chunk = mono[i:i + hop]
            # Resetting the session clock can happen while the Windows audio
            # callback is running. Serialize aubio access and discard a block
            # from the old timeline instead of leaking it into a seek/loop.
            with self._detector_lock:
                if session_generation != self._session_generation:
                    return
                result = self.detector.process(chunk)
                self._signal_db = self.detector.last_signal_db
                self._tuner_freq = self.detector.last_freq
                self._tuner_confidence = self.detector.last_confidence
            if result is not None:
                timestamp_ms = self._capture_timestamp_ms(
                    callback_elapsed_ms,
                    frames,
                    i + hop,
                    input_latency_ms,
                )
                self.note_queue.put(TimestampedNote(note=result, timestamp_ms=timestamp_ms))

    def _reset_session_clock(self) -> None:
        """Reset scoring timestamps without closing the Windows device."""
        with self._detector_lock:
            self.detector.reset()
            while not self.note_queue.empty():
                try:
                    self.note_queue.get_nowait()
                except queue.Empty:
                    break
            self._start_time = time.perf_counter()
            self._session_generation += 1

    def start(self):
        """Start or re-arm audio capture while keeping an open stream warm."""
        if sd is None:
            raise RuntimeError(
                "Audio capture is unavailable because PortAudio could not be loaded. "
                "Install the PortAudio runtime or use the packaged Windows build."
            )
        ac = self.config.audio
        # Monitoring normally starts before playback. Re-arm its clock and
        # detector in-place so Play, Pause, seek, loop, and count-in do not pay
        # the Windows device-open penalty again.
        self._reset_session_clock()
        if self._stream is not None:
            try:
                if self._stream.active:
                    return
            except (AttributeError, RuntimeError):
                pass
            self.stop()

        self._last_callback_time = 0.0
        self._callback_interval_ms = 0.0
        # Prefer PortAudio's low-latency settings and a 256-frame callback.
        # Some Windows drivers reject these hints, so fall back without
        # failing the microphone entirely.
        attempts = (
            (ac.hop_size, "low"),
            (max(512, ac.hop_size), "low"),
            (max(512, ac.hop_size), None),
        )
        last_error: Exception | None = None
        for blocksize, latency in attempts:
            kwargs = dict(
                device=ac.device_index,
                channels=1,
                samplerate=self._sample_rate,
                blocksize=blocksize,
                dtype="float32",
                callback=self._audio_callback,
            )
            if latency is not None:
                kwargs["latency"] = latency
            try:
                self._stream = sd.InputStream(**kwargs)
                self._callback_blocksize = blocksize
                break
            except (sd.PortAudioError, ValueError) as exc:
                last_error = exc
                self._stream = None
        if self._stream is None:
            raise RuntimeError(f"Could not open low-latency microphone: {last_error}")
        try:
            self._reported_input_latency_ms = max(0.0, float(self._stream.latency) * 1000.0)
        except (AttributeError, TypeError, ValueError):
            self._reported_input_latency_ms = 0.0
        self._stream.start()

    def stop(self):
        """Stop audio capture."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def set_noise_gate_db(self, db: float) -> None:
        """Update the noise gate threshold on the detector.

        Thread-safe: single float attribute write is atomic under the GIL.
        """
        self.detector.set_noise_gate_db(db)

    def set_expected_notes(self, midi_notes, target_token=None) -> None:
        """Update tab pitches used by the arpeggio-aware detector."""
        with self._detector_lock:
            self.detector.set_expected_notes(midi_notes, target_token)

    def get_signal_db(self) -> float:
        """Return the latest signal level in dB. Thread-safe (single float read under GIL)."""
        return self._signal_db

    def get_tuner_data(self) -> tuple[float, float]:
        """Return (frequency_hz, confidence) for tuner display. Thread-safe."""
        return (self._tuner_freq, self._tuner_confidence)

    def get_latency_diagnostics(self) -> dict[str, float | int]:
        """Return live device/buffer/detector latency diagnostics.

        The estimate is deliberately decomposed: device latency is reported
        by PortAudio, callback latency comes from the selected block size, and
        detector latency is the YIN analysis window. This makes driver issues
        distinguishable from pitch-analysis limits on the user's PC.
        """
        block_ms = (self._callback_blocksize / self._sample_rate) * 1000.0
        detector_ms = (self.detector.buf_size / self._sample_rate) * 1000.0
        return {
            "sample_rate": self._sample_rate,
            "block_size": self._callback_blocksize,
            "device_ms": self._reported_input_latency_ms,
            "block_ms": block_ms,
            "detector_ms": detector_ms,
            "callback_ms": self._callback_interval_ms,
            "estimated_ms": self._reported_input_latency_ms + block_ms + detector_ms,
            "overflows": self._overflow_count,
            "guided_hits": self.detector.guided_hits,
        }

    def is_receiving_audio(self) -> bool:
        """Whether PortAudio has delivered a buffer recently."""
        return (
            self._last_callback_time > 0
            and time.perf_counter() - self._last_callback_time < 1.0
        )

    def get_notes(self) -> list[TimestampedNote]:
        """Drain all pending detected notes from the queue (non-blocking)."""
        notes = []
        while True:
            try:
                notes.append(self.note_queue.get_nowait())
            except queue.Empty:
                break
        return notes


def list_audio_devices() -> list[dict]:
    """List available audio input devices."""
    if sd is None:
        return []
    devices = sd.query_devices()
    inputs = []
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            inputs.append({
                "index": i,
                "name": dev["name"],
                "channels": dev["max_input_channels"],
                "sample_rate": dev["default_samplerate"],
            })
    return inputs


def validate_device_index(index: int | None) -> bool:
    """Check if a device index exists and has input channels.

    Returns True for None (system default) or a valid input device index.
    """
    if sd is None:
        return index is None
    if index is None:
        return True
    try:
        info = sd.query_devices(index)
        return info["max_input_channels"] > 0
    except (sd.PortAudioError, IndexError, ValueError):
        return False


def run_console_demo():
    """Interactive console demo for testing pitch detection.

    Lists audio devices, lets user pick one, then prints detected notes in real-time.
    """
    if sd is None:
        print("PortAudio is unavailable; audio devices cannot be opened.")
        return

    print("Available audio input devices:")
    print("-" * 50)
    devices = list_audio_devices()
    if not devices:
        print("No audio input devices found!")
        return

    for dev in devices:
        marker = " *" if dev["index"] == sd.default.device[0] else ""
        print(f"  [{dev['index']}] {dev['name']} ({dev['channels']}ch, {dev['sample_rate']:.0f}Hz){marker}")
    print()

    choice = input("Select device index (Enter for default): ").strip()
    config = Config()
    if choice:
        try:
            config.audio.device_index = int(choice)
        except ValueError:
            print("Invalid input, using default device.")

    print()
    print("Listening... play some notes! (Ctrl+C to stop)")
    print(f"  Config: buf={config.audio.buf_size}, hop={config.audio.hop_size}, "
          f"confidence>={config.audio.confidence_threshold}, noise_gate={config.audio.noise_gate_db}dB")
    print("-" * 60)
    print(f"{'Time':>8}  {'Note':>5}  {'MIDI':>4}  {'Freq':>8}  {'Conf':>5}  {'Onset'}")
    print("-" * 60)

    capture = AudioCapture(config)
    capture.start()

    last_note = ""
    try:
        while True:
            notes = capture.get_notes()
            for tn in notes:
                n = tn.note
                # Only print on onset or note change to reduce spam
                current = n.name
                if n.is_onset or current != last_note:
                    onset_marker = ">>>" if n.is_onset else "   "
                    print(f"{tn.timestamp_ms:7.0f}ms  {n.name:>5}  {n.midi_note:>4}  "
                          f"{n.frequency:7.1f}Hz  {n.confidence:.2f}  {onset_marker}")
                    last_note = current
            time.sleep(0.01)  # ~100Hz polling, avoid busy-wait
    finally:
        capture.stop()
