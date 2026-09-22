"""Pitch and onset detection using aubio.

Wraps aubio's YIN pitch detector and onset detector.
Processes audio buffers and returns detected notes.
"""

from dataclasses import dataclass

import aubio
import numpy as np

from pickhero.audio.note_utils import freq_to_midi, midi_to_name, is_in_guitar_range


@dataclass
class DetectedNote:
    """A single detected pitch event."""
    midi_note: int
    frequency: float
    confidence: float
    name: str
    is_onset: bool  # True if a new note strike was detected
    source: str = "yin"  # "yin" or tab-guided harmonic detector


class PitchDetector:
    """Real-time pitch and onset detection for guitar audio.

    Processes audio buffers (hop_size samples each) and returns
    detected notes with confidence values.
    """

    def __init__(
        self,
        buf_size: int = 2048,
        hop_size: int = 512,
        sample_rate: int = 44100,
        confidence_threshold: float = 0.8,
        onset_threshold: float = 0.3,
        noise_gate_db: float = -60.0,
        calibration: dict | None = None,
    ):
        self.buf_size = buf_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.confidence_threshold = confidence_threshold
        self.onset_threshold = onset_threshold
        self.noise_gate_db = noise_gate_db
        self.last_signal_db: float = -120.0
        self.last_freq: float = 0.0
        self.last_confidence: float = 0.0
        self.last_source: str = "yin"

        # Tab-guided polyphonic assistance. YIN deliberately estimates one
        # fundamental, which is unreliable when previous arpeggio strings are
        # still ringing. The rolling spectrum lets us test the *known next tab
        # note* and its harmonics without trying to transcribe every voice.
        self._expected_midis: tuple[int, ...] = ()
        self._expected_token = None
        self._expected_baseline: dict[int, float] = {}
        self._expected_age_frames = 0
        self._analysis_buffer = np.zeros(buf_size, dtype=np.float32)
        self._analysis_samples = 0
        self._window = np.hanning(buf_size).astype(np.float32)
        self._guided_hits = 0

        # Octave jump protection
        self._prev_freq: float = 0.0
        self._calibration = calibration

        # Pitch detector (YIN algorithm)
        self._pitch = aubio.pitch("yin", buf_size, hop_size, sample_rate)
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(confidence_threshold)

        # Onset detector
        self._onset = aubio.onset("default", buf_size, hop_size, sample_rate)
        self._onset.set_threshold(onset_threshold)

    def process(self, audio_buffer: np.ndarray) -> DetectedNote | None:
        """Process a single audio buffer (hop_size float32 samples).

        Args:
            audio_buffer: 1D numpy array of float32 samples, length == hop_size.

        Returns:
            DetectedNote if a confident pitch was detected, None otherwise.
        """
        # Ensure correct format for aubio
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # Maintain our own rolling window for tab-guided harmonic evidence.
        hop = len(audio_buffer)
        if hop >= self.buf_size:
            self._analysis_buffer[:] = audio_buffer[-self.buf_size:]
        else:
            self._analysis_buffer[:-hop] = self._analysis_buffer[hop:]
            self._analysis_buffer[-hop:] = audio_buffer
        self._analysis_samples = min(self.buf_size, self._analysis_samples + hop)

        # Check noise gate (RMS level)
        rms = np.sqrt(np.mean(audio_buffer ** 2))
        if rms > 0:
            db = 20 * np.log10(rms)
        else:
            db = -120.0

        self.last_signal_db = db

        # Feed both aubio engines continuously, including quiet frames. This
        # clears their history between plucks and improves onset detection.
        freq = float(self._pitch(audio_buffer)[0])
        confidence = float(self._pitch.get_confidence())
        is_onset = bool(self._onset(audio_buffer))

        if db < self.noise_gate_db:
            self.last_freq = 0.0
            self.last_confidence = 0.0
            return None

        guided = self._detect_expected_pitch(is_onset)
        if guided is not None:
            self.last_freq = guided.frequency
            self.last_confidence = guided.confidence
            self.last_source = "guided"
            self._prev_freq = guided.frequency
            self._guided_hits += 1
            return guided

        # Correct octave jumps before exposing values
        if freq > 0:
            freq = self._correct_octave_jump(freq, confidence, is_onset)

        # Store values for tuner (after octave correction)
        self.last_freq = freq
        self.last_confidence = confidence
        self.last_source = "yin"

        # Filter: need minimum confidence and valid frequency
        if confidence < self.confidence_threshold or freq <= 0:
            return None

        midi_note = freq_to_midi(freq)
        if not is_in_guitar_range(midi_note):
            return None

        return DetectedNote(
            midi_note=midi_note,
            frequency=freq,
            confidence=confidence,
            name=midi_to_name(midi_note),
            is_onset=is_onset,
            source="yin",
        )

    def _harmonic_score(self, midi_note: int, magnitude: np.ndarray,
                        n_fft: int) -> tuple[float, int, float]:
        """Return (weighted score, strong harmonic count, fundamental score)."""
        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        ratios: list[float] = []
        weights: list[float] = []
        for harmonic in range(1, 6):
            harmonic_freq = freq * harmonic
            if harmonic_freq >= self.sample_rate * 0.45:
                break
            center = int(round(harmonic_freq * n_fft / self.sample_rate))
            if center < 2 or center + 2 >= len(magnitude):
                continue
            peak = float(np.max(magnitude[center - 2:center + 3]))
            # With 4x zero-padding the Hann main lobe spans several bins; use
            # a farther ring as the local spectral floor.
            lo = magnitude[max(1, center - 32):max(1, center - 12)]
            hi = magnitude[min(len(magnitude), center + 13):
                           min(len(magnitude), center + 33)]
            surroundings = np.concatenate((lo, hi))
            floor = float(np.median(surroundings)) if surroundings.size else 0.0
            ratio = peak / max(floor, 1e-8)
            ratios.append(min(ratio, 30.0))
            weights.append(1.0 / np.sqrt(harmonic))

        if not ratios:
            return (0.0, 0, 0.0)
        score = float(np.average(ratios, weights=weights))
        strong = sum(1 for ratio in ratios if ratio >= 3.0)
        return (score, strong, ratios[0])

    def _detect_expected_pitch(self, is_onset: bool) -> DetectedNote | None:
        """Detect the next tab note inside an overlapping arpeggio spectrum."""
        if not self._expected_midis or self._analysis_samples < self.buf_size // 2:
            return None

        n_fft = 1
        while n_fft < self.buf_size * 4:
            n_fft <<= 1
        spectrum = np.fft.rfft(self._analysis_buffer * self._window, n=n_fft)
        magnitude = np.abs(spectrum)

        best: tuple[float, int, int, float] | None = None
        for midi_note in self._expected_midis:
            score, strong, fundamental = self._harmonic_score(
                midi_note, magnitude, n_fft
            )
            candidate = (score, strong, midi_note, fundamental)
            if best is None or candidate[0] > best[0]:
                best = candidate

        if best is None:
            return None
        score, strong, midi_note, fundamental = best

        # Compare against neighbouring semitones to reject broadband attacks
        # and room noise. A real string should form a local harmonic maximum.
        neighbour_scores = []
        for neighbour in (midi_note - 2, midi_note - 1, midi_note + 1, midi_note + 2):
            neighbour_score, _, _ = self._harmonic_score(
                neighbour, magnitude, n_fft
            )
            neighbour_scores.append(neighbour_score)
        contrast = score / max(max(neighbour_scores, default=0.0), 1e-6)

        baseline = self._expected_baseline.get(midi_note)
        if baseline is None:
            self._expected_baseline[midi_note] = max(score, 1e-6)
            baseline = max(score, 1e-6)
        growth = score / max(baseline, 1e-6)
        self._expected_age_frames += 1

        has_evidence = (
            (strong >= 2 and score >= 3.2 and contrast >= 1.05)
            or (fundamental >= 8.0 and score >= 3.8 and contrast >= 1.08)
        )
        if not has_evidence:
            # Follow a decaying noise/ringing floor, but never chase a new
            # attack upward. This keeps the detector armed for the next pick.
            self._expected_baseline[midi_note] = min(
                baseline,
                baseline * 0.995 + score * 0.005,
            )
            return None

        # Each tab event must be re-articulated. This prevents a ringing lower
        # octave, or a repeated same pitch, from auto-completing the next note.
        # Aubio onset is preferred; spectral growth is the fallback when its
        # transient detector misses a soft finger-picked attack.
        if not is_onset and growth < 1.18:
            return None

        freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
        guided_confidence = min(0.99, 0.62 + (score - 3.2) * 0.04)
        return DetectedNote(
            midi_note=midi_note,
            frequency=freq,
            confidence=guided_confidence,
            name=midi_to_name(midi_note),
            is_onset=is_onset,
            source="guided",
        )

    def set_expected_notes(self, midi_notes, target_token=None) -> None:
        """Set the next pending tab pitches used by the guided detector."""
        expected = tuple(sorted({
            int(note) for note in midi_notes if is_in_guitar_range(int(note))
        }))
        token = target_token if target_token is not None else expected
        if expected != self._expected_midis or token != self._expected_token:
            self._expected_midis = expected
            self._expected_token = token
            self._expected_baseline = {}
            self._expected_age_frames = 0

    @property
    def guided_hits(self) -> int:
        return self._guided_hits

    def _correct_octave_jump(self, freq: float, confidence: float,
                             is_onset: bool = False) -> float:
        """Suppress octave jumps caused by harmonic detection.

        If the new frequency is ~2x or ~0.5x the previous, and confidence
        isn't very high, prefer the previous frequency (likely the fundamental).
        When calibration data is available, also check if freq/2 matches a
        known open-string fundamental.
        """
        corrected = freq

        # A fresh string attack is allowed to be a real octave leap. The old
        # guard treated common arpeggio shapes as harmonic mistakes.
        if is_onset:
            self._prev_freq = freq
            return freq

        # Calibration-based correction: if freq/2 is near a calibrated string,
        # prefer freq/2 (the fundamental was likely the intended note)
        if self._calibration and freq > 0:
            cal_strings = self._calibration.get("strings", {})
            half_freq = freq / 2.0
            for cal in cal_strings.values():
                cal_freq = cal.get("frequency", 0)
                if cal_freq > 0:
                    ratio = half_freq / cal_freq
                    # Within ±1 semitone of a calibrated fundamental
                    if 0.944 < ratio < 1.06:
                        corrected = half_freq
                        self._prev_freq = corrected
                        return corrected

        # Generic ratio-based correction
        if self._prev_freq > 0 and confidence < 0.95:
            ratio = freq / self._prev_freq
            if 1.95 <= ratio <= 2.05:
                # One octave up — prefer previous (fundamental)
                corrected = self._prev_freq
            elif 0.48 <= ratio <= 0.52:
                # One octave down — prefer previous
                corrected = self._prev_freq

        self._prev_freq = corrected
        return corrected

    def set_noise_gate_db(self, db: float) -> None:
        """Update the noise gate threshold (dB). Takes effect on next process() call."""
        self.noise_gate_db = db

    def reset(self):
        """Reset detector state. Call when starting a new song/session."""
        self._prev_freq = 0.0
        self._analysis_buffer.fill(0.0)
        self._analysis_samples = 0
        self._expected_baseline = {}
        self._expected_age_frames = 0
        self.last_source = "yin"

        # Re-create detectors to clear internal state
        self._pitch = aubio.pitch(
            "yin", self.buf_size, self.hop_size, self.sample_rate
        )
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(self.confidence_threshold)

        self._onset = aubio.onset(
            "default", self.buf_size, self.hop_size, self.sample_rate
        )
        self._onset.set_threshold(self.onset_threshold)
