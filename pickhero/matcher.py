"""Note matching engine.

Compares detected audio notes against the tab timeline to produce
hit/close/miss feedback. No pygame dependency — pure logic.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from pickhero.audio.note_utils import semitone_distance
from pickhero.audio.input import TimestampedNote
from pickhero.tabs.timeline import NoteEvent, Timeline


class MatchType(Enum):
    PENDING = "pending"
    HIT = "hit"
    CLOSE = "close"
    WRONG = "wrong"
    MISS = "miss"


@dataclass
class MatchResult:
    """Result of matching a detected note against the timeline."""
    match_type: MatchType
    matched_events: list[NoteEvent] = field(default_factory=list)
    semitone_distance: int | None = None
    expected_midi: int | None = None
    detected_midi: int | None = None
    signed_pitch_error: int | None = None
    timing_error_ms: float | None = None
    confidence: float | None = None

    @property
    def timing_label(self) -> str | None:
        """Human-friendly timing direction for live feedback."""
        if self.timing_error_ms is None:
            return None
        if abs(self.timing_error_ms) <= 25.0:
            return "on time"
        return "late" if self.timing_error_ms > 0 else "early"


class NoteMatcher:
    """Matches detected audio notes against tab timeline events.

    Each NoteEvent in the timeline is tracked by state (PENDING -> HIT/CLOSE/MISS).
    Detected notes are compared to PENDING events within the timing window.
    """

    def __init__(
        self,
        timeline: Timeline,
        timing_window_ms: float = 100.0,
        audio_offset_ms: float = 0.0,
        chord_threshold_ms: float = 50.0,
        note_filter: Callable[[NoteEvent], bool] | None = None,
        chord_partial_credit: bool = True,
    ):
        self._timeline = timeline
        self._timing_window_ms = timing_window_ms
        self._audio_offset_ms = audio_offset_ms
        self._chord_threshold_ms = chord_threshold_ms
        self.note_filter = note_filter
        self.chord_partial_credit = chord_partial_credit

        # State per note event, keyed by (timestamp_ms, string)
        self._note_states: dict[tuple[float, int], MatchType] = {}

        # Statistics
        self.hits = 0
        self.close = 0
        self.misses = 0
        self.wrong_attempts = 0
        self._timing_errors_ms: list[float] = []

        # Per-measure statistics: {measure_idx: {"hits": n, "close": n, "misses": n}}
        self._measure_stats: dict[int, dict[str, int]] = defaultdict(
            lambda: {"hits": 0, "close": 0, "misses": 0}
        )

    @property
    def audio_offset_ms(self) -> float:
        return self._audio_offset_ms

    @audio_offset_ms.setter
    def audio_offset_ms(self, value: float) -> None:
        self._audio_offset_ms = value

    def _note_key(self, event: NoteEvent) -> tuple[float, int]:
        return (event.timestamp_ms, event.string)

    def _get_state(self, event: NoteEvent) -> MatchType:
        return self._note_states.get(self._note_key(event), MatchType.PENDING)

    def _set_state(self, event: NoteEvent, state: MatchType) -> None:
        self._note_states[self._note_key(event)] = state

    def _is_filtered(self, event: NoteEvent) -> bool:
        """Return True if this note should be excluded by the difficulty filter."""
        if self.note_filter is None:
            return False
        return not self.note_filter(event)

    def get_note_state(self, event: NoteEvent) -> MatchType:
        """Get the current match state of a timeline note."""
        return self._get_state(event)

    def _find_chord_siblings(self, event: NoteEvent) -> list[NoteEvent]:
        """Find notes within chord_threshold_ms of the given event."""
        return [
            n for n in self._timeline.get_active_notes_at_time(
                event.timestamp_ms, self._chord_threshold_ms
            )
            if abs(n.timestamp_ms - event.timestamp_ms) <= self._chord_threshold_ms
        ]

    def _record_match(self, event: NoteEvent, match_type: MatchType) -> None:
        """Record a match for a note, updating stats and measure stats."""
        self._set_state(event, match_type)
        if match_type == MatchType.HIT:
            self.hits += 1
            self._measure_stats[event.measure]["hits"] += 1
        elif match_type == MatchType.CLOSE:
            self.close += 1
            self._measure_stats[event.measure]["close"] += 1
        elif match_type == MatchType.MISS:
            self.misses += 1
            self._measure_stats[event.measure]["misses"] += 1

    def has_pending_notes_at(self, playback_ms: float) -> bool:
        """Return True if there are unmatched notes at or before playback_ms."""
        return self.pending_note_time_at(playback_ms) is not None

    def pending_note_time_at(self, playback_ms: float) -> float | None:
        """Return the earliest unmatched note time at the playhead, if any."""
        window_start = playback_ms - self._timing_window_ms
        candidates = self._timeline.get_notes_in_range(window_start, playback_ms + 1)
        for note in candidates:
            if self._is_filtered(note):
                continue
            if self._get_state(note) == MatchType.PENDING:
                return note.timestamp_ms
        return None

    def expected_midi_notes_at(
        self,
        playback_ms: float,
        lookahead_ms: float | None = None,
    ) -> tuple[int, ...]:
        """Return the earliest pending tab pitch group near the playhead.

        This small target set guides polyphonic arpeggio detection. It never
        exposes later notes at the same time, preventing a ringing string from
        stealing a future event.
        """
        target = self.expected_target_at(playback_ms, lookahead_ms)
        return target[1] if target is not None else ()

    def expected_target_at(
        self,
        playback_ms: float,
        lookahead_ms: float | None = None,
    ) -> tuple[float, tuple[int, ...]] | None:
        """Return (event timestamp, MIDI pitches) for the next pending group."""
        if lookahead_ms is None:
            lookahead_ms = self._timing_window_ms
        candidates = self._timeline.get_notes_in_range(
            playback_ms - self._timing_window_ms,
            playback_ms + max(0.0, lookahead_ms) + 0.001,
        )
        pending = [
            note for note in candidates
            if self._get_state(note) == MatchType.PENDING
            and not self._is_filtered(note)
        ]
        if not pending:
            return None
        earliest = min(note.timestamp_ms for note in pending)
        midi_notes = tuple(sorted({
            note.midi_note for note in pending
            if abs(note.timestamp_ms - earliest) <= self._chord_threshold_ms
        }))
        return (earliest, midi_notes)

    def _mark_missed_notes(self, playback_ms: float) -> list[MatchResult]:
        """Mark PENDING notes that have passed the timing window as MISS."""
        results = []
        cutoff = playback_ms - self._timing_window_ms
        if cutoff <= 0:
            return results

        # Check notes that should have been played by now
        candidates = self._timeline.get_notes_in_range(0, cutoff)
        for note in candidates:
            if self._is_filtered(note):
                continue
            if self._get_state(note) == MatchType.PENDING:
                self._record_match(note, MatchType.MISS)
                results.append(MatchResult(
                    match_type=MatchType.MISS,
                    matched_events=[note],
                    semitone_distance=None,
                    expected_midi=note.midi_note,
                ))
        return results

    def process_detected_notes(
        self,
        detected: list[TimestampedNote],
        playback_ms: float,
        allow_sustained: bool = False,
    ) -> list[MatchResult]:
        """Process detected notes against the timeline.

        Args:
            detected: Notes from AudioCapture.get_notes()
            playback_ms: Current playback position in the song
            allow_sustained: Accept confident pitch frames without a fresh
                onset. Used by Wait Mode after playback has frozen, where the
                player may already be sustaining the expected note.

        Returns:
            List of match results for this frame.
        """
        results = []

        # First, mark any notes that have passed the window as missed
        results.extend(self._mark_missed_notes(playback_ms))

        # Normal scoring uses picked onsets for accurate timing. Wait Mode may
        # also accept a sustained confident pitch so a missed onset does not
        # leave playback stuck forever.
        for ts_note in detected:
            if not ts_note.note.is_onset:
                # Sustained YIN output can be the previous ringing string.
                # Only the tab-guided detector has re-articulation protection.
                if not allow_sustained or ts_note.note.source != "guided":
                    continue

            adjusted_ms = ts_note.timestamp_ms + self._audio_offset_ms
            detected_midi = ts_note.note.midi_note

            # Match picked onsets to note onsets, not the full sustain range.
            # Using the sounding duration here lets a note struck hundreds of
            # milliseconds late count as correct for long notes.
            candidates = self._timeline.get_notes_in_range(
                adjusted_ms - self._timing_window_ms,
                adjusted_ms + self._timing_window_ms + 0.001,
            )

            # Filter to PENDING and non-filtered only
            pending = [
                n for n in candidates
                if self._get_state(n) == MatchType.PENDING and not self._is_filtered(n)
            ]
            if not pending:
                continue

            def effective_pitch_distance(note: NoteEvent) -> int:
                # Octaves are different frets/strings and must not count as
                # the same note. The old pitch-class shortcut made a ringing
                # lower arpeggio note complete an upcoming upper octave.
                return semitone_distance(detected_midi, note.midi_note)

            # Timing is the primary signal. This prevents a same-pitch future
            # note from stealing a detection from the note at the playhead.
            best = min(
                pending,
                key=lambda note: (
                    abs(adjusted_ms - note.timestamp_ms),
                    effective_pitch_distance(note),
                ),
            )
            best_dist = effective_pitch_distance(best)
            raw_pitch_error = detected_midi - best.midi_note
            timing_error_ms = adjusted_ms - best.timestamp_ms

            # Classify match
            if best_dist == 0:
                match_type = MatchType.HIT
            elif best_dist == 1:
                match_type = MatchType.CLOSE
            else:
                # Report the attempt but keep the expected event pending. The
                # player can still correct it, which is essential in Wait Mode.
                self.wrong_attempts += 1
                results.append(MatchResult(
                    match_type=MatchType.WRONG,
                    matched_events=[best],
                    semitone_distance=best_dist,
                    expected_midi=best.midi_note,
                    detected_midi=detected_midi,
                    signed_pitch_error=raw_pitch_error,
                    timing_error_ms=timing_error_ms,
                    confidence=ts_note.note.confidence,
                ))
                continue

            self._timing_errors_ms.append(timing_error_ms)

            # Chord handling
            siblings = self._find_chord_siblings(best)
            # Filter out excluded notes from siblings
            siblings = [s for s in siblings if not self._is_filtered(s)]

            if self.chord_partial_credit and len(siblings) > 1:
                # Partial credit mode: only mark the matched note
                matched_events = []
                if self._get_state(best) == MatchType.PENDING:
                    self._record_match(best, match_type)
                    matched_events.append(best)

                # Check if majority of chord is now matched
                total_in_chord = len(siblings)
                needed = math.ceil(total_in_chord / 2)
                matched_count = sum(
                    1 for s in siblings
                    if self._get_state(s) in (MatchType.HIT, MatchType.CLOSE)
                )
                if matched_count >= needed:
                    # Auto-complete remaining pending notes
                    for s in siblings:
                        if self._get_state(s) == MatchType.PENDING:
                            self._record_match(s, match_type)
                            matched_events.append(s)
            else:
                # Easy mode (old behavior): mark all chord siblings
                matched_events = []
                for sibling in siblings:
                    if self._get_state(sibling) == MatchType.PENDING:
                        self._record_match(sibling, match_type)
                        matched_events.append(sibling)

                # Ensure the best note itself is included
                if best not in matched_events:
                    if self._get_state(best) == MatchType.PENDING:
                        self._record_match(best, match_type)
                        matched_events.append(best)

            results.append(MatchResult(
                match_type=match_type,
                matched_events=matched_events,
                semitone_distance=best_dist,
                expected_midi=best.midi_note,
                detected_midi=detected_midi,
                signed_pitch_error=raw_pitch_error,
                timing_error_ms=timing_error_ms,
                confidence=ts_note.note.confidence,
            ))

        return results

    def get_statistics(self) -> dict:
        """Return current match statistics."""
        total = self.hits + self.close + self.misses
        accuracy = (self.hits / total * 100) if total > 0 else 0.0
        timing_errors = self._timing_errors_ms
        mean_abs_timing_error = (
            sum(abs(error) for error in timing_errors) / len(timing_errors)
            if timing_errors else None
        )
        mean_timing_error = (
            sum(timing_errors) / len(timing_errors)
            if timing_errors else None
        )
        return {
            "hits": self.hits,
            "close": self.close,
            "misses": self.misses,
            "wrong_attempts": self.wrong_attempts,
            "total": total,
            "accuracy_percent": accuracy,
            "mean_abs_timing_error_ms": mean_abs_timing_error,
            "mean_timing_error_ms": mean_timing_error,
            "early": sum(error < -25.0 for error in timing_errors),
            "on_time": sum(abs(error) <= 25.0 for error in timing_errors),
            "late": sum(error > 25.0 for error in timing_errors),
        }

    def get_weakest_sections(
        self, threshold: float = 0.6, min_length: int = 2
    ) -> list[tuple[int, int, float]]:
        """Find contiguous measures below accuracy threshold.

        Returns list of (start_measure, end_measure, accuracy) sorted by
        accuracy ascending. Only returns sections of at least min_length measures.
        """
        if not self._measure_stats:
            return []

        max_measure = max(self._measure_stats.keys())
        weak_runs: list[tuple[int, int, float]] = []
        run_start = None
        run_hits = 0
        run_total = 0

        for m in range(max_measure + 1):
            stats = self._measure_stats.get(m)
            if stats is None:
                # No notes in this measure — not weak, break any run
                if run_start is not None and (m - run_start) >= min_length:
                    acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0
                continue

            total = stats["hits"] + stats["close"] + stats["misses"]
            if total == 0:
                if run_start is not None and (m - run_start) >= min_length:
                    acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0
                continue

            acc = stats["hits"] / total
            if acc < threshold:
                if run_start is None:
                    run_start = m
                    run_hits = 0
                    run_total = 0
                run_hits += stats["hits"]
                run_total += total
            else:
                if run_start is not None and (m - run_start) >= min_length:
                    run_acc = run_hits / run_total if run_total > 0 else 0.0
                    weak_runs.append((run_start, m - 1, run_acc * 100))
                run_start = None
                run_hits = 0
                run_total = 0

        # Close any open run
        if run_start is not None and (max_measure + 1 - run_start) >= min_length:
            acc = run_hits / run_total if run_total > 0 else 0.0
            weak_runs.append((run_start, max_measure, acc * 100))

        # Sort by accuracy ascending (weakest first)
        weak_runs.sort(key=lambda x: x[2])
        return weak_runs

    def reset(self) -> None:
        """Clear all state. Call on seek/restart."""
        self._note_states.clear()
        self.hits = 0
        self.close = 0
        self.misses = 0
        self.wrong_attempts = 0
        self._timing_errors_ms.clear()
        self._measure_stats.clear()
