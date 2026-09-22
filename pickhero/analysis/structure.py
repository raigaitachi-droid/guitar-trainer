"""Symbolic song-structure and pedagogical phrase analysis.

This is the score side of the future fusion engine.  It deliberately produces
a complete, serialisable contract that an audio structure model can enrich
later without changing the playing-screen API.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

from pickhero.tabs.timeline import MeasureInfo, NoteEvent, Timeline


@dataclass(frozen=True)
class MeasureFeatures:
    """Compact musical and technical description of one measure."""

    index: int
    start_ms: float
    end_ms: float
    note_count: int
    onset_count: int
    pitch_center: float
    fret_center: float
    max_fret_span: int
    string_changes: int
    position_shifts: int
    chord_count: int
    longest_gap_ratio: float
    token_signature: tuple[str, ...]

    @property
    def duration_ms(self) -> float:
        return max(1.0, self.end_ms - self.start_ms)

    @property
    def density(self) -> float:
        return self.note_count / (self.duration_ms / 1000.0)


@dataclass(frozen=True)
class SongSection:
    """A playable, named region of a song."""

    id: str
    label: str
    family: str
    start_measure: int
    end_measure: int
    start_ms: float
    end_ms: float
    difficulty: int
    techniques: tuple[str, ...]
    confidence: float
    occurrence: int = 1
    occurrence_count: int = 1

    @property
    def measure_count(self) -> int:
        return self.end_measure - self.start_measure + 1


@dataclass(frozen=True)
class SongStructure:
    """Full symbolic analysis result, ready for UI or JSON persistence."""

    sections: tuple[SongSection, ...]
    measures: tuple[MeasureFeatures, ...]
    analyzer_version: str = "symbolic-v1"

    def section_at(self, playback_ms: float) -> SongSection | None:
        if not self.sections:
            return None
        for section in self.sections:
            if section.start_ms <= playback_ms < section.end_ms:
                return section
        if playback_ms >= self.sections[-1].end_ms:
            return self.sections[-1]
        return self.sections[0]

    def to_dict(self) -> dict:
        return {
            "analyzer_version": self.analyzer_version,
            "sections": [asdict(section) for section in self.sections],
            "measures": [asdict(measure) for measure in self.measures],
        }


def _measure_ranges(timeline: Timeline) -> list[MeasureInfo]:
    """Return usable ranges, repairing empty/zero-length imported measures."""
    source = timeline.measures
    max_note_measure = max((note.measure for note in timeline.notes), default=-1)
    count = max(len(source), max_note_measure + 1)
    if count <= 0:
        return []

    tempo = max(1, timeline.metadata.tempo)
    fallback_duration = 4.0 * 60_000.0 / tempo
    starts: list[float | None] = [None] * count
    ends: list[float | None] = [None] * count

    for measure in source:
        if 0 <= measure.index < count:
            starts[measure.index] = measure.start_ms
            if measure.end_ms > measure.start_ms:
                ends[measure.index] = measure.end_ms

    notes_by_measure: dict[int, list[NoteEvent]] = {}
    for note in timeline.notes:
        notes_by_measure.setdefault(note.measure, []).append(note)

    cursor = 0.0
    ranges: list[MeasureInfo] = []
    for index in range(count):
        notes = notes_by_measure.get(index, [])
        start = starts[index]
        if start is None or start < cursor - 1.0:
            start = cursor
        end = ends[index]
        if end is None or end <= start:
            next_known_start = next(
                (value for value in starts[index + 1:] if value is not None and value > start),
                None,
            )
            note_end = max((note.end_ms for note in notes), default=start)
            end = max(note_end, next_known_start or (start + fallback_duration))
        ranges.append(MeasureInfo(index=index, start_ms=start, end_ms=end))
        cursor = end
    return ranges


def _quantize(value: float, steps: int = 16) -> int:
    return max(0, min(steps, int(round(value * steps))))


def _features_for_measure(
    measure: MeasureInfo,
    notes: list[NoteEvent],
) -> MeasureFeatures:
    duration = max(1.0, measure.end_ms - measure.start_ms)
    ordered = sorted(notes, key=lambda note: (note.timestamp_ms, note.string, note.fret))
    onset_groups: dict[int, list[NoteEvent]] = {}
    tokens: list[str] = []
    first_pitch = ordered[0].midi_note if ordered else 0

    for note in ordered:
        rel = (note.timestamp_ms - measure.start_ms) / duration
        onset_bin = _quantize(rel)
        onset_groups.setdefault(onset_bin, []).append(note)
        dur_bin = max(1, _quantize(note.duration_ms / duration))
        interval = note.midi_note - first_pitch
        # Pitch intervals make transposed repetitions comparable, while string
        # and fret movement retain the guitar-specific technical identity.
        tokens.append(f"{onset_bin}:{dur_bin}:{interval}:{note.string}:{note.fret // 3}")

    distinct_onsets = sorted(onset_groups)
    string_changes = 0
    position_shifts = 0
    previous_string: int | None = None
    previous_fret: float | None = None
    chord_count = 0
    max_fret_span = 0

    for onset in distinct_onsets:
        group = onset_groups[onset]
        if len(group) > 1:
            chord_count += 1
            frets = [note.fret for note in group if note.fret > 0]
            if frets:
                max_fret_span = max(max_fret_span, max(frets) - min(frets))
        representative = group[0]
        if previous_string is not None and representative.string != previous_string:
            string_changes += 1
        if previous_fret is not None and abs(representative.fret - previous_fret) >= 5:
            position_shifts += 1
        previous_string = representative.string
        previous_fret = sum(note.fret for note in group) / len(group)

    gaps = []
    if distinct_onsets:
        positions = [0] + distinct_onsets + [16]
        gaps = [b - a for a, b in zip(positions, positions[1:])]

    return MeasureFeatures(
        index=measure.index,
        start_ms=measure.start_ms,
        end_ms=measure.end_ms,
        note_count=len(ordered),
        onset_count=len(distinct_onsets),
        pitch_center=(sum(note.midi_note for note in ordered) / len(ordered) if ordered else 0.0),
        fret_center=(sum(note.fret for note in ordered) / len(ordered) if ordered else 0.0),
        max_fret_span=max_fret_span,
        string_changes=string_changes,
        position_shifts=position_shifts,
        chord_count=chord_count,
        longest_gap_ratio=(max(gaps) / 16.0 if gaps else 1.0),
        token_signature=tuple(tokens),
    )


def _measure_similarity(left: MeasureFeatures, right: MeasureFeatures) -> float:
    if not left.token_signature and not right.token_signature:
        return 1.0
    if not left.token_signature or not right.token_signature:
        return 0.0
    token_score = SequenceMatcher(
        None, left.token_signature, right.token_signature, autojunk=False
    ).ratio()
    rhythm_left = tuple(tuple(token.split(":", 2)[:2]) for token in left.token_signature)
    rhythm_right = tuple(tuple(token.split(":", 2)[:2]) for token in right.token_signature)
    rhythm_score = SequenceMatcher(None, rhythm_left, rhythm_right, autojunk=False).ratio()
    density_delta = abs(left.density - right.density) / max(left.density, right.density, 1.0)
    return max(0.0, min(1.0, token_score * 0.62 + rhythm_score * 0.28 + (1 - density_delta) * 0.10))


def _window_similarity(
    features: list[MeasureFeatures],
    start_a: int,
    start_b: int,
    length: int,
) -> float:
    scores = [
        _measure_similarity(features[start_a + offset], features[start_b + offset])
        for offset in range(length)
    ]
    return sum(scores) / len(scores)


def _boundary_strengths(features: list[MeasureFeatures]) -> list[float]:
    """Score plausible phrase boundaries between measures."""
    count = len(features)
    strengths = [0.0] * (count + 1)
    strengths[0] = strengths[count] = 10.0

    for boundary in range(1, count):
        left, right = features[boundary - 1], features[boundary]
        score = 0.0
        if boundary % 4 == 0:
            score += 1.6
        elif boundary % 2 == 0:
            score += 0.5
        if left.note_count == 0 or right.note_count == 0:
            score += 2.2
        density_delta = abs(left.density - right.density) / max(left.density, right.density, 1.0)
        score += min(1.8, density_delta * 2.0)
        if left.pitch_center and right.pitch_center:
            score += min(1.4, abs(left.pitch_center - right.pitch_center) / 8.0)
        score += min(1.0, left.longest_gap_ratio)
        strengths[boundary] = score

    # Cache pairwise measure similarity once. A naive nested window comparison
    # repeats SequenceMatcher millions of times on long scores.
    similarity_matrix = [[0.0] * count for _ in range(count)]
    for first in range(count):
        similarity_matrix[first][first] = 1.0
        for second in range(first + 1, count):
            value = _measure_similarity(features[first], features[second])
            similarity_matrix[first][second] = value
            similarity_matrix[second][first] = value

    # Repeated windows provide strong evidence for their own starts/ends.
    for first in range(count):
        for second in range(first + 2, count):
            running = 0.0
            for length in range(1, 9):
                if first + length > second or second + length > count:
                    break
                running += similarity_matrix[first + length - 1][second + length - 1]
                if length < 2:
                    continue
                similarity = running / length
                if similarity >= 0.80:
                    vote = 0.8 + (similarity - 0.80) * 4.0 + length * 0.06
                    for boundary in (first, first + length, second, second + length):
                        strengths[boundary] += vote

    # Highly repetitive material can vote for every possible boundary. Capping
    # keeps musical 4/8-bar phrasing stronger than accidental 2-bar fragments.
    for boundary in range(1, count):
        strengths[boundary] = min(6.0, strengths[boundary])
    return strengths


def _choose_boundaries(features: list[MeasureFeatures]) -> list[int]:
    """Select 2–8 measure phrases with dynamic programming."""
    count = len(features)
    if count <= 4:
        return [0, count]
    strengths = _boundary_strengths(features)
    best = [-math.inf] * (count + 1)
    previous = [-1] * (count + 1)
    best[0] = 0.0

    for end in range(1, count + 1):
        for length in range(2, 9):
            start = end - length
            if start < 0 or best[start] == -math.inf:
                continue
            length_bonus = {
                2: -1.5,
                3: 0.4,
                4: 2.5,
                5: 0.0,
                6: 0.8,
                7: 0.0,
                8: 2.0,
            }[length]
            candidate = best[start] + strengths[end] + length_bonus - 4.0
            if candidate > best[end]:
                best[end] = candidate
                previous[end] = start

    if previous[count] < 0:
        return list(range(0, count, 4)) + [count]

    boundaries = [count]
    cursor = count
    while cursor > 0:
        cursor = previous[cursor]
        if cursor < 0:
            break
        boundaries.append(cursor)
    return sorted(set(boundaries + [0]))


def _section_similarity(
    features: list[MeasureFeatures],
    left: tuple[int, int],
    right: tuple[int, int],
) -> float:
    left_start, left_end = left
    right_start, right_end = right
    if left_end - left_start != right_end - right_start:
        return 0.0
    return _window_similarity(features, left_start, right_start, left_end - left_start)


def _difficulty_and_techniques(
    measures: list[MeasureFeatures],
) -> tuple[int, tuple[str, ...]]:
    if not measures:
        return 0, ("rest",)
    duration_s = sum(measure.duration_ms for measure in measures) / 1000.0
    notes = sum(measure.note_count for measure in measures)
    onsets = sum(measure.onset_count for measure in measures)
    strings = sum(measure.string_changes for measure in measures)
    shifts = sum(measure.position_shifts for measure in measures)
    chords = sum(measure.chord_count for measure in measures)
    stretch = max((measure.max_fret_span for measure in measures), default=0)
    density = notes / max(0.1, duration_s)
    change_ratio = strings / max(1, onsets - 1)

    score = 10.0 + min(34.0, density * 5.5)
    score += min(20.0, change_ratio * 30.0)
    score += min(16.0, shifts * 3.0)
    score += min(12.0, chords * 1.5)
    score += min(8.0, stretch * 1.5)

    techniques: list[str] = []
    if density >= 4.0:
        techniques.append("speed and synchronization")
    if change_ratio >= 0.45:
        techniques.append("string crossing")
    if shifts >= 2:
        techniques.append("position shifts")
    if chords >= max(2, len(measures)):
        techniques.append("chord changes")
    if stretch >= 4:
        techniques.append("left-hand stretch")
    if not techniques:
        techniques.append("timing and clean notes")
    return int(max(1, min(100, round(score)))), tuple(techniques[:3])


def analyze_structure(timeline: Timeline) -> SongStructure:
    """Analyze a tab timeline into repeated, pedagogically useful sections."""
    ranges = _measure_ranges(timeline)
    if not ranges:
        return SongStructure(sections=(), measures=())

    notes_by_measure: dict[int, list[NoteEvent]] = {}
    for note in timeline.notes:
        notes_by_measure.setdefault(note.measure, []).append(note)
    features = [
        _features_for_measure(measure, notes_by_measure.get(measure.index, []))
        for measure in ranges
    ]
    boundaries = _choose_boundaries(features)
    raw_sections = list(zip(boundaries, boundaries[1:]))

    families: list[int] = []
    family_members: list[list[int]] = []
    for index, candidate in enumerate(raw_sections):
        assigned = None
        for family_index, members in enumerate(family_members):
            reference = raw_sections[members[0]]
            if _section_similarity(features, candidate, reference) >= 0.76:
                assigned = family_index
                break
        if assigned is None:
            assigned = len(family_members)
            family_members.append([])
        family_members[assigned].append(index)
        families.append(assigned)

    # Repeated families receive A/B/C names ordered by first appearance.
    repeated_families = [
        family for family, members in enumerate(family_members) if len(members) > 1
    ]
    family_names = {
        family: chr(ord("A") + index) if index < 26 else f"P{index + 1}"
        for index, family in enumerate(repeated_families)
    }

    sections: list[SongSection] = []
    occurrence_seen: dict[int, int] = {}
    for index, (start, end) in enumerate(raw_sections):
        family = families[index]
        occurrence_seen[family] = occurrence_seen.get(family, 0) + 1
        count = len(family_members[family])
        family_name = family_names.get(family, "")

        if family_name:
            label = f"Theme {family_name}"
        elif index == 0:
            label = "Intro"
        elif index == len(raw_sections) - 1:
            label = "Ending"
        else:
            difficulty_probe, _ = _difficulty_and_techniques(features[start:end])
            label = "Solo" if difficulty_probe >= 64 else f"Phrase {index + 1}"

        difficulty, techniques = _difficulty_and_techniques(features[start:end])
        repeated = count > 1
        confidence = 0.88 if repeated else (0.72 if label in ("Intro", "Ending") else 0.60)
        sections.append(SongSection(
            id=f"section-{index + 1}",
            label=label,
            family=family_name or f"unique-{index + 1}",
            start_measure=start,
            end_measure=end - 1,
            start_ms=ranges[start].start_ms,
            end_ms=ranges[end - 1].end_ms,
            difficulty=difficulty,
            techniques=techniques,
            confidence=confidence,
            occurrence=occurrence_seen[family],
            occurrence_count=count,
        ))

    return SongStructure(sections=tuple(sections), measures=tuple(features))
