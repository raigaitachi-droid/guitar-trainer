"""Tests for symbolic song structure analysis."""

from pickhero.analysis.structure import analyze_structure
from pickhero.tabs.timeline import MeasureInfo, NoteEvent, SongMetadata, Timeline


def _timeline(patterns: list[list[tuple[float, int, int, int]]]) -> Timeline:
    """Build 4/4 measures from (beat, midi, string, fret) patterns."""
    measure_ms = 2000.0
    notes = []
    measures = []
    for measure_index, pattern in enumerate(patterns):
        start = measure_index * measure_ms
        measures.append(MeasureInfo(measure_index, start, start + measure_ms))
        for beat, midi, string, fret in pattern:
            notes.append(NoteEvent(
                timestamp_ms=start + beat * 500.0,
                duration_ms=300.0,
                midi_note=midi,
                string=string,
                fret=fret,
                measure=measure_index,
            ))
    return Timeline(notes, SongMetadata(title="Structure Test", tempo=120), measures)


RIFF_A = [
    (0.0, 64, 1, 0),
    (1.0, 67, 2, 3),
    (2.0, 69, 2, 5),
    (3.0, 67, 2, 3),
]
RIFF_B = [
    (0.0, 52, 5, 3),
    (0.5, 55, 4, 5),
    (1.0, 57, 3, 7),
    (1.5, 60, 2, 8),
    (2.0, 64, 1, 12),
    (2.5, 60, 2, 8),
    (3.0, 57, 3, 7),
    (3.5, 55, 4, 5),
]


def test_repeated_phrases_share_a_family():
    timeline = _timeline([RIFF_A] * 4 + [RIFF_B] * 4 + [RIFF_A] * 4)
    structure = analyze_structure(timeline)

    repeated = [section for section in structure.sections if section.occurrence_count > 1]
    assert repeated
    assert any(section.label == "Theme A" for section in repeated)
    theme_a = [section for section in repeated if section.label == "Theme A"]
    assert len(theme_a) >= 2
    assert {section.family for section in theme_a} == {"A"}


def test_section_contract_contains_pedagogical_data():
    timeline = _timeline([RIFF_A] * 4 + [RIFF_B] * 4)
    structure = analyze_structure(timeline)

    assert structure.sections
    for section in structure.sections:
        assert 1 <= section.difficulty <= 100
        assert section.techniques
        assert section.start_ms < section.end_ms
        assert section.start_measure <= section.end_measure


def test_dense_string_crossing_phrase_is_harder():
    easy = analyze_structure(_timeline([RIFF_A] * 4)).sections[0]
    hard = analyze_structure(_timeline([RIFF_B] * 4)).sections[0]

    assert hard.difficulty > easy.difficulty
    assert "string crossing" in hard.techniques


def test_structure_is_serializable_and_queryable():
    structure = analyze_structure(_timeline([RIFF_A] * 4 + [RIFF_B] * 4))
    data = structure.to_dict()

    assert data["analyzer_version"] == "symbolic-v1"
    assert len(data["sections"]) == len(structure.sections)
    first = structure.sections[0]
    assert structure.section_at(first.start_ms) == first


def test_empty_timeline_returns_empty_structure():
    structure = analyze_structure(Timeline([]))
    assert structure.sections == ()
    assert structure.measures == ()
