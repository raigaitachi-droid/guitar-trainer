"""Scrolling note display for the playing screen.

Renders 6 string lanes with notes scrolling right-to-left, synchronized
to a playback clock. Optionally captures audio and shows hit/miss feedback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pygame

from pickhero.analysis.structure import analyze_structure
from pickhero.audio.midi_playback import BackingTrack, MidiPlayer
from pickhero.config import Config
from pickhero.matcher import NoteMatcher
from pickhero.progress import ProgressTracker
from pickhero.tabs.timeline import NoteEvent, Timeline
from pickhero.audio.note_utils import freq_to_cents_deviation, midi_to_name
from pickhero.ui.colors import (
    STRING_COLORS,
    cycle_theme,
    dimmed,
    get_theme,
)
from pickhero.ui.feedback import FeedbackRenderer

# Layout constants
LANE_TOP_MARGIN = 154
LANE_BOTTOM_MARGIN = 92
MIN_NOTE_WIDTH_PX = 30
NOTE_HEIGHT_FRACTION = 0.52
NOTE_CORNER_RADIUS = 12
STRING_LABELS = ("e", "B", "G", "D", "A", "E")
SECTION_COLORS = (
    (78, 205, 188),
    (77, 145, 235),
    (168, 104, 232),
    (242, 159, 67),
    (239, 91, 116),
    (105, 190, 104),
)

# Left margin for notes that already passed the hit zone (ms)
LEFT_MARGIN_MS = 2000
# Right margin for notes not yet visible (ms)
RIGHT_MARGIN_MS = 500

# Difficulty filter: fret limit cycle values
FRET_LIMITS = [24, 12, 7, 5, 3]


def _get_font(name: str, size: int) -> pygame.font.Font:
    """Try to load a system font with fallbacks."""
    for family in (name, "Segoe UI", "Arial", "sans-serif"):
        font = pygame.font.SysFont(family, size)
        if font:
            return font
    return pygame.font.Font(None, size)


def format_time(ms: float) -> str:
    """Format milliseconds as M:SS."""
    total_seconds = max(0, int(ms / 1000))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


@dataclass
class _Layout:
    """Computed layout dimensions for current surface size."""

    screen_w: int
    screen_h: int
    lane_height: float
    note_h: float
    hit_zone_x: float
    usable_width: float
    pixels_per_ms: float
    visible_window_ms: float


class PlayingScreen:
    """Scrolling tab display with playback clock and optional audio matching."""

    def __init__(self, timeline: Timeline, visible_beats: int = 4,
                 hit_zone_fraction: float = 0.20, config: Config | None = None,
                 backing_track: BackingTrack | None = None,
                 progress_tracker: ProgressTracker | None = None,
                 song_key: str = ""):
        self._timeline = timeline
        self._visible_beats = visible_beats
        self._hit_zone_fraction = hit_zone_fraction
        self._config = config or Config()
        self._structure = analyze_structure(timeline)

        self._tempo_factor = max(0.5, min(1.0, self._config.tempo_factor))

        self._playback_ms: float = 0.0
        self._playing = False
        self._last_tick: float | None = None

        tempo = max(1, self._timeline.metadata.tempo)
        self._ms_per_beat = 60_000 / tempo
        self._visible_window_ms = 8000.0  # Fixed 8-second window

        # Count-in state
        count_in_beats = max(0, self._config.count_in_beats)
        self._count_in_ms = count_in_beats * self._ms_per_beat
        self._last_count_in_beat: int = -1

        # Audio matching
        self._audio_capture = None  # AudioCapture, created on demand
        self._audio_error: str | None = None
        self._audio_started_at: float = 0.0
        self._matcher: NoteMatcher | None = None
        self._feedback = FeedbackRenderer()
        self._audio_enabled = True
        self._noise_gate_db: float = self._config.audio.noise_gate_db

        # Loop state
        self._loop_start_ms: float | None = None
        self._loop_end_ms: float | None = None
        self._loop_enabled: bool = False

        # Progress tracking
        self._progress_tracker = progress_tracker
        self._song_key = song_key
        self._song_completed = False
        self._is_new_best = False
        self._recommendations: list[str] = []

        # MIDI backing track
        self._midi_player: MidiPlayer | None = None
        self._backing_muted = not self._config.backing_track_enabled
        if backing_track is not None and len(backing_track) > 0:
            self._init_midi_player(backing_track)

        # Difficulty filter
        self._max_fret: int = self._config.max_fret
        self._active_strings: list[bool] = list(self._config.active_strings)

        # Signal level meter
        self._signal_db: float = -120.0
        self._signal_db_smooth: float = -120.0

        # Tuner display
        self._tuner_freq: float = 0.0
        self._tuner_confidence: float = 0.0
        self._tuner_freq_smooth: float = 0.0
        self._tuner_displayed_note: int = -1
        self._tuner_note_stable_frames: int = 0

        # Chord partial credit mode
        self._chord_partial_credit: bool = self._config.chord_partial_credit

        # Help overlay
        self._show_help: bool = False
        self._show_latency_details: bool = False

        # Wait mode
        self._wait_mode: bool = self._config.wait_mode
        self._wait_mode_frozen: bool = False
        self._wait_mode_hold_ms: float | None = None

    def _note_passes_filter(self, note: NoteEvent) -> bool:
        """Check if a note passes the difficulty filter."""
        if note.fret > self._max_fret:
            return False
        if not self._active_strings[note.string - 1]:
            return False
        return True

    def _is_filter_active(self) -> bool:
        """Check if any difficulty filter is active."""
        return self._max_fret < 24 or not all(self._active_strings)

    def toggle_play(self) -> None:
        """Toggle play/pause. Restarts with count-in if at beginning or past end."""
        if self._playback_ms >= self._timeline.duration_ms and not self._playing:
            # Restart from beginning with count-in
            self._playback_ms = -self._count_in_ms if self._count_in_ms > 0 else 0.0
            self._last_count_in_beat = -1
            self._song_completed = False
            self._is_new_best = False
            self._weakest_sections = []
            self._recommendations = []
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
        elif self._playback_ms == 0.0 and not self._playing and self._count_in_ms > 0:
            # Starting from the very beginning — add count-in
            self._playback_ms = -self._count_in_ms
            self._last_count_in_beat = -1
            self._song_completed = False
            self._is_new_best = False
            self._weakest_sections = []
            self._recommendations = []
        self._playing = not self._playing
        if self._playing:
            self._last_tick = time.perf_counter()
            if self._audio_enabled:
                if self._playback_ms >= 0:
                    self._start_audio()
                else:
                    # Pre-warm the Windows device throughout count-in.
                    self._start_capture_only()
            if self._midi_player is not None:
                if self._playback_ms >= 0:
                    self._midi_player.seek(self._playback_ms)
        else:
            self._last_tick = None
            # Keep a lightweight monitoring stream active while paused so the
            # signal meter immediately proves whether Windows is supplying mic
            # audio.
            if self._audio_enabled:
                self._start_capture_only()
            else:
                self._stop_audio()
            if self._midi_player is not None:
                self._midi_player.pause()

    def seek(self, ms: float) -> None:
        """Seek to an absolute position in ms, clamped to [0, duration]."""
        self._playback_ms = max(0.0, min(ms, self._timeline.duration_ms))
        if self._matcher:
            self._matcher.reset()
        self._feedback.reset()
        if self._midi_player is not None:
            self._midi_player.seek(self._playback_ms)
        # Re-arm audio timing with the stream kept open.
        if self._audio_enabled and self._playing:
            self._start_audio()

    def is_playing(self) -> bool:
        return self._playing

    def set_tempo_factor(self, factor: float) -> None:
        """Set tempo scaling factor, clamped to [0.5, 1.0] and rounded to nearest 0.05."""
        factor = max(0.5, min(1.0, factor))
        factor = round(factor * 20) / 20  # round to nearest 0.05
        self._tempo_factor = factor
        self._config.tempo_factor = factor
        if self._matcher:
            self._matcher.reset()
        self._feedback.reset()

    def set_noise_gate_db(self, db: float) -> None:
        """Set noise gate threshold, clamped to [-80, -20] and rounded to int."""
        db = max(-80, min(-20, round(db)))
        self._noise_gate_db = db
        self._config.audio.noise_gate_db = db
        if self._audio_capture is not None:
            self._audio_capture.set_noise_gate_db(db)
        self._config.save()

    def update(self) -> None:
        """Advance playback clock by real elapsed time."""
        # Update signal level meter and tuner even when paused (so user can verify signal)
        if self._audio_capture is not None:
            raw_db = self._audio_capture.get_signal_db()
            self._signal_db = raw_db
            self._signal_db_smooth = self._signal_db_smooth * 0.7 + raw_db * 0.3
            freq, conf = self._audio_capture.get_tuner_data()
            self._tuner_freq = freq
            self._tuner_confidence = conf
            if freq > 0 and conf > 0.5:
                # Frequency jump guard: ignore wild jumps (> 50% change)
                if (self._tuner_freq_smooth > 0
                        and abs(freq - self._tuner_freq_smooth) / self._tuner_freq_smooth > 0.5):
                    # Wild jump — use very low alpha to dampen
                    alpha = 0.02
                else:
                    # Adaptive EMA: high confidence → faster, low → slower
                    alpha = 0.10 if conf > 0.8 else 0.03
                if self._tuner_freq_smooth > 0:
                    self._tuner_freq_smooth = self._tuner_freq_smooth * (1 - alpha) + freq * alpha
                else:
                    self._tuner_freq_smooth = freq
                # Note hysteresis: only change displayed note after 8 stable frames (~130ms)
                from pickhero.audio.note_utils import freq_to_midi
                candidate_note = freq_to_midi(self._tuner_freq_smooth)
                if candidate_note != self._tuner_displayed_note:
                    self._tuner_note_stable_frames += 1
                    if self._tuner_note_stable_frames >= 8:
                        self._tuner_displayed_note = candidate_note
                        self._tuner_note_stable_frames = 0
                else:
                    self._tuner_note_stable_frames = 0
            else:
                # Slow decay instead of instant reset
                self._tuner_freq_smooth *= 0.92
                if self._tuner_freq_smooth < 20.0:
                    self._tuner_freq_smooth = 0.0
                    self._tuner_displayed_note = -1
                    self._tuner_note_stable_frames = 0

            # An opened stream with no callbacks usually means Windows privacy
            # settings or the chosen input device is blocking microphone data.
            if (self._audio_started_at > 0
                    and time.perf_counter() - self._audio_started_at > 2.0
                    and not self._audio_capture.is_receiving_audio()):
                self._audio_error = "No microphone data from Windows"
            elif (self._audio_error == "No microphone data from Windows"
                    and self._audio_capture.is_receiving_audio()):
                self._audio_error = None

        if not self._playing:
            return

        now = time.perf_counter()
        prev_ms = self._playback_ms
        if self._last_tick is not None:
            elapsed_ms = (now - self._last_tick) * 1000.0 * self._tempo_factor
            self._playback_ms += elapsed_ms
        self._last_tick = now

        # Wait mode: freeze if there are pending notes the player hasn't hit yet
        if (self._wait_mode and self._audio_enabled
                and self._playback_ms >= 0 and self._matcher is not None):
            pending_time = self._matcher.pending_note_time_at(self._playback_ms)
            if pending_time is not None:
                if not self._wait_mode_frozen or self._wait_mode_hold_ms is None:
                    self._wait_mode_hold_ms = pending_time
                # Lock to one exact timestamp on every frame. Merely setting
                # the frozen flag still lets the normal clock creep forward.
                self._playback_ms = self._wait_mode_hold_ms
                self._last_tick = now
                self._wait_mode_frozen = True
                if self._midi_player is not None and not self._backing_muted:
                    self._midi_player.pause()
            elif self._wait_mode_frozen:
                self._wait_mode_frozen = False
                self._wait_mode_hold_ms = None
                if self._midi_player is not None and not self._backing_muted:
                    self._midi_player.seek(self._playback_ms)

        # Count-in: play metronome clicks and start audio/midi when crossing 0
        if prev_ms < 0:
            # Play count-in clicks at beat boundaries
            if self._count_in_ms > 0 and self._midi_player is not None:
                beat_index = int((self._count_in_ms + self._playback_ms) / self._ms_per_beat)
                if beat_index > self._last_count_in_beat:
                    self._midi_player.play_click(100)
                    self._last_count_in_beat = beat_index

            # Crossed from negative to non-negative — song starts
            if self._playback_ms >= 0:
                if self._audio_enabled:
                    self._start_audio()
                if self._midi_player is not None:
                    self._midi_player.seek(0)

        # Process audio matching (only during actual song, not count-in)
        if (self._playback_ms >= 0
                and self._audio_enabled
                and self._audio_capture is not None
                and self._matcher is not None):
            target = self._matcher.expected_target_at(self._playback_ms)
            if target is None:
                self._audio_capture.set_expected_notes(())
            else:
                target_time, target_midis = target
                self._audio_capture.set_expected_notes(target_midis, target_time)
            detected = self._audio_capture.get_notes()
            for d in detected:
                d.timestamp_ms *= self._tempo_factor
            # While frozen in wait mode, pin detected timestamps to the frozen
            # playback position so matching hits the notes at the hit zone,
            # not future notes that drift ahead as real time passes.
            if self._wait_mode_frozen and detected:
                pinned_ts = self._playback_ms - self._matcher.audio_offset_ms
                for d in detected:
                    d.timestamp_ms = pinned_ts
            results = self._matcher.process_detected_notes(
                detected,
                self._playback_ms,
                allow_sustained=self._wait_mode_frozen,
            )
            self._feedback.add_results(results, self._playback_ms)
            self._feedback.cleanup(self._playback_ms)

        # Advance MIDI backing track (only during actual song)
        if self._playback_ms >= 0 and self._midi_player is not None:
            self._midi_player.update(self._playback_ms)

        # Loop check — jump back to start marker when reaching end marker
        # (no count-in on loop)
        if (self._loop_enabled and self._loop_end_ms is not None
                and self._loop_start_ms is not None
                and self._playback_ms >= self._loop_end_ms):
            if self._midi_player is not None:
                self._midi_player.pause()
            self._playback_ms = self._loop_start_ms
            self._last_tick = time.perf_counter()
            if self._matcher:
                self._matcher.reset()
            self._feedback.reset()
            if self._midi_player is not None:
                self._midi_player.seek(self._loop_start_ms)
            if self._audio_enabled and self._playing:
                self._start_audio()
            return

        if self._playback_ms >= self._timeline.duration_ms:
            self._playback_ms = self._timeline.duration_ms
            self._playing = False
            self._last_tick = None
            if self._midi_player is not None:
                self._midi_player.pause()
            if self._audio_enabled:
                self._start_capture_only()
            else:
                self._stop_audio()

            if not self._song_completed:
                if (self._audio_enabled
                        and self._matcher is not None
                        and self._progress_tracker is not None
                        and self._song_key):
                    # Audio-scored completion
                    stats = self._matcher.get_statistics()
                    if stats["total"] > 0:
                        weakest = self._matcher.get_weakest_sections()
                        self._is_new_best, self._recommendations = (
                            self._progress_tracker.record_detailed_result(
                                self._song_key, stats,
                                weakest, self._tempo_factor,
                            )
                        )
                        self._weakest_sections = weakest
                        self._song_completed = True
                elif not self._audio_enabled:
                    # Auto-scroll (passive) completion
                    self._weakest_sections = []
                    self._song_completed = True

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Handle input. Returns 'menu' to go back, else None."""
        if event.type != pygame.KEYDOWN:
            return None

        if event.key == pygame.K_SPACE:
            self.toggle_play()
        elif event.key == pygame.K_ESCAPE:
            self.stop_audio()
            return "menu"
        elif event.key == pygame.K_LEFT:
            self.seek(self._playback_ms - self._ms_per_beat)
        elif event.key == pygame.K_RIGHT:
            self.seek(self._playback_ms + self._ms_per_beat)
        elif event.key == pygame.K_HOME:
            self.seek(0)
        elif event.key == pygame.K_a:
            self._toggle_audio()
        elif event.key == pygame.K_PAGEDOWN:
            self.set_tempo_factor(self._tempo_factor - 0.05)
        elif event.key == pygame.K_PAGEUP:
            self.set_tempo_factor(self._tempo_factor + 0.05)
        elif event.key == pygame.K_i:
            self._set_loop_start(self._playback_ms)
        elif event.key == pygame.K_o:
            self._set_loop_end(self._playback_ms)
        elif event.key == pygame.K_p:
            self._toggle_loop()
        elif event.key == pygame.K_b:
            self._toggle_backing()
        elif event.key == pygame.K_x:
            self.set_noise_gate_db(self._noise_gate_db - 5)
        elif event.key == pygame.K_c:
            self.set_noise_gate_db(self._noise_gate_db + 5)
        elif event.key == pygame.K_t:
            self._cycle_theme()
        elif event.key == pygame.K_f:
            self._cycle_fret_limit()
        elif event.key == pygame.K_F1:
            self._toggle_string(1)
        elif event.key == pygame.K_F2:
            self._toggle_string(2)
        elif event.key == pygame.K_F3:
            self._toggle_string(3)
        elif event.key == pygame.K_F4:
            self._toggle_string(4)
        elif event.key == pygame.K_F5:
            self._toggle_string(5)
        elif event.key == pygame.K_F6:
            self._toggle_string(6)
        elif event.key == pygame.K_v:
            self._toggle_chord_mode()
        elif event.key == pygame.K_l:
            self._loop_weakest_section()
        elif event.key == pygame.K_h:
            self._show_help = not self._show_help
        elif event.key == pygame.K_w:
            self._toggle_wait_mode()
        elif event.key == pygame.K_TAB:
            direction = -1 if event.mod & pygame.KMOD_SHIFT else 1
            self._seek_adjacent_section(direction)
        elif event.key in (pygame.K_LEFTBRACKET, pygame.K_F7):
            self._adjust_latency(-10.0)
        elif event.key in (pygame.K_RIGHTBRACKET, pygame.K_F8):
            self._adjust_latency(10.0)
        elif event.key in (pygame.K_r, pygame.K_F9):
            self._set_latency_offset(0.0)
        elif event.key == pygame.K_F10:
            self._show_latency_details = not self._show_latency_details

        return None

    def render(self, surface: pygame.Surface) -> None:
        """Draw the full playing screen."""
        t = get_theme()
        layout = self._layout(surface)

        surface.fill(t.bg)
        self._draw_stage_background(surface, layout)
        self._draw_lanes(surface, layout)
        self._draw_loop_region(surface, layout)
        self._draw_hit_zone(surface, layout)
        self._draw_notes(surface, layout)
        self._draw_string_labels(surface, layout)
        self._draw_hud(surface, layout)

        if self._show_help:
            self._draw_help_overlay(surface, layout)

    def _draw_stage_background(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw a layered app shell and a raised practice surface."""
        pygame.draw.rect(surface, (8, 13, 21), (0, 0, layout.screen_w, LANE_TOP_MARGIN))
        footer_y = layout.screen_h - LANE_BOTTOM_MARGIN
        pygame.draw.rect(
            surface, (10, 15, 23),
            (0, footer_y, layout.screen_w, LANE_BOTTOM_MARGIN),
        )
        pygame.draw.line(surface, (39, 51, 67), (0, footer_y), (layout.screen_w, footer_y))

        # The track reads as one intentional surface instead of a spreadsheet.
        track_rect = pygame.Rect(
            14, LANE_TOP_MARGIN - 10,
            layout.screen_w - 28, footer_y - LANE_TOP_MARGIN + 20,
        )
        pygame.draw.rect(surface, (5, 10, 17), track_rect.move(0, 6), border_radius=22)
        pygame.draw.rect(surface, (17, 27, 41), track_rect, border_radius=22)
        pygame.draw.rect(surface, (45, 61, 79), track_rect, 1, border_radius=22)

    # -- Pure math helpers (testable without display) --

    def _layout(self, surface: pygame.Surface) -> _Layout:
        """Compute layout from current surface dimensions."""
        w, h = surface.get_size()
        lane_area = h - LANE_TOP_MARGIN - LANE_BOTTOM_MARGIN
        lane_height = lane_area / 6
        note_h = lane_height * NOTE_HEIGHT_FRACTION
        hit_zone_x = w * self._hit_zone_fraction
        usable_width = w - hit_zone_x
        pixels_per_ms = usable_width / self._visible_window_ms if self._visible_window_ms > 0 else 1.0
        return _Layout(
            screen_w=w,
            screen_h=h,
            lane_height=lane_height,
            note_h=note_h,
            hit_zone_x=hit_zone_x,
            usable_width=usable_width,
            pixels_per_ms=pixels_per_ms,
            visible_window_ms=self._visible_window_ms,
        )

    @staticmethod
    def note_x(note_timestamp_ms: float, playback_ms: float,
               hit_zone_x: float, pixels_per_ms: float) -> float:
        """Calculate the x position of a note."""
        return hit_zone_x + (note_timestamp_ms - playback_ms) * pixels_per_ms

    @staticmethod
    def note_width(duration_ms: float, pixels_per_ms: float) -> float:
        """Calculate note rectangle width, enforcing minimum."""
        return max(duration_ms * pixels_per_ms, MIN_NOTE_WIDTH_PX)

    # -- Drawing --

    def _draw_lanes(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        for i in range(6):
            y = LANE_TOP_MARGIN + i * layout.lane_height
            bg = (20, 31, 46) if i % 2 == 0 else (18, 28, 42)
            lane_rect = pygame.Rect(15, int(y), layout.screen_w - 30, int(layout.lane_height + 1))
            pygame.draw.rect(surface, bg, lane_rect)
            # A centered string line makes the view read like tablature rather
            # than six large arcade lanes.
            line_y = int(y + layout.lane_height / 2)
            pygame.draw.line(
                surface, t.lane_line,
                (56, line_y), (layout.screen_w - 26, line_y),
                2,
            )

    def _draw_string_labels(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw tuning labels above notes so every string stays identifiable."""
        t = get_theme()
        scale = max(1.0, min(1.45, layout.screen_h / 720.0))
        font = _get_font("Segoe UI Semibold", int(15 * scale))
        for index, label in enumerate(STRING_LABELS):
            center_y = int(
                LANE_TOP_MARGIN + index * layout.lane_height + layout.lane_height / 2
            )
            text = font.render(label, True, t.hud_text)
            pad_x, pad_y = 9, 5
            bg_rect = pygame.Rect(
                25,
                center_y - text.get_height() // 2 - pad_y,
                text.get_width() + pad_x * 2,
                text.get_height() + pad_y * 2,
            )
            color = STRING_COLORS.get(index + 1, t.hud_text)
            size = max(bg_rect.width, bg_rect.height)
            bg_rect.width = size
            bg_rect.height = size
            bg_rect.centery = center_y
            pygame.draw.ellipse(surface, (9, 16, 26), bg_rect)
            pygame.draw.ellipse(surface, color, bg_rect, width=3)
            surface.blit(
                text,
                (bg_rect.centerx - text.get_width() // 2,
                 bg_rect.centery - text.get_height() // 2),
            )

    def _draw_hit_zone(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        x = int(layout.hit_zone_x)
        top = int(LANE_TOP_MARGIN)
        bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        glow = pygame.Surface((38, bottom - top), pygame.SRCALPHA)
        for width, alpha in ((36, 9), (24, 16), (12, 28)):
            pygame.draw.rect(
                glow, (*t.hit_zone, alpha),
                ((38 - width) // 2, 0, width, bottom - top),
                border_radius=width // 2,
            )
        surface.blit(glow, (x - 19, top))
        pygame.draw.line(surface, t.hit_zone, (x, top), (x, bottom), 4)
        pygame.draw.circle(surface, (8, 13, 21), (x, top), 11)
        pygame.draw.circle(surface, t.hit_zone, (x, top), 7)

    def _draw_notes(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        # Visible time range with margins for long notes
        view_start = self._playback_ms - LEFT_MARGIN_MS
        view_end = self._playback_ms + self._visible_window_ms + RIGHT_MARGIN_MS

        notes = self._timeline.get_notes_in_range(view_start, view_end)

        fret_font_size = min(30, max(17, int(layout.note_h * 0.48)))
        fret_font = _get_font("Segoe UI Semibold", fret_font_size)

        for note in notes:
            # Difficulty filter: skip notes that fail
            if not self._note_passes_filter(note):
                continue

            x = self.note_x(
                note.timestamp_ms, self._playback_ms,
                layout.hit_zone_x, layout.pixels_per_ms,
            )
            fret_label = str(note.fret)
            fret_text = fret_font.render(fret_label, True, t.note_text)
            w = max(
                self.note_width(note.duration_ms, layout.pixels_per_ms),
                fret_text.get_width() + 20,
            )

            # Skip notes fully off-screen
            if x + w < 0 or x > layout.screen_w:
                continue

            # Y position: string 1-6
            lane_y = LANE_TOP_MARGIN + (note.string - 1) * layout.lane_height
            y = lane_y + layout.lane_height / 2 - layout.note_h / 2

            # Color: feedback color if matched, dimmed if past the hit zone
            base_color = STRING_COLORS.get(note.string, (180, 180, 180))
            past_hit_zone = note.timestamp_ms < self._playback_ms
            if self._audio_enabled:
                color = self._feedback.get_note_color(
                    note, base_color, self._playback_ms, past_hit_zone,
                )
            else:
                color = dimmed(base_color) if past_hit_zone else base_color

            rect = pygame.Rect(int(x), int(y), int(w), int(layout.note_h))
            shadow = rect.move(0, 5)
            pygame.draw.rect(surface, (5, 9, 15), shadow, border_radius=NOTE_CORNER_RADIUS)
            pygame.draw.rect(surface, color, rect, border_radius=NOTE_CORNER_RADIUS)
            border = tuple(min(255, channel + 35) for channel in color)
            pygame.draw.rect(surface, border, rect, width=2, border_radius=NOTE_CORNER_RADIUS)
            highlight = pygame.Rect(rect.x + 7, rect.y + 5, max(1, rect.width - 14), 3)
            pygame.draw.rect(surface, (255, 255, 255), highlight, border_radius=2)

            # Always show the fret number, including on short notes.
            tx = rect.centerx - fret_text.get_width() // 2
            ty = rect.centery - fret_text.get_height() // 2
            outline = fret_font.render(fret_label, True, (20, 25, 32))
            surface.blit(outline, (tx + 1, ty + 2))
            surface.blit(fret_text, (tx, ty))

    def _draw_hud(self, surface: pygame.Surface, layout: _Layout) -> None:
        t = get_theme()
        w = layout.screen_w
        h = layout.screen_h
        scale = max(1.0, min(1.45, h / 720.0))
        title_font = _get_font("Segoe UI Semibold", int(22 * scale))
        time_font = _get_font("Segoe UI", int(16 * scale))
        hint_font = _get_font("Segoe UI", int(13 * scale))

        meta = self._timeline.metadata

        # Count-in overlay — large centered beat countdown
        if self._playback_ms < 0 and self._count_in_ms > 0:
            remaining_beats = int(-self._playback_ms / self._ms_per_beat) + 1
            remaining_beats = min(remaining_beats, self._config.count_in_beats)
            countdown_font = _get_font("arial", 120)
            countdown_surf = countdown_font.render(
                str(remaining_beats), True, t.hud_accent
            )
            surface.blit(
                countdown_surf,
                (w // 2 - countdown_surf.get_width() // 2,
                 h // 2 - countdown_surf.get_height() // 2),
            )

        # Song completion overlay
        if self._song_completed:
            self._draw_completion_overlay(surface, layout)

        # Top-left: title + artist
        title = meta.title or "Untitled"
        title = self._ellipsize_text(title, title_font, int(w * 0.38))
        title_surf = title_font.render(title, True, t.hud_text)
        surface.blit(title_surf, (28, 34))
        eyebrow_font = _get_font("Segoe UI Semibold", int(11 * scale))
        eyebrow = eyebrow_font.render("NOW PRACTICING", True, t.hud_accent)
        surface.blit(eyebrow, (29, 15))
        subtitle = meta.artist or meta.track_name or "Guitar lesson"
        current_section = self._structure.section_at(max(0.0, self._playback_ms))
        if current_section is not None:
            subtitle += (
                f"   •   {current_section.label}   •   "
                f"bars {current_section.start_measure + 1}-{current_section.end_measure + 1}"
            )
        subtitle = self._ellipsize_text(subtitle, hint_font, int(w * 0.40))
        subtitle_surf = hint_font.render(subtitle, True, (137, 154, 176))
        surface.blit(subtitle_surf, (29, 66))

        # Top-center: BPM with tempo percentage (and streak below it)
        pct = int(self._tempo_factor * 100)
        bpm_text = f"{pct}%   •   {meta.tempo} BPM"
        bpm_surf = title_font.render(bpm_text, True, t.hud_accent)
        tempo_box = bpm_surf.get_rect(center=(w // 2, 46)).inflate(38, 22)
        pygame.draw.rect(surface, (19, 40, 49), tempo_box.move(0, 3), border_radius=15)
        pygame.draw.rect(surface, (24, 54, 62), tempo_box, border_radius=15)
        pygame.draw.rect(surface, (55, 104, 105), tempo_box, 1, border_radius=15)
        surface.blit(bpm_surf, bpm_surf.get_rect(center=tempo_box.center))

        # Loop status below BPM
        loop_y = 80
        loop_info = self._loop_hud_text()
        if loop_info:
            loop_color = t.hud_accent if self._loop_enabled else t.hud_text
            loop_surf = hint_font.render(loop_info, True, loop_color)
            surface.blit(loop_surf, (w // 2 - loop_surf.get_width() // 2, loop_y))
            loop_y += 18

        if self._audio_enabled:
            self._feedback.draw_streak(surface, title_font, w // 2, loop_y)
            self._feedback.draw_live_feedback(
                surface,
                title_font,
                hint_font,
                w // 2,
                70,
                self._playback_ms,
            )

        # Top-right: time
        current = format_time(self._playback_ms)
        total = format_time(self._timeline.duration_ms)
        time_text = f"{current} / {total}"
        time_surf = time_font.render(time_text, True, t.hud_text)
        time_label = time_font.render("TIME", True, (122, 140, 162))
        time_x = w - max(time_surf.get_width(), time_label.get_width()) - 29
        surface.blit(time_label, (time_x, 17))
        surface.blit(time_surf, (w - time_surf.get_width() - 28, 39))

        self._draw_structure_map(surface, hint_font, w)

        # Top-right second line: accuracy stats
        stats_bottom_y = 70
        if self._audio_enabled and self._matcher is not None:
            stats = self._matcher.get_statistics()
            if stats["total"] > 0:
                accuracy = f"{stats['accuracy_percent']:.0f}% ACCURACY"
                acc_surf = hint_font.render(accuracy, True, t.feedback_hit)
                surface.blit(acc_surf, (w - acc_surf.get_width() - 180, 77))
                stats_bottom_y = 70

        # Top-right: noise gate + signal meter + tuner (below stats, when audio capture exists)
        if self._audio_enabled:
            gate_text = (
                f"GATE {int(self._noise_gate_db)} dB   •   "
                f"LAT {self._config.audio_latency_offset_ms:+.0f} ms"
            )
            gate_surf = hint_font.render(gate_text, True, (123, 143, 166))
            surface.blit(gate_surf, (w - gate_surf.get_width() - 28, stats_bottom_y))
            if self._audio_capture is not None:
                self._draw_signal_meter(surface, hint_font, w, stats_bottom_y + 22)
                self._draw_tuner(surface, hint_font, w, stats_bottom_y + 45)
                if self._show_latency_details:
                    diag = self._audio_capture.get_latency_diagnostics()
                    diag_text = (
                        f"{diag['sample_rate'] / 1000:.1f}K AUDIO  "
                        f"IN {diag['device_ms']:.0f}  "
                        f"BUF {diag['block_ms']:.0f}  "
                        f"DET {diag['detector_ms']:.0f}  "
                        f"EST {diag['estimated_ms']:.0f} ms  "
                        f"ARP {diag['guided_hits']}"
                    )
                    diag_surf = hint_font.render(diag_text, True, t.hud_accent)
                    surface.blit(diag_surf, (w - diag_surf.get_width() - 28, 116))
        elif self._audio_capture is not None:
            # Audio off but capture exists — still show meter and tuner
            self._draw_signal_meter(surface, hint_font, w, stats_bottom_y)
            self._draw_tuner(surface, hint_font, w, stats_bottom_y + 18)

        if self._audio_error:
            error_text = self._ellipsize_text(
                f"MIC ERROR: {self._audio_error}  |  ESC, then D to choose Microphone Array",
                hint_font,
                w - 24,
            )
            error_surf = hint_font.render(error_text, True, t.feedback_miss)
            surface.blit(error_surf, (12, 104))

        if self._wait_mode_frozen:
            self._draw_wait_prompt(surface, layout)

        # Bottom-center: play state + controls
        if self._playback_ms < 0:
            state = "Count-in"
        elif self._playing:
            if self._wait_mode_frozen:
                state = "Waiting..."
            elif self._audio_enabled:
                state = "Playing"
            else:
                state = "Auto-scroll"
        else:
            state = "Paused"
        audio_state = "ON" if self._audio_enabled else "off"
        loop_state = "ON" if self._loop_enabled else "off"
        backing_state = ""
        if self._midi_player is not None:
            backing_state = f"|  B: backing {'off' if self._backing_muted else 'ON'}  "
        wait_state = ""
        if self._wait_mode:
            wait_state = f"|  W: wait {'WAIT' if self._wait_mode_frozen else 'ON'}  "
        elif self._audio_enabled:
            wait_state = "|  W: wait off  "
        timing_state = ""
        if self._matcher is not None:
            timing_stats = self._matcher.get_statistics()
            mean_timing = timing_stats.get("mean_timing_error_ms")
            timed_notes = timing_stats.get("hits", 0) + timing_stats.get("close", 0)
            if mean_timing is not None and timed_notes >= 3:
                direction = "LATE" if mean_timing > 0 else "EARLY"
                timing_state = f"     AVG {abs(mean_timing):.0f}ms {direction}"
        status = (
            f"{state}     MIC {audio_state}     "
            f"WAIT {'ON' if self._wait_mode else 'OFF'}     LOOP {loop_state.upper()}"
            f"{timing_state}"
        )
        controls = "SPACE  Play/Pause     TAB  Section     W  Wait     F7/F8  Offset     F10  Latency     F11  Fullscreen     ESC  Menu"
        footer_y = layout.screen_h - LANE_BOTTOM_MARGIN
        status_surf = hint_font.render(status, True, t.hud_accent)
        controls_surf = hint_font.render(controls, True, (145, 160, 180))
        surface.blit(
            status_surf,
            (w // 2 - status_surf.get_width() // 2, footer_y + 16),
        )
        surface.blit(
            controls_surf,
            (w // 2 - controls_surf.get_width() // 2, footer_y + 53),
        )

        # Top-left second line: track name + filter info
        info_y = 92

        # Difficulty filter HUD
        filter_text = self._filter_hud_text()
        if filter_text:
            filter_surf = hint_font.render(filter_text, True, t.hud_accent)
            surface.blit(filter_surf, (18, info_y))
            info_y += 16

        # Chord mode HUD
        if self._chord_partial_credit != self._config._default_chord_partial_credit:
            chord_text = "Chords: strict" if self._chord_partial_credit else "Chords: easy"
            chord_surf = hint_font.render(chord_text, True, t.hud_accent)
            surface.blit(chord_surf, (12, info_y))

    def _draw_wait_prompt(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Show an unmistakable but unobtrusive cue while practice is paused."""
        t = get_theme()
        font = _get_font("Segoe UI Semibold", 18)
        text = font.render("YOUR TURN  •  PLAY THE NOTE", True, (9, 24, 27))
        box = text.get_rect(center=(layout.hit_zone_x + 105, LANE_TOP_MARGIN + 22)).inflate(32, 16)
        # Keep the pill fully visible even on narrow windows.
        box.left = max(10, box.left)
        box.right = min(layout.screen_w - 10, box.right)
        pygame.draw.rect(surface, t.hud_accent, box, border_radius=box.height // 2)
        surface.blit(text, text.get_rect(center=box.center))

    def _draw_structure_map(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        screen_w: int,
    ) -> None:
        """Draw detected sections as a compact map of the whole song."""
        x, y = 28, 126
        width = max(40, screen_w - 56)
        height = 18
        structure_end = (
            self._structure.sections[-1].end_ms if self._structure.sections else 0.0
        )
        total = max(1.0, self._timeline.duration_ms, structure_end)
        pygame.draw.rect(surface, (26, 37, 51), (x, y, width, height), border_radius=7)

        family_colors: dict[str, tuple[int, int, int]] = {}
        for section in self._structure.sections:
            if section.family not in family_colors:
                family_colors[section.family] = SECTION_COLORS[
                    len(family_colors) % len(SECTION_COLORS)
                ]
            left = x + int((section.start_ms / total) * width)
            right = x + int((section.end_ms / total) * width)
            rect = pygame.Rect(left + 1, y + 1, max(3, right - left - 2), height - 2)
            color = family_colors[section.family]
            pygame.draw.rect(surface, color, rect, border_radius=5)

            if rect.width >= 88:
                label = section.label
                if section.occurrence_count > 1:
                    label += f" {section.occurrence}/{section.occurrence_count}"
                label_surf = font.render(label, True, (8, 15, 22))
                if label_surf.get_width() <= rect.width - 10:
                    surface.blit(label_surf, label_surf.get_rect(center=rect.center))

        progress = max(0.0, min(1.0, self._playback_ms / total))
        cursor_x = x + int(progress * width)
        pygame.draw.line(
            surface, (255, 255, 255),
            (cursor_x, y - 3), (cursor_x, y + height + 3), 2,
        )

    def _seek_adjacent_section(self, direction: int) -> None:
        """Jump to the previous or next detected practice section."""
        sections = self._structure.sections
        if not sections:
            return
        current = self._structure.section_at(max(0.0, self._playback_ms))
        try:
            index = sections.index(current) if current is not None else 0
        except ValueError:
            index = 0
        target = max(0, min(len(sections) - 1, index + direction))
        if direction < 0 and current is not None and self._playback_ms - current.start_ms > 800:
            target = index
        self.seek(sections[target].start_ms)

    @staticmethod
    def _ellipsize_text(text: str, font: pygame.font.Font, max_width: int) -> str:
        """Shorten text with an ellipsis so HUD regions never overlap."""
        if font.size(text)[0] <= max_width:
            return text
        suffix = "..."
        shortened = text
        while shortened and font.size(shortened + suffix)[0] > max_width:
            shortened = shortened[:-1]
        return shortened.rstrip() + suffix

    def _draw_signal_meter(self, surface: pygame.Surface, font: pygame.font.Font,
                           screen_w: int, y: int) -> None:
        """Draw a compact horizontal signal level meter with dB label."""
        t = get_theme()
        db = self._signal_db_smooth

        bar_w = 100
        bar_h = 8
        db_min = -80.0
        db_max = -10.0

        # dB label
        db_display = max(db_min, min(db_max, db))
        label = f"Signal: {int(db_display)} dB"
        label_surf = font.render(label, True, t.hud_text)
        label_x = screen_w - label_surf.get_width() - 12
        surface.blit(label_surf, (label_x, y))

        # Bar position: to the left of the label
        bar_x = label_x - bar_w - 8
        bar_y = y + label_surf.get_height() // 2 - bar_h // 2

        # Bar background
        pygame.draw.rect(surface, t.signal_cold, (bar_x, bar_y, bar_w, bar_h))

        # Fill proportion
        fill_frac = max(0.0, min(1.0, (db - db_min) / (db_max - db_min)))
        fill_w = int(fill_frac * bar_w)

        if fill_w > 0:
            if db >= -30:
                color = t.signal_hot
            elif db >= self._noise_gate_db:
                color = t.signal_warm
            else:
                color = t.signal_cold
            pygame.draw.rect(surface, color, (bar_x, bar_y, fill_w, bar_h))

        # Bar border
        pygame.draw.rect(surface, t.hud_text, (bar_x, bar_y, bar_w, bar_h), 1)

        # Noise gate tick mark
        gate_frac = max(0.0, min(1.0, (self._noise_gate_db - db_min) / (db_max - db_min)))
        gate_x = bar_x + int(gate_frac * bar_w)
        pygame.draw.line(surface, t.hud_accent, (gate_x, bar_y - 2), (gate_x, bar_y + bar_h + 2), 1)

    def _draw_tuner(self, surface: pygame.Surface, font: pygame.font.Font,
                    screen_w: int, y: int) -> None:
        """Draw a compact tuner display with cents bar and note name."""
        t = get_theme()

        bar_w = 100
        bar_h = 8

        freq = self._tuner_freq_smooth

        if freq <= 0 or self._tuner_displayed_note < 0:
            # No pitch — show placeholder
            label = "Tuner: ---"
            label_surf = font.render(label, True, t.hud_text)
            surface.blit(label_surf, (screen_w - label_surf.get_width() - 12, y))
            return

        # Use hysteresis-stabilized note for the label, smoothed freq for cents
        midi_note, cents = freq_to_cents_deviation(freq)
        if midi_note < 0:
            return

        note_name = midi_to_name(self._tuner_displayed_note)
        # Recompute cents relative to the displayed note for consistency
        from pickhero.audio.note_utils import midi_to_freq as _mtf
        target_freq = _mtf(self._tuner_displayed_note)
        if target_freq > 0:
            import math
            cents = 1200 * math.log2(freq / target_freq)

        # Choose color based on cents deviation
        abs_cents = abs(cents)
        if abs_cents < 5:
            fill_color = t.tuner_in_tune
        elif abs_cents < 15:
            fill_color = t.tuner_close
        else:
            fill_color = t.tuner_off

        # Note name + cents label
        sign = "+" if cents >= 0 else ""
        label = f"{note_name} {sign}{int(cents)}\u00A2"
        label_surf = font.render(label, True, fill_color)
        label_x = screen_w - label_surf.get_width() - 12
        surface.blit(label_surf, (label_x, y))

        # Bar position: to the left of the label
        bar_x = label_x - bar_w - 8
        bar_y = y + label_surf.get_height() // 2 - bar_h // 2

        # Bar background
        pygame.draw.rect(surface, t.signal_cold, (bar_x, bar_y, bar_w, bar_h))

        # Fill indicator: center = in-tune, left = flat, right = sharp
        center_x = bar_x + bar_w // 2
        fill_offset = int((cents / 50.0) * (bar_w // 2))
        fill_offset = max(-bar_w // 2, min(bar_w // 2, fill_offset))

        if fill_offset >= 0:
            pygame.draw.rect(surface, fill_color,
                             (center_x, bar_y, fill_offset, bar_h))
        else:
            pygame.draw.rect(surface, fill_color,
                             (center_x + fill_offset, bar_y, -fill_offset, bar_h))

        # Bar border
        pygame.draw.rect(surface, t.hud_text, (bar_x, bar_y, bar_w, bar_h), 1)

        # Center tick mark (in-tune reference)
        pygame.draw.line(surface, t.hud_text,
                         (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), 1)

    def _draw_completion_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw the song completion results overlay."""
        t = get_theme()
        w, h = layout.screen_w, layout.screen_h

        # Semi-transparent dark overlay
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        surface.blit(overlay, (0, 0))

        header_font = _get_font("arial", 48)
        stat_font = _get_font("consolas", 28)
        hint_font = _get_font("arial", 18)

        center_y = h // 2 - 80

        # "Song Complete!" header
        header_surf = header_font.render("Song Complete!", True, t.hud_accent)
        surface.blit(header_surf, (w // 2 - header_surf.get_width() // 2, center_y))

        if self._audio_enabled and self._matcher is not None:
            # Accuracy stats
            stats = self._matcher.get_statistics()
            accuracy_text = (
                f"Accuracy: {stats['accuracy_percent']:.1f}%  "
                f"({stats['hits']}/{stats['total']})"
            )
            acc_surf = stat_font.render(accuracy_text, True, t.hud_text)
            surface.blit(acc_surf, (w // 2 - acc_surf.get_width() // 2, center_y + 60))

            # "New Best!" indicator
            if self._is_new_best:
                best_surf = stat_font.render("New Best!", True, (255, 220, 50))
                surface.blit(best_surf, (w // 2 - best_surf.get_width() // 2, center_y + 100))

            # Weakest sections
            weak = getattr(self, "_weakest_sections", [])
            if weak:
                section = weak[0]
                weak_text = (
                    f"Weakest: bars {section[0]+1}-{section[1]+1} "
                    f"({section[2]:.0f}%) -- press L to loop"
                )
                weak_surf = hint_font.render(weak_text, True, t.feedback_close)
                surface.blit(weak_surf, (w // 2 - weak_surf.get_width() // 2, center_y + 140))

            # Practice recommendations
            rec_y = center_y + 170
            for rec in self._recommendations:
                rec_surf = hint_font.render(rec, True, t.hud_accent)
                surface.blit(rec_surf, (w // 2 - rec_surf.get_width() // 2, rec_y))
                rec_y += 24

            # Controls hint
            hint_y = max(center_y + 180, rec_y + 10)
            hint_text = "SPACE to replay  |  L to loop weak section  |  ESC to menu"
            hint_surf = hint_font.render(hint_text, True, t.hud_text)
            surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, hint_y))
        else:
            # Auto-scroll completion — no stats
            hint_text = "SPACE to replay  |  ESC to menu"
            hint_surf = hint_font.render(hint_text, True, t.hud_text)
            surface.blit(hint_surf, (w // 2 - hint_surf.get_width() // 2, center_y + 70))

    def _draw_help_overlay(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw a help overlay explaining the track, note colors, and controls."""
        t = get_theme()
        w, h = layout.screen_w, layout.screen_h

        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        surface.blit(overlay, (0, 0))

        title_font = _get_font("arial", 28)
        section_font = _get_font("arial", 20)
        body_font = _get_font("arial", 17)
        hint_font = _get_font("arial", 14)

        cx = w // 2
        y = 30

        title_surf = title_font.render("Help", True, t.hud_accent)
        surface.blit(title_surf, (cx - title_surf.get_width() // 2, y))
        y += 40

        # -- How to read the track --
        lx = cx - 260  # left edge for content
        section = section_font.render("Reading the Track", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        track_lines = [
            "Notes scroll right-to-left toward the hit zone (white vertical line).",
            "The number on each note is the fret to press (0 = open string).",
            "Play the right fret on the right string as the note crosses the line.",
        ]
        for line in track_lines:
            surf = body_font.render(line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 21
        y += 6

        # -- The 6 rows --
        section = section_font.render("The 6 Rows = 6 Guitar Strings", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        row_lines = [
            "Each horizontal row is one guitar string, top to bottom:",
        ]
        for line in row_lines:
            surf = body_font.render(line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 21

        string_info = [
            (1, "Row 1 (top)     = high E  (thinnest)"),
            (2, "Row 2              = B"),
            (3, "Row 3              = G"),
            (4, "Row 4              = D"),
            (5, "Row 5              = A"),
            (6, "Row 6 (bottom) = low E  (thickest)"),
        ]
        for s, label in string_info:
            color = STRING_COLORS.get(s, (180, 180, 180))
            pygame.draw.rect(surface, color, (lx, y + 3, 14, 14),
                             border_radius=2)
            surf = body_font.render(label, True, t.hud_text)
            surface.blit(surf, (lx + 20, y))
            y += 20
        y += 4

        surf = body_font.render(
            "A note's color tells you which string to play — it matches the row.",
            True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 20
        surf = body_font.render(
            "Dimmed notes have already passed the hit zone.",
            True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 24

        # -- Feedback colors --
        section = section_font.render("Scoring (colors change after you play)", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 26

        surf = body_font.render(
            "When audio is on, notes change color after they pass the hit zone:",
            True, t.hud_text)
        surface.blit(surf, (lx, y))
        y += 22

        feedback = [
            (t.feedback_hit, "Turns green", "you played the correct note"),
            (t.feedback_close, "Turns yellow", "close, off by 1 semitone"),
            (t.feedback_miss, "Turns red", "you missed it (not played in time)"),
        ]
        for color, label, desc in feedback:
            pygame.draw.rect(surface, color, (lx + 10, y + 3, 14, 14),
                             border_radius=2)
            surf = body_font.render(f"{label} — {desc}", True, t.hud_text)
            surface.blit(surf, (lx + 30, y))
            y += 21
        y += 10

        # -- Controls --
        section = section_font.render("Controls", True, t.hud_accent)
        surface.blit(section, (lx, y))
        y += 24

        controls = [
            "SPACE: play/pause    LEFT/RIGHT: seek    HOME: restart",
            "TAB: next section    SHIFT+TAB: previous section",
            "A: toggle audio    PgDn/PgUp: tempo    X/C: noise gate",
            "B: backing track    T: theme    I/O: loop markers    P: toggle loop",
            "F: fret limit    F1-F6: toggle strings    V: chord mode    L: loop weakest",
            "W: wait mode (pause until correct note played)",
            "F7 / F8: latency -/+ 10 ms    F9: reset latency to automatic",
            "F10: live latency diagnostics",
            "F11: switch fullscreen / windowed mode",
        ]
        for line in controls:
            surf = hint_font.render(line, True, t.hud_text)
            surface.blit(surf, (lx, y))
            y += 18

        y += 10
        close_surf = hint_font.render("Press H to close", True, t.hud_accent)
        surface.blit(close_surf, (cx - close_surf.get_width() // 2, y))

    # -- Difficulty filter --

    def _cycle_fret_limit(self) -> None:
        """Cycle through fret limit options."""
        try:
            idx = FRET_LIMITS.index(self._max_fret)
            self._max_fret = FRET_LIMITS[(idx + 1) % len(FRET_LIMITS)]
        except ValueError:
            self._max_fret = FRET_LIMITS[0]
        self._config.max_fret = self._max_fret
        self._config.save()
        self._reset_matcher_for_filter()

    def _toggle_string(self, string: int) -> None:
        """Toggle a string on/off in the difficulty filter."""
        idx = string - 1
        self._active_strings[idx] = not self._active_strings[idx]
        # Don't allow all strings to be off
        if not any(self._active_strings):
            self._active_strings[idx] = True
            return
        self._config.active_strings = list(self._active_strings)
        self._config.save()
        self._reset_matcher_for_filter()

    def _reset_matcher_for_filter(self) -> None:
        """Reset matcher when filter changes mid-song."""
        if self._matcher:
            self._matcher.reset()
            self._matcher.note_filter = self._note_passes_filter
        self._feedback.reset()

    def _filter_hud_text(self) -> str | None:
        """Return difficulty filter text for HUD, or None if default."""
        parts = []
        if self._max_fret < 24:
            parts.append(f"Fret: 0-{self._max_fret}")
        if not all(self._active_strings):
            strs = " ".join(
                str(i + 1) if on else "_"
                for i, on in enumerate(self._active_strings)
            )
            parts.append(f"Strings: {strs}")
        return "  |  ".join(parts) if parts else None

    # -- Theme --

    def _cycle_theme(self) -> None:
        """Toggle between dark and light theme."""
        name = cycle_theme()
        self._config.theme = name
        self._config.save()

    # -- Chord mode --

    def _toggle_chord_mode(self) -> None:
        """Toggle chord partial credit on/off."""
        self._chord_partial_credit = not self._chord_partial_credit
        self._config.chord_partial_credit = self._chord_partial_credit
        self._config.save()
        if self._matcher:
            self._matcher.chord_partial_credit = self._chord_partial_credit

    # -- Wait mode --

    def _toggle_wait_mode(self) -> None:
        """Toggle wait mode on/off."""
        self._wait_mode = not self._wait_mode
        self._config.wait_mode = self._wait_mode
        self._config.save()
        if not self._wait_mode:
            self._wait_mode_frozen = False
            self._wait_mode_hold_ms = None

    # -- Latency compensation --

    def _adjust_latency(self, delta_ms: float) -> None:
        """Fine-tune timing compensation for the current Windows device."""
        self._set_latency_offset(self._config.audio_latency_offset_ms + delta_ms)

    def _set_latency_offset(self, value_ms: float) -> None:
        """Save a bounded manual correction and apply it to the live matcher."""
        old_value = self._config.audio_latency_offset_ms
        new_value = max(-250.0, min(250.0, round(value_ms / 10.0) * 10.0))
        self._config.audio_latency_offset_ms = new_value
        if self._matcher is not None:
            # Matcher time follows the slowed song clock, so a real-world
            # device delay must be scaled by the active tempo factor.
            self._matcher.audio_offset_ms += (new_value - old_value) * self._tempo_factor
        self._config.save()

    # -- Loop weakest section --

    def _loop_weakest_section(self) -> None:
        """Set loop to weakest section from completion screen."""
        weak = getattr(self, "_weakest_sections", [])
        if not weak or not self._song_completed:
            return
        section = weak[0]
        start_measure, end_measure = section[0], section[1]
        # Get measure time ranges from timeline
        measures = self._timeline.measures
        if not measures or start_measure >= len(measures):
            return
        start_ms = measures[start_measure].start_ms
        end_idx = min(end_measure + 1, len(measures) - 1)
        end_ms = measures[end_idx].end_ms if end_idx < len(measures) else self._timeline.duration_ms
        self._loop_start_ms = start_ms
        self._loop_end_ms = end_ms
        self._loop_enabled = True
        self._song_completed = False
        self._is_new_best = False
        self._weakest_sections = []
        self.seek(start_ms)

    # -- Loop control --

    def _set_loop_start(self, ms: float) -> None:
        """Set loop start marker. Auto-swap if after end, auto-enable when both set."""
        self._loop_start_ms = ms
        if self._loop_end_ms is not None and self._loop_start_ms > self._loop_end_ms:
            self._loop_start_ms, self._loop_end_ms = self._loop_end_ms, self._loop_start_ms
        self._enforce_min_loop()
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            self._loop_enabled = True

    def _set_loop_end(self, ms: float) -> None:
        """Set loop end marker. Auto-swap if before start, auto-enable when both set."""
        self._loop_end_ms = ms
        if self._loop_start_ms is not None and self._loop_end_ms < self._loop_start_ms:
            self._loop_start_ms, self._loop_end_ms = self._loop_end_ms, self._loop_start_ms
        self._enforce_min_loop()
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            self._loop_enabled = True

    def _enforce_min_loop(self) -> None:
        """Ensure loop region is at least one beat long."""
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            if self._loop_end_ms - self._loop_start_ms < self._ms_per_beat:
                self._loop_end_ms = self._loop_start_ms + self._ms_per_beat

    def _toggle_loop(self) -> None:
        """Toggle loop off (keep markers), then clear markers on second press."""
        if self._loop_enabled:
            self._loop_enabled = False
        elif self._loop_start_ms is not None or self._loop_end_ms is not None:
            self._loop_start_ms = None
            self._loop_end_ms = None
            self._loop_enabled = False
        # If everything is already None/False, do nothing

    def _loop_hud_text(self) -> str | None:
        """Return loop status text for HUD, or None if no markers."""
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            s = format_time(self._loop_start_ms)
            e = format_time(self._loop_end_ms)
            if self._loop_enabled:
                return f"LOOP {s} - {e}"
            return f"loop {s} - {e} (off)"
        if self._loop_start_ms is not None:
            return f"loop start: {format_time(self._loop_start_ms)}"
        if self._loop_end_ms is not None:
            return f"loop end: {format_time(self._loop_end_ms)}"
        return None

    def _draw_loop_region(self, surface: pygame.Surface, layout: _Layout) -> None:
        """Draw loop markers and shaded region between them."""
        if self._loop_start_ms is None and self._loop_end_ms is None:
            return

        t = get_theme()
        lane_top = int(LANE_TOP_MARGIN)
        lane_bottom = int(LANE_TOP_MARGIN + 6 * layout.lane_height)
        lane_h = lane_bottom - lane_top

        marker_color = t.loop_marker if self._loop_enabled else t.loop_marker_disabled
        region_color = t.loop_region if self._loop_enabled else t.loop_region_disabled

        # Draw shaded region between both markers
        if self._loop_start_ms is not None and self._loop_end_ms is not None:
            x_start = int(self.note_x(self._loop_start_ms, self._playback_ms,
                                      layout.hit_zone_x, layout.pixels_per_ms))
            x_end = int(self.note_x(self._loop_end_ms, self._playback_ms,
                                    layout.hit_zone_x, layout.pixels_per_ms))
            # Clamp to screen
            x_start = max(0, min(x_start, layout.screen_w))
            x_end = max(0, min(x_end, layout.screen_w))
            if x_end > x_start:
                overlay = pygame.Surface((x_end - x_start, lane_h), pygame.SRCALPHA)
                overlay.fill(region_color)
                surface.blit(overlay, (x_start, lane_top))

        # Draw start marker
        if self._loop_start_ms is not None:
            x = int(self.note_x(self._loop_start_ms, self._playback_ms,
                                layout.hit_zone_x, layout.pixels_per_ms))
            if 0 <= x <= layout.screen_w:
                pygame.draw.line(surface, marker_color, (x, lane_top), (x, lane_bottom), 2)
                # Right-pointing triangle at top
                pygame.draw.polygon(surface, marker_color, [
                    (x, lane_top), (x + 10, lane_top + 7), (x, lane_top + 14),
                ])

        # Draw end marker
        if self._loop_end_ms is not None:
            x = int(self.note_x(self._loop_end_ms, self._playback_ms,
                                layout.hit_zone_x, layout.pixels_per_ms))
            if 0 <= x <= layout.screen_w:
                pygame.draw.line(surface, marker_color, (x, lane_top), (x, lane_bottom), 2)
                # Left-pointing triangle at top
                pygame.draw.polygon(surface, marker_color, [
                    (x, lane_top), (x - 10, lane_top + 7), (x, lane_top + 14),
                ])

    # -- Audio control --

    def _toggle_audio(self) -> None:
        """Toggle audio capture on/off."""
        self._audio_enabled = not self._audio_enabled
        if self._audio_enabled:
            if self._playing:
                self._start_audio()
            else:
                # Start capture for signal monitoring even while paused
                self._start_capture_only()
        else:
            self._stop_audio()

    def _start_audio(self) -> None:
        """Start audio capture and create matcher."""
        try:
            from pickhero.audio.input import AudioCapture
            if self._audio_capture is None:
                self._audio_capture = AudioCapture(self._config)
            self._audio_capture.start()
            self._audio_error = None
            self._audio_started_at = time.perf_counter()
            self._matcher = NoteMatcher(
                self._timeline,
                timing_window_ms=self._config.timing_window_ms,
                audio_offset_ms=(
                    self._playback_ms
                    + self._config.audio_latency_offset_ms * self._tempo_factor
                ),
                chord_threshold_ms=self._config.chord_threshold_ms,
                note_filter=self._note_passes_filter if self._is_filter_active() else None,
                chord_partial_credit=self._chord_partial_credit,
            )
            target = self._matcher.expected_target_at(self._playback_ms)
            if target is None:
                self._audio_capture.set_expected_notes(())
            else:
                target_time, target_midis = target
                self._audio_capture.set_expected_notes(target_midis, target_time)
            self._feedback.reset()
        except Exception as e:
            print(f"Audio start failed: {e}")
            self._audio_error = str(e)

    def _start_capture_only(self) -> None:
        """Start audio capture for signal monitoring (no matcher)."""
        try:
            from pickhero.audio.input import AudioCapture
            if self._audio_capture is None:
                self._audio_capture = AudioCapture(self._config)
            self._audio_capture.set_expected_notes(())
            self._audio_capture.start()
            self._audio_error = None
            self._audio_started_at = time.perf_counter()
        except Exception as e:
            print(f"Audio capture start failed: {e}")
            self._audio_error = str(e)

    def start_monitoring(self) -> None:
        """Start the input meter before playback, including while paused."""
        if self._audio_enabled:
            self._start_capture_only()

    def _stop_audio(self) -> None:
        """Stop audio capture."""
        if self._audio_capture is not None:
            self._audio_capture.stop()

    def stop_audio(self) -> None:
        """Public method to stop audio (called on state transitions)."""
        self._stop_audio()
        self._audio_enabled = False
        if self._midi_player is not None:
            self._midi_player.close()
            self._midi_player = None

    # -- MIDI backing track --

    def _init_midi_player(self, backing_track: BackingTrack) -> None:
        """Create and open MidiPlayer. Silently continues if MIDI unavailable."""
        try:
            player = MidiPlayer(backing_track)
            if player.open():
                player.set_muted(self._backing_muted)
                self._midi_player = player
            else:
                player.close()
        except Exception as e:
            print(f"MIDI player init failed: {e}")

    def _toggle_backing(self) -> None:
        """Toggle backing track mute on/off."""
        if self._midi_player is None:
            return
        self._backing_muted = not self._backing_muted
        self._midi_player.set_muted(self._backing_muted)
