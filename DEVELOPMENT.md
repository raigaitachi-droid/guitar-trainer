# Guitar Trainer — development brief

This branch is the first development checkpoint for our real-time guitar tab
trainer. It starts from PickHero v1.1.0 and keeps the existing audio capture,
Guitar Pro parsing, scrolling practice view, looping, tempo control, and Wait
Mode while we improve the feedback and practice engine.

## First milestone

The matcher now returns structured performance feedback for every picked note:

- expected and detected MIDI note
- signed pitch error in semitones
- exact timing error in milliseconds
- early, on-time, or late classification
- detector confidence
- explicit wrong-note attempts that do not consume the expected note

This enables feedback such as:

```text
CORRECT
Expected E4 | Played E4 | 37 ms late
```

and:

```text
WRONG NOTE
Expected E4 | Played F4 | 18 ms early
```

## Next milestones

1. Test the detector with the user's actual guitar and audio interface.
2. Add an input-latency calibration wizard.
3. Save detailed note attempts for session review.
4. Add automatic section repetition and tempo progression.
5. Choose a product name and replace the inherited visual identity.

## Upstream licensing note

The PickHero v1.1.0 README declares the project MIT licensed, but that tag does
not include a standalone LICENSE file or full copyright notice. The upstream
history and README are preserved. Before public or commercial distribution,
obtain or confirm the complete license notice with the upstream author and
include it with redistributed source or binaries.
