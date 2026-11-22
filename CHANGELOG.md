# Changelog

## 2.0.0 — 2026-11-23

A correctness release. Measured against real recordings, 1.0.0 duplicated text
on roughly a third of chunks and misattributed almost all speech: in one
36-minute call, 97 of 97 chunks were labelled "You", including the whole other
side of the conversation. Both causes are fixed, and the engine was restructured
so a UI can drive it.

### Breaking

- **`--device`, `--device-id` and `--mic-channel` removed.** scribe no longer
  needs a macOS Aggregate Device; it opens the microphone and the loopback as
  two independent streams. Running the old flags prints a migration note.

  ```
  Old: scribe record --device "Meeting Input" --mic-channel 0
  New: scribe record --mic "Plantronics"
  ```

- **`scribe summarize` removed.** It sent transcripts to a third-party service,
  which is at odds with the rest of the tool running entirely locally. Pipe the
  markdown file to whatever summariser you prefer.

- Audio MIDI Setup now needs only **one** Multi-Output Device. The Aggregate
  Device and any per-headset duplicates can be deleted.

### Fixed

- **Overlap duplication.** Consecutive windows re-transcribe the same audio and
  Whisper words it differently each pass ("AWS product" / "AWS production"), so
  the old phrase-matching fallback re-emitted whole turns. Overlap removal now
  works on word indices with an alignment anchored at the head of the window,
  tolerating re-wording inside the overlap.
- **Speaker attribution.** The hardcoded `0.001` RMS threshold sat below the
  noise floor of an ordinary microphone, so every decision collapsed into a raw
  level comparison between channels with different gains — which the near-field
  mic always won. Each channel is now calibrated against its own noise floor and
  compared by relative prominence in dB.
- **Mid-sentence speaker flips.** Attribution runs per segment rather than per
  word.
- **Lost tail.** The final partial window is transcribed on stop; recordings no
  longer end mid-sentence.
- **Slow Ctrl+C.** Stopping is a lifecycle step, not a flag checked after a
  blocking wait.
- **Silent overwrite.** Transcript filenames had minute granularity and were
  opened for truncation, so two recordings started in the same minute destroyed
  the first. Colliding names now get a `-2` suffix.
- **Encoding.** All file I/O is explicitly UTF-8.
- **Headset disconnects.** A stalled stream is detected and excluded from
  alignment, so unplugging a headset costs speaker labels rather than the
  recording.
- **Hallucinations.** Segments are filtered on `avg_logprob` and
  `no_speech_prob`, both of which were previously ignored, and Whisper's
  repeated-phrase loops ("Permission? Permission? Permission?") are collapsed.
- A failed session no longer reports itself as a clean finish.

### Added

- `session.py` — a headless recording engine that owns the loop and publishes
  `state`, `segments`, `levels`, `notice`, `session` and `summary` events.
  Prints nothing; the CLI is one consumer of that stream.
- JSON sidecar beside each transcript with structured segments and metadata.
- YAML frontmatter and continued paragraphs for same-speaker turns.
- `--language auto`, detected once per recording and then pinned.
- `--mic-margin` to tune speaker detection, and `--debug` to see the energy
  behind each decision.
- `--no-mic` / `--no-system` to record one side only.
- Runtime warning for the most common setup mistake: speech being transcribed
  while the loopback is silent, which means the meeting app is bypassing the
  Multi-Output Device.
- Warning when transcription falls behind, reporting seconds of audio lost.
- Test suite: 170 tests, no audio hardware or model required.

### Changed

- `scribe devices` lists outputs as well as inputs.
- `scribe doctor` no longer requires devices named "Meeting Input"; it checks
  the loopback, output routing and microphone, and calls out the Zoom/Teams
  speaker setting.
- Default model documented as `small`; `tiny` is for smoke tests.

## 1.0.0

Initial release.
