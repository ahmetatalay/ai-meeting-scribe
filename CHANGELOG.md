# Changelog

## 2.6.0 — 2026-11-24

### Added

- **`scribe transcribe <file>` re-transcribes a recording.** 2.5.0 added
  `--keep-audio` and claimed it was "useful for re-running a better model
  later" — but nothing read the file, so the documentation promised something
  the code did not do. This is that missing half.

  It matters more than closing a gap. A machine that cannot keep up live no
  longer has to choose a single model for two different jobs: record with one
  fast enough to follow the meeting, then re-transcribe with a better one
  afterwards, when nothing is waiting. The live transcript is rough but
  immediate; the saved one is as good as the hardware can ever produce.

  `SpoolReader` presents a finished recording through the same interface a
  live capture does, so `RecordingSession` drives it unchanged — the same
  overlap removal, boundary revision and speaker attribution. Verified on a
  two-channel recording: identical text and identical speaker labels to the
  live path. A mono file is read as meeting audio and says so, because there
  is no channel separation to attribute from and splitting one would invent
  an attribution.

  Also recovers a meeting whose live transcription went wrong — the wrong
  `--language`, or a crash — provided the audio was kept.

- **`--backend metal` runs whisper.cpp on the GPU.** The default backend runs
  ctranslate2 with int8 weights on the CPU, because ctranslate2 has no Metal
  support compiled in: `get_supported_compute_types("mps")` raises
  "unsupported device mps", so the GPU sat idle.

  Measured on an M5 Pro through the full pipeline, 30-second window against a
  15-second budget: `small` 4.71s → 0.50s, `large-v3-turbo` 7.78s → 0.73s,
  and `large-v3` **16.85s → 2.07s**. Between 8x and 11x.

  The last figure is the one that matters: `large-v3` goes from a model
  nothing can follow live to one with seven times the headroom it needs. A
  first measurement outside the pipeline suggested 15x; the honest number,
  with this project's own voice-activity filtering and confidence checks in
  the path, is 8–11x.

  It stays **opt-in** (`uv tool install --force '.[metal]'`) for two honest
  reasons. It is an extra dependency, and it reports one fewer confidence
  signal: whisper.cpp gives a segment probability but no `no_speech_prob`.
  The probability maps onto `avg_logprob` so the confidence threshold still
  works — measured 0.78–0.90 on real speech, i.e. −0.25 to −0.10 against a
  −1.0 threshold — and the missing signal is recorded as `None` rather than
  `0.0`, because zero would read as "certainly speech" instead of "unknown".

  Two things that had to be got right, both found by measuring rather than
  reading: the probability field comes back **NaN** unless
  `extract_probability=True` is passed, which would have disabled confidence
  filtering silently (cost of asking: 0.37s against 0.35s); and whisper.cpp
  reports segment bounds in **hundredths of a second**, which taken as
  seconds would place every segment hours into the recording.

- **`scribe doctor --speed --backend metal`** measures whichever backend you
  are considering.

## 2.5.0 — 2026-11-24

### Changed

- **Captured audio is spooled to disk, so falling behind no longer loses it.**
  This is the architectural fix behind the warning added in 2.4.1. Windows
  used to be read from a 30-second ring buffer addressed by wall clock: when
  transcription could not keep up, the next read skipped ahead and whatever
  sat in the middle was overwritten before anyone had looked at it. Measured
  on a passively cooled MacBook Air running `large-v3`, a single cycle
  reported 117 seconds of audio gone, twenty-five times over, with matching
  holes in the transcript.

  Audio is small — 16kHz stereo int16 is 64KB a second, 230MB an hour — so
  disk absorbs any backlog a laptop can produce. Capture appends to a plain
  WAV; transcription reads at a cursor that advances independently of the
  clock. A machine managing 0.8x realtime now finishes late instead of
  destroying what it could not reach in time, and stopping drains the rest
  rather than discarding it. Verified end to end: twenty seconds of marked
  audio fed to a consumer that never ran produced ten windows covering every
  second, none dropped.

  The spool is **deleted once the transcript is written**. A recording of a
  meeting is more sensitive than a transcript of it, and keeping one
  unintentionally is the worse failure. `--keep-audio`, or the checkbox in
  the web UI, keeps it beside the transcript — useful for re-running a better
  model later instead of losing the meeting.

- **Falling behind is reported as a delay, not a loss.** The warning now says
  how far behind transcription is and that nothing is lost, and withdraws
  itself once caught up. `dropped_seconds` is retained and still reported,
  because it should now always be zero and would indicate a real defect.

### Added

- **`scribe doctor --speed` measures this machine and recommends a model.**
  Timing one window per model answers "will this keep up here" before a
  meeting rather than twenty-five windows into one. The budget is the step,
  not the window: a window is produced every step seconds, so each must
  finish within one step — which is why keeping up needs twice realtime at
  the defaults.

  Measured on an M5 Pro at a 30s window and 15s step: `tiny` 0.83s, `small`
  5.24s, `large-v3-turbo` 7.61s, `large-v3` 17.04s — so `large-v3` falls
  behind on fast hardware, let alone a fanless laptop. Recommends the best
  model with real headroom rather than the best that merely fits, since the
  numbers come from an idle machine and a fanless one slows down once a
  meeting has been running a while.

  Two things this check got wrong while being built, both measured: timing
  against noise makes every model look instant, because the voice-activity
  filter strips it before the encoder (0.00s against 4.59s for speech); and
  timing against thirty unbroken seconds of speech overstates the cost badly
  enough to recommend `tiny` on a machine that runs `small` comfortably
  (`small` measured 21.8s that way against 5.2s at realistic density).

- **The step widens automatically when transcription falls behind**, and
  narrows back once it catches up. `--no-adaptive-step` turns this off.

  Each second of audio is transcribed `window / step` times, and the extra
  pass is what pays for correcting the previous window's boundary guess:
  `align_tail()` only replaces a batch it re-transcribes in full, which needs
  `window >= 2 * step`. The factor of two is therefore the correction's price
  rather than slack to be tuned away — widening the step past half the window
  does not reduce correction, it ends it. That makes it the right thing to
  give up under pressure, and the notice says so when it happens.

## 2.4.1 — 2026-11-23

### Fixed

- **Warnings that were wrong, unactionable, or never withdrawn.** All four were
  seen in one real meeting, where eight stacked banners pushed the transcript
  off screen — the thing the window exists to show.

  "Transcription fell behind: **0s** of audio was not transcribed. Try a
  smaller model." A window is emitted on wall-clock time, so a cycle finishing
  0.4s late tripped a `> 0` check and then rendered as `0s` through `:.0f`.
  That is scheduling jitter, no audio was lost, and changing model would not
  have helped. Only a shortfall of a second or more is reported now, never
  rounded to zero; shorter overruns still count towards the session total so
  the statistic stays honest.

  Notices appended one banner per occurrence. Repeated conditions now coalesce
  into a single banner with an occurrence count.

  "No meeting audio is arriving" stayed on screen after meeting audio started
  arriving — two of them sat above a transcript plainly containing meeting
  speech, with every setup check green, sending the reader to re-check routing
  that already worked. The warning is withdrawn when audio returns, and can
  fire again if routing breaks a second time.

- **A ctranslate2 tip advertising a backend that does not exist.** Every model
  load printed "experimental MPS backend is available and often 2x+ faster,
  try device=mps" into the transcript stream. On this build MPS is not
  compiled in — `get_supported_compute_types("mps")` raises "unsupported
  device mps" — so the suggestion could not be taken. Suppressed when MPS is
  genuinely unavailable; errors still come through.

## 2.4.0 — 2026-11-23

### Added

- **AI summaries, locally.** A "Notes so far" panel above the live transcript
  produces decisions, action items and open questions on demand; past
  recordings get a Summarise button in the reader. Summaries run through
  [Ollama](https://ollama.com) on the same machine, so transcripts still never
  leave it. Stored in the JSON sidecar, and the library marks which recordings
  have one.

  The live summary is **incremental**: each refresh passes the model only the
  speech since the last one plus the summary it already wrote, and asks for an
  updated whole. Re-reading the entire transcript each time both churns the
  text and gets steadily slower — measured on qwen2.5:14b, a from-scratch pass
  grew from 5.0s to 15.3s within five cycles.

  Refreshing is **on demand, not automatic**. An earlier build refreshed every
  12 seconds; that kept the model permanently resident at ~21GB and drove a
  48GB machine 23GB into swap. Automatic refresh is now opt-in via
  `--auto-notes SECONDS`.

  Two measured reductions to the model's footprint: an explicit `num_ctx` of
  8192 (the compacted prompt needs a few thousand tokens, and a large window
  only buys a large KV cache) took resident size from 21GB to 9.9GB, and a
  60-second `keep_alive` releases it between summaries — 0GB a minute after a
  press.

  The prompt asks for **substance rather than topics**, after a recording of a
  systems-design talk summarised as "Discussing handling large blobs (large
  files like photos or videos)" — section headings restated as bullets, with
  every mechanism and figure dropped. It now carries a worked bad/good
  contrast, requires stated numbers to survive, and tells the model to use its
  bullet budget rather than merging distinct mechanisms away. The same
  recording now yields "The load balancer cuts the connection at 30 seconds,
  so the API server queues the job and returns a job ID".

  Empty sections are framed as the **expected** case. The same summary had
  invented an action item, turning a speaker's "I am preparing for the
  presentation today with <colleague>" into a task assigned to both of them —
  current activity rather than a commitment, and the colleague had never been
  assigned anything. With Decisions and Open questions genuinely empty, the
  model filled the one section it could. An action item now requires a
  commitment to work not yet done.

### Fixed

- A summariser availability check reached the model server on every UI state
  change and view switch, in every open tab; the answer is now reused for 30
  seconds on both sides, and the check runs off the event loop.

  The summary size is bounded **in code**, not asked for in the prompt: a
  running summary instructed to stay under 250 words reached 555 after nine
  passes, with latency growing from 15s to 27s alongside it. Each section's
  bullets and the prose section's sentences are now trimmed before the summary
  is fed back, so the input to every pass stays bounded however long the
  meeting runs.

  It runs on the server rather than in the browser, so it survives a reload,
  and is stored with the recording on stop. Output is English whatever language
  the meeting is in.

- `--notes-model` to run the live panel on a smaller, faster model than the
  final write-up.

  Prompts actively suppress invention: the model is told to use only what the
  transcript states, to write "None" for empty sections, that the text is
  machine-generated and contains misheard words, and that the speaker labels
  come from audio channels rather than voice recognition and are occasionally
  wrong. Transcripts longer than one pass are summarised in parts and merged
  rather than truncated.

  When Ollama is not installed the UI says so with the command to fix it, and
  the button stays disabled instead of failing mid-request.

- **Related recordings.** Earlier meetings sharing distinctive terms with the
  one being read, found by term overlap weighted by document frequency — no
  model involved. The shared terms are shown so the link can be judged rather
  than trusted. This is the deliberate alternative to feeding several
  transcripts to a model, which invents cross-references.

- `scribe web --summary-model` to choose the Ollama model (default
  `qwen2.5:14b`).

### Fixed

- **`scribe web` could not be stopped while a browser tab was open.** The SSE
  response never completes, and uvicorn's graceful shutdown waits for in-flight
  requests, so the process lingered — several accumulated during development,
  suspended in the background, holding ports and burning CPU. The stream now
  ends on shutdown.
- Idle event streams sent a keep-alive five times a second, which is what gave
  idle servers measurable CPU time. Now every 15 seconds.
- Starting `scribe web` on a port already in use now says so and exits,
  instead of lingering as a process that serves nothing.

## 2.3.0 — 2026-11-23

### Changed

- **Window boundaries no longer keep a wrong guess.** Consecutive windows
  overlap, so the last batch's audio is transcribed a second time — with the
  following context the first pass lacked. Observed live: "Airbnb has the same
  purpose." became "...the same problem when two guests book..." on the second
  pass, but the first guess had already been written. The second reading now
  replaces the first instead of being appended around it.

  Replacement is only attempted when the new window re-transcribes the previous
  batch *in full*. A partial match would delete the part the window does not
  cover, and a match reaching further back than the batch would duplicate text
  written before it — both were reproduced in testing and are now refused
  outright, falling back to plain appending.

- The engine publishes a `revise` event carrying the id to replace from, and
  the UI swaps the affected text in place. Each segment is rendered as its own
  addressable node for this.

- Transcript statistics are recomputed from what is on disk after each write,
  rather than accumulated, so a revision cannot leave the counts drifting.

## 2.2.0 — 2026-11-23

### Added

- **Transcript library** in the web UI. A Library tab listing every recording
  newest-first with title, date, duration, word count and speaker breakdown;
  click to read, rename inline, or delete. Search covers both titles and
  transcript bodies.
- Recordings predating sidecars are listed too: the date comes from the
  filename and the title from the document heading, so an older `raw/`
  directory is readable without migration.
- Renaming writes only the JSON sidecar, so the markdown is never rewritten.

### Fixed

- **The library listed any markdown file, with a Delete button next to it.**
  Pointed at a project directory or an Obsidian vault it offered your notes as
  recordings, one click from permanent deletion. Listing now requires positive
  evidence that scribe produced the file — a sidecar, a `started` frontmatter
  key, the transcript filename pattern, or a `# Transcript —` heading. Content
  heuristics were not enough: this project's own README *contains* an example
  transcript. The filter applies to reading and deleting as well, so a direct
  URL cannot bypass it, and the UI reports how many files were ignored.
- Recordings without a sidecar sorted by file mtime, so copying a directory
  put old transcripts above new ones. The start time now comes from the
  filename.

### Security

- The library routes take a filename from the request and read, rename and
  delete files with it. Names are restricted to a bare filename pattern and
  the resolved path must sit directly inside the output directory, so `../`,
  absolute paths and symlinks pointing outside are all refused. Deleting the
  recording currently in progress is refused.

## 2.1.0 — 2026-11-23

### Added

- **Local web UI** — `scribe web`, served on loopback only. Device pickers for
  the microphone and meeting audio separately, live transcript, a setup panel,
  and two dB level meters that make a broken audio route obvious in seconds
  rather than after the call.
- **Loopback-only by design.** The bind address is hardcoded and there is no
  host option, so the server cannot be exposed by a stray flag. Host-header
  validation (DNS rebinding), a per-run CSRF token on every route (another site
  cannot read responses but can send a POST), no CORS headers, API docs off.
- **`Why?` toggle** in the transcript — shows the per-segment decision and the
  dB numbers behind it, so an unexpected speaker label can be diagnosed from
  the transcript instead of by re-recording.
- **`scribe doctor` microphone level check** — measures your voice against your
  mic's own noise floor and says whether the margin is enough for labelling.

### Fixed

- Both device pickers defaulted to whichever input came first. `new Option(...)`
  sets selectedness on a detached option, which browsers do not honour; the
  selection is now assigned after the options are in the DOM.
- The `hidden` attribute was being defeated by author CSS — author styles beat
  the user-agent origin regardless of specificity — which left the "no meeting
  audio" warning and an empty footer on screen permanently.
- New speech scrolled the whole page. The transcript is now its own scroll
  region, and auto-scroll only follows when you are already at the bottom;
  otherwise a "jump to latest" control appears.
- Selecting the same device for both streams is refused with an explanation.
- Early windows no longer hand Whisper a mostly-zero buffer, which returned one
  long segment where trimmed audio returns several (finer segments give speaker
  attribution more to work with).

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
