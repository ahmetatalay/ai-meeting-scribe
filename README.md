# Scribe — AI Meeting Transcriber

Scribe transcribes meetings on your own machine. It captures audio from any meeting platform — Microsoft Teams, Zoom, Slack Huddle, Google Meet — transcribes it locally with Whisper, and labels who was speaking.

Nothing leaves your laptop: no account, no API key, no network calls after the model is downloaded once.

## How it works

```
Meeting audio (Teams / Zoom / Slack Huddle)
    ↓
Multi-Output Device → you hear it + BlackHole gets a copy
    ↓
scribe opens TWO independent input streams
    • your microphone      → "You"
    • BlackHole (loopback) → "Meeting"
    ↓
aligned on the wall clock, mixed, transcribed locally
    ↓
per-channel energy decides who spoke
    ↓
raw/transcript_YYYY-MM-DD_HHMM.md  (+ .json sidecar)
```

Output:

```markdown
**You**: How is the migration going?
**Meeting**: It is going well — we finished the first phase last night.
```

Two separate streams rather than one Aggregate Device is deliberate. An aggregate breaks when you unplug a headset: the missing member becomes "Offline Device", the channel map shifts, and recordings corrupt silently. With separate streams, unplugging a headset costs you speaker labels, not the meeting.

| Component | Technology |
|-----------|-----------|
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) |
| Models | [Whisper](https://huggingface.co/Systran), MIT licensed, cached in `~/.cache/huggingface/` |
| Audio capture | [sounddevice](https://python-sounddevice.readthedocs.io/) + [BlackHole](https://existential.audio/blackhole/) |
| Speaker detection | Calibrated per-channel energy — no ML diarization |

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- BlackHole 2ch — `brew install blackhole-2ch`
- PortAudio — `brew install portaudio`
- Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/)

## One-time setup

You need **one** Multi-Output Device, so you can hear the meeting while BlackHole receives a copy. No Aggregate Device, and no second set of devices per headset.

Open **Audio MIDI Setup** (`Cmd + Space` → "Audio MIDI Setup"):

1. Click **+** → **Create Multi-Output Device**
2. Tick your headset *and* every other device you listen through (speakers, a second headset)
3. Tick **BlackHole 2ch**
4. Set **Primary Device** to **BlackHole 2ch** — it is always present, so the group survives a Bluetooth headset dropping out
5. Tick **Drift Correction** on every row *except* BlackHole
6. Right-click → rename to **"Meeting Capture"**

Before each meeting, set **Sound Output → Meeting Capture**. Afterwards, set it back.

A device can only be ticked while it is connected, so add each new headset once. The group skips members that are absent, which is why one device serves every headset.

### The setting people get wrong

**In Zoom/Teams, set Speaker to "Same as System" — not your headset.**

Choosing your headset inside the meeting app bypasses the Multi-Output Device entirely. BlackHole then receives nothing and you get a transcript containing only your own voice. This is by far the most common cause of a one-sided recording. Microphone can stay set to your headset.

scribe warns about this at runtime as soon as it sees your speech with a silent loopback. `scribe doctor` checks the rest.

## Quick start

```bash
uv sync --extra web

scribe doctor      # check the audio setup
scribe web         # browser UI (recommended)
scribe record      # or stay in the terminal; Ctrl+C to stop
```

## Web UI

```bash
scribe web                    # opens http://127.0.0.1:8765
scribe web --port 9000
scribe web --no-browser
scribe web --raw ~/Desktop/transcripts
```

Two tabs:

**Record** — pick your microphone and meeting-audio device, press **Start transcribing**, and the transcript appears as it is produced. Two live level meters sit under the controls: if the **Meeting** meter stays flat, the meeting audio is not reaching scribe and you can fix it in seconds instead of discovering it after the call. The **Why?** toggle shows the per-segment speaker decision and the dB numbers behind it.

**Library** — every recording in your output directory, newest first, with title, date, duration, word count and speaker breakdown. Click one to read it, rename it inline, or delete it. Search covers both titles and transcript bodies.

Recordings made before sidecars existed are listed too: the date comes from the filename and the title from the document heading. Renaming only ever writes the JSON sidecar, so your markdown is never rewritten.

### Why it is loopback-only

The server binds `127.0.0.1` and there is deliberately **no host option**, so it cannot be exposed by a stray flag. Transcripts are verbatim records of private conversations, so the localhost-specific attacks are treated as real:

- **Host header validation** — a page you visit can point a hostname it controls at `127.0.0.1` (DNS rebinding) and then talk to the server as same-origin. Foreign `Host` values are refused.
- **CSRF token** — the browser's same-origin policy stops another site *reading* responses but not *sending* a `POST`. Without a token, any page you had open could start or stop a recording. The token is minted per run, embedded in the page, and required on every route.
- **No CORS headers**, and the API docs endpoints are disabled.
- **Path containment** — the library routes take a filename from the request and read, rename and *delete* files with it, so names are restricted to a bare filename and the resolved path must sit directly inside your output directory. `../`, absolute paths and symlinks pointing outside are all refused.

To reach the UI from another machine, use an SSH local port-forward rather than widening the bind:

```bash
ssh -L 8765:127.0.0.1:8765 your-mac
```

## Usage

```bash
# Defaults: system default microphone + BlackHole
scribe record

# Pick a microphone explicitly (partial name match)
scribe record --mic "Plantronics"

# Meeting audio only — no speaker labels, nothing to break on reconnect
scribe record --no-mic

# Your microphone only
scribe record --no-system

# Model and language
scribe record --model medium
scribe record --language tr
scribe record --language auto      # detect once, then pin

# Output
scribe record --raw ~/Desktop/transcripts --title "Sprint planning"

# Overlapping window (default: 30s window, 15s step)
scribe record --window 45 --step 20

# Speaker-detection tuning
scribe record --mic-margin 12      # dB above the mic's noise floor = your speech
scribe record --debug              # per-segment channel energy
scribe record --no-speakers        # disable labelling

# Keep low-confidence filler segments ("You", "Thank you.")
scribe record --keep-filler
```

### Output files

Each recording produces two files side by side:

- `transcript_<date>_<time>.md` — YAML frontmatter plus speaker-labelled turns. The file to read; indexes cleanly in Obsidian.
- `transcript_<date>_<time>.json` — the same content as structured segments with timings, plus metadata.

A recording started in the same minute as an existing one gets a `-2` suffix rather than overwriting it.

## Install globally

```bash
uv tool install .
scribe record --raw ~/Desktop/transcripts
```

## Environment variables

```bash
export SCRIBE_RAW_DIR=~/Desktop/raw   # default output directory
```

## Whisper models

| Model | Download | Speed vs realtime | Good for |
|-------|----------|-------------------|----------|
| `tiny` | ~40MB | 37x | Smoke tests only |
| `base` | ~150MB | not measured | Fast, lower accuracy |
| `small` | ~500MB | 8x | **Recommended (default)** |
| `large-v3-turbo` | ~1.6GB | 2.4x | Best quality that still keeps up |
| `medium` / `large-v3` | ~1.5GB / ~3GB | not measured | Maximum quality |

Measured on Apple Silicon with `int8` on an 11-second clip. Even `tiny` handles clean audio reasonably; quality problems are far more often a wrong `--language` than a small model.

If transcription cannot keep up, scribe reports how many seconds of audio were lost — drop to a smaller model if you see that.

## How speaker detection works

Your microphone carries only your voice; BlackHole carries only the meeting. For each transcribed segment, scribe compares how far each channel rises above **its own measured noise floor**.

A fixed threshold does not work. A near-field headset mic is simply louder than post-mixer loopback audio, so comparing raw levels attributes everything to you. Measuring each channel against its own floor also handles the opposite case, where a built-in mic picks up the laptop speakers — that bleed is attenuated, so it stays below the loopback's.

Decisions are made per segment, not per word, so a sentence is never split between speakers.

## Troubleshooting

### Transcript contains only my voice
Your meeting app is bypassing the Multi-Output Device. Set **Speaker → "Same as System"** in Zoom/Teams.

### Everything labelled "You", or everything "Meeting"
Run with `--debug` to see per-segment energy and which rule fired. Raise `--mic-margin` if ambient noise is being treated as your speech; lower it if your speech is being missed.

### Confident nonsense in the transcript
Whisper forced into the wrong language invents fluent text rather than failing. Set `--language` correctly, or use `--language auto`.

### "No input devices found"
```bash
brew install portaudio
uv pip install --force-reinstall sounddevice
```

### BlackHole missing, or audio broken after reconnecting a headset
```bash
sudo killall coreaudiod          # restarts automatically
brew reinstall blackhole-2ch     # then restart macOS
```

### Sentences breaking mid-word
Increase the window: `--window 45 --step 20`.

### Quick diagnostic
```bash
scribe doctor
```

## Development

```bash
uv sync --group dev --extra web
uv run pytest
```

Layout:

| Module | Role |
|--------|------|
| `capture.py` | Two input streams, wall-clock aligned ring buffers |
| `transcriber.py` | faster-whisper wrapper, confidence filtering, loop collapsing |
| `speakers.py` | Calibrated per-channel speaker attribution |
| `dedup.py` | Overlap removal between consecutive windows |
| `writer.py` | Markdown + JSON sidecar output |
| `session.py` | Recording engine — owns the loop, publishes events |
| `main.py` | CLI, rendering the engine's event stream |
| `doctor.py` | Audio setup diagnostics |
| `library.py` | Listing, reading, renaming and deleting transcripts |
| `web/server.py` | Loopback web UI — FastAPI + SSE over the same event bus |
| `web/static/index.html` | The whole front end: no build step, no dependencies |

`session.py` is the headless core: it prints nothing and publishes `state`, `segments`, `levels`, `notice`, `session` and `summary` events. The CLI is one consumer of that stream.

## Privacy

Transcripts are verbatim records of conversations, often including things people did not expect to be written down.

- Recording does not stop when the meeting does. Press Ctrl+C when the call ends.
- If your output directory sits in a synced folder (iCloud, Dropbox, Obsidian Sync), transcripts leave your machine.
- Recording other people may require their consent where you are.

## License

MIT — see [LICENSE](LICENSE).
