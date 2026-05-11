# Scribe — AI Meeting Transcriber

Scribe is an AI-powered meeting transcription tool that runs entirely on your local machine. It captures live audio from any meeting platform including **Microsoft Teams**, **Zoom**, **Slack Huddle**, **Google Meet** and more, transcribes speech in real-time using OpenAI's Whisper model, and automatically identifies who's speaking. No cloud services, no API costs, no data leaving your laptop. Everything stays private and secure.

Generate near real-time AI summaries during or after meetings with key discussion points, decisions, and action items using [Auggie CLI](https://www.augmentcode.com/product/CLI) by [Augment Code](https://www.augmentcode.com).

## Technology

| Component | Technology | Details |
|-----------|-----------|---------|
| **Speech-to-Text** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | CTranslate2 re-implementation of OpenAI's [Whisper](https://github.com/openai/whisper) — 4x faster inference on Apple Silicon |
| **Models** | [OpenAI Whisper](https://huggingface.co/Systran) (via Hugging Face) | Open-source (MIT license), free to download, runs 100% locally |
| **Audio Capture** | [sounddevice](https://python-sounddevice.readthedocs.io/) + [BlackHole](https://existential.audio/blackhole/) | PortAudio bindings for Python + macOS virtual audio loopback |
| **Speaker Detection** | Channel-based energy analysis | Physical mic/meeting separation — no ML diarization needed |
| **Summarization** | [Auggie CLI](https://www.augmentcode.com/product/CLI) (optional) | AI-powered summary via Augment Code |

### About Hugging Face

The Whisper models are hosted on [Hugging Face Hub](https://huggingface.co/Systran) and downloaded automatically on first use:

- **No account required** — models download without authentication
- **No API key needed** — everything runs locally after download
- **No cost** — all models are open-source (MIT license)
- **No data sent to cloud** — transcription happens entirely on your machine
- **One-time download** — models are cached in `~/.cache/huggingface/`

> The `HF_TOKEN` warning you may see is optional — it only speeds up downloads slightly. You can safely ignore it.

## How it works

```
Meeting audio (Teams/Zoom/Slack Huddle etc.)
    ↓
Multi-Output Device → you hear + BlackHole captures
    ↓
Aggregate Device → combines BlackHole (meeting) + your mic
    ↓
scribe captures from Aggregate Device
    ↓
faster-whisper transcribes locally (30s overlapping windows)
    ↓
Channel energy detection → labels "You" vs "Meeting"
    ↓
raw/transcript_YYYY-MM-DD_HHMM.md (appended in real-time)
```

Output looks like:

```markdown
**You**: Hello, how are you? So today I think you will teach me how to pronounce it, right?
**Meeting**: Hello lovely students and welcome to your pronunciation training session.
**You**: Thank you very much. So you will teach me 100 everyday words.
```

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **BlackHole 2ch** installed ([download](https://existential.audio/blackhole/))
- **Python 3.11+**
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **PortAudio** (for sounddevice): `brew install portaudio`

## One-time macOS Audio Setup

Open **Audio MIDI Setup** (`Cmd + Space` → type "Audio MIDI Setup").

There are two setup modes. Choose the one that fits your needs:

### Option A: Full Setup (with speaker detection)

This captures both meeting audio AND your microphone, labeling who said what.

#### 1. Create Multi-Output Device (output routing)

This lets you **hear the meeting AND route audio to BlackHole** simultaneously.

1. Click **+** → **Create Multi-Output Device**
2. Tick ✅ your headset or speakers (e.g. AirPods, Jabra, any USB headset)
3. Tick ✅ **BlackHole 2ch**
4. Make sure your **headset is listed FIRST** (drag to reorder)
5. Right-click → rename to **"Meeting Capture"**

#### 2. Create Aggregate Device (input capture)

This combines your mic + meeting audio into one input with separate channels.

1. Click **+** → **Create Aggregate Device**
2. Tick ✅ your microphone (e.g. AirPods, USB headset, or MacBook Air Microphone)
3. Tick ✅ **BlackHole 2ch**
4. Right-click → rename to **"Meeting Input"**

This creates a 3-channel device:
- **Channel 0** = your mic
- **Channels 1-2** = BlackHole (meeting audio)

#### 3. Before each meeting

- Sound **output** → **Meeting Capture** (Multi-Output Device)
- The tool records from **Meeting Input** (Aggregate Device) automatically

#### 4. After the meeting

- Switch sound output back to your headset or speakers

### Option B: Simple Setup (meeting audio only, no speaker detection)

If you don't need "You" vs "Meeting" labels, or want a more reliable setup that doesn't depend on a specific headset/mic:

#### 1. Create Multi-Output Device (same as Option A)

Follow steps above to create **"Meeting Capture"**.

#### 2. Before each meeting

- Sound **output** → **Meeting Capture**
- Run with `--no-speakers` flag:

```bash
scribe record --device "BlackHole" --no-speakers
```

This captures only meeting audio (what others say). Your voice won't be recorded but the setup is simpler and doesn't break when you switch headsets.

> **Note on headset switching (Option A):** macOS Aggregate and Multi-Output devices can break when you disconnect/reconnect a headset (USB unplug, Bluetooth drop). If audio stops working, open Audio MIDI Setup and re-add your headset to the devices. Option B avoids this issue entirely.

## Quick Start

```bash
cd gpe-sre-ai-meeting-transcribe

# Install
uv sync

# List audio devices
uv run scribe devices

# Start recording with speaker detection
uv run scribe record --device "Meeting Input"

# Ctrl+C to stop — transcript saved to raw/
```

## Usage

```bash
# List available input devices
scribe devices

# Record with speaker detection (recommended)
scribe record --device "Meeting Input"

# Custom output directory
scribe record --device "Meeting Input" --raw ~/Desktop/transcripts

# Better quality model (default: small)
scribe record --device "Meeting Input" --model medium

# Non-English meeting
scribe record --device "Meeting Input" --language tr

# Disable speaker detection (plain text mode)
scribe record --device "Meeting Input" --no-speakers

# Debug: see channel energy levels for tuning
scribe record --device "Meeting Input" --debug

# Adjust overlapping window (default: 30s window, 15s step)
scribe record --device "Meeting Input" --window 30 --step 15

# If your mic is on a different channel
scribe record --device "Meeting Input" --mic-channel 0
```

## Summarize Transcripts

After a meeting, use the `summarize` command to generate an AI summary via [Auggie CLI](https://www.augmentcode.com/product/CLI).

### Prerequisites

```bash
npm install -g @augmentcode/auggie
```

### Usage

```bash
# Summarize a transcript
scribe summarize raw/transcript_2026-05-05_2141.md

# Save summary to file
scribe summarize raw/transcript_2026-05-05_2141.md -o summary.md

# Stream output in real-time
scribe summarize raw/transcript_2026-05-05_2141.md --stream

# Custom prompt
scribe summarize raw/transcript_2026-05-05_2141.md -p "List only action items"
```

The default summary includes: **Summary**, **Key Discussion Points**, **Decisions Made**, **Action Items**, and **Notable Quotes**.

## Install globally (run from anywhere)

```bash
cd gpe-sre-ai-meeting-transcribe
uv tool install .

# Now run from any directory:
scribe record --device "Meeting Input" --raw ~/Desktop/transcripts
scribe summarize ~/Desktop/transcripts/transcript_2026-05-05_2141.md
```

## Environment Variables

```bash
# Add to ~/.zshrc for persistent config
export SCRIBE_RAW_DIR=~/Desktop/raw
```

## Whisper Models

Models are downloaded from [Hugging Face](https://huggingface.co/Systran) on first use and cached locally. Change with `--model`:

| Model | Download | Disk Cache | Speed | Quality | Good for |
|-------|----------|-----------|-------|---------|----------|
| `tiny` | ~40MB | `~/.cache/huggingface/` | ★★★★★ | ★★ | Quick test |
| `base` | ~150MB | `~/.cache/huggingface/` | ★★★★ | ★★★ | Fast, lower accuracy |
| `small` | ~500MB | `~/.cache/huggingface/` | ★★★ | ★★★★ | **Recommended (default)** |
| `medium` | ~1.5GB | `~/.cache/huggingface/` | ★★ | ★★★★★ | Best local quality |
| `large-v3` | ~3GB | `~/.cache/huggingface/` | ★ | ★★★★★ | Maximum quality |

> All models use [CTranslate2](https://github.com/OpenNMT/CTranslate2) for optimised inference on Apple Silicon (M1/M2/M3/M4).

## How Speaker Detection Works

Instead of ML-based diarization, we use **physical channel separation**:

1. Your mic (channel 0) only picks up **your voice**
2. BlackHole (channels 1-2) only carries **meeting audio**
3. For each transcribed word, we check **which channel has energy**
4. Words are merged into sentences per speaker

This is faster and more accurate than ML diarization because it uses physical separation rather than audio similarity guessing.

## Troubleshooting

### "No input devices found"
- Install PortAudio: `brew install portaudio`
- Reinstall sounddevice: `uv pip install --force-reinstall sounddevice`

### BlackHole not showing up
- Download from https://existential.audio/blackhole/
- After install, restart Audio MIDI Setup

### No audio captured / empty transcript
- Make sure **Meeting Capture** (Multi-Output Device) is selected as sound output
- Check that your meeting app is playing audio
- Test: play music → run `scribe record --device "Meeting Input"`

### Everything labeled as "Meeting"
- Your mic energy may be too low — run with `--debug` to see energy levels
- If mic energy is below 0.001 when you speak, the threshold may need adjusting

### Poor transcription quality
- Try a larger model: `--model medium`
- Make sure the meeting audio is clear (not too quiet)

### Sentences breaking mid-word
- Increase window size: `--window 45 --step 20`

### Headset disconnected/reconnected and audio stopped working
This is a known macOS limitation with Aggregate and Multi-Output devices.
1. Open **Audio MIDI Setup**
2. Click on **Meeting Capture** and **Meeting Input**
3. Uncheck and re-check your headset/mic
4. If that doesn't work, delete the device and recreate it

To avoid this issue, use **Option B** (simple setup) which only requires BlackHole.
