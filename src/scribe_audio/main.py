"""CLI entrypoint for gpe-sre-ai-meeting-transcribe."""

import argparse
import os
import shutil
import signal
import subprocess
import sys

from .audio import AudioCapture, find_device, list_devices
from .dedup import extract_new_text
from .speakers import label_segments
from .transcriber import Transcriber
from .writer import TranscriptWriter


def cmd_devices(args):
    """List available audio input devices."""
    devices = list_devices()
    if not devices:
        print("No input devices found.")
        return
    print(f"\n{'ID':<5} {'Name':<45} {'Ch':<5} {'Rate'}")
    print("-" * 70)
    for d in devices:
        print(f"{d['id']:<5} {d['name']:<45} {d['channels']:<5} {int(d['sample_rate'])}")
    print(f"\nTip: Use --device 'Meeting Input' to capture meeting + mic audio")


def cmd_record(args):
    """Capture audio, transcribe with Whisper, write to file."""
    # Resolve device
    device_id = None
    if args.device_id is not None:
        device_id = args.device_id
    elif args.device:
        device_id = find_device(args.device)
        if device_id is None:
            print(f"Device '{args.device}' not found. Run: scribe devices")
            sys.exit(1)
    else:
        # Try Meeting Input first, then BlackHole
        device_id = find_device("Meeting Input")
        if device_id is None:
            device_id = find_device("BlackHole")
        if device_id is None:
            print("No device specified. Run: scribe devices")
            sys.exit(1)

    # Init components
    raw_dir = args.raw or os.environ.get("SCRIBE_RAW_DIR", "./raw")
    capture = AudioCapture(
        device_id=device_id,
        window_duration=args.window,
        step_duration=args.step,
        mic_channel=args.mic_channel,
    )
    transcriber = Transcriber(
        model_size=args.model,
        language=args.language,
    )
    writer = TranscriptWriter(raw_dir=raw_dir)

    # Load transcription model
    transcriber.load()

    # Graceful shutdown
    running = True
    def handle_signal(sig, frame):
        nonlocal running
        print("\n[main] Stopping...")
        running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Start capture loop
    capture.start()
    use_speakers = not args.no_speakers and capture.channels > 1
    if use_speakers:
        print(f"[main] Speaker detection: ON (mic=channel {args.mic_channel})")
    else:
        print(f"[main] Speaker detection: OFF (single channel or disabled)")
    print(f"[main] Recording... Press Ctrl+C to stop.\n")

    prev_text = ""
    try:
        while running:
            buffers = capture.get_window(timeout=args.step + 5)
            if buffers is None:
                continue

            mono = buffers["mono"]

            if use_speakers:
                # Transcribe with timestamps, then label by channel energy
                text_segments = transcriber.transcribe_segments(mono)
                if not text_segments:
                    continue

                labeled = label_segments(
                    text_segments, buffers["mic"], buffers["meeting"],
                    sample_rate=capture.sample_rate,
                    debug=args.debug,
                )

                if labeled:
                    full_text = " ".join(s["text"] for s in labeled)
                    new_text = extract_new_text(prev_text, full_text)
                    if new_text:
                        new_segments = _find_new_segments(labeled, new_text)
                        if new_segments:
                            writer.write_speaker_segments(new_segments)
                    prev_text = full_text
            else:
                # Plain mode: no speaker labels
                full_text = transcriber.transcribe(mono)
                if not full_text:
                    continue

                new_text = extract_new_text(prev_text, full_text)
                if new_text:
                    writer.write(new_text)

                prev_text = full_text
    finally:
        capture.stop()
        if writer.transcript_path:
            print(f"\n[main] Transcript saved: {writer.transcript_path}")


def _find_new_segments(labeled: list[dict], new_text: str) -> list[dict]:
    """Find which labeled segments correspond to the new (non-overlapping) text.

    Walks through segments accumulating text from the end until we've
    covered all of new_text. This avoids bag-of-words matching issues.
    """
    if not new_text or not labeled:
        return labeled

    # Build cumulative text from end to find where new_text starts
    new_words = new_text.lower().split()
    if not new_words:
        return []

    # Find the first segment that contains the start of new_text
    # by looking for the first few words of new_text in each segment
    search_phrase = " ".join(new_words[:5]).lower()

    for i, seg in enumerate(labeled):
        seg_text_lower = seg["text"].lower()
        if search_phrase in seg_text_lower:
            # Found the segment where new text starts
            # Include this segment and all after it
            result = labeled[i:]
            # Trim the first segment to only include new text
            if i < len(labeled):
                # Find where in this segment the new text starts
                pos = seg_text_lower.find(search_phrase)
                if pos > 0:
                    trimmed_text = seg["text"][pos:]
                    result[0] = {**seg, "text": trimmed_text}
            return result

    # Fallback: if we can't find the exact match, return last segment(s)
    # that roughly cover the new_text length
    total_new_words = len(new_words)
    result = []
    word_count = 0
    for seg in reversed(labeled):
        result.insert(0, seg)
        word_count += len(seg["text"].split())
        if word_count >= total_new_words:
            break

    return result


SUMMARY_PROMPT = """\
Summarise this meeting transcript. Provide:

## Summary
A concise 2-3 sentence overview of what the meeting was about.

## Key Discussion Points
- Bullet points of the main topics discussed

## Decisions Made
- Any decisions that were agreed upon (or "None" if no clear decisions)

## Action Items
- [ ] Action item with owner if mentioned (or "None" if no clear actions)

## Notable Quotes
- Any important or noteworthy quotes from participants
"""


def cmd_summarize(args):
    """Summarize a transcript file using auggie CLI."""
    transcript_path = args.file

    if not os.path.isfile(transcript_path):
        print(f"[summarize] Error: file not found: {transcript_path}")
        sys.exit(1)

    # Check auggie is installed
    if not shutil.which("auggie"):
        print("[summarize] Error: 'auggie' CLI not found.")
        print("  Install: npm install -g @augmentcode/auggie")
        print("  Info: https://www.augmentcode.com/product/CLI")
        sys.exit(1)

    # Read transcript
    with open(transcript_path, "r") as f:
        transcript = f.read()

    if not transcript.strip():
        print("[summarize] Error: transcript file is empty")
        sys.exit(1)

    prompt = args.prompt or SUMMARY_PROMPT

    print(f"[summarize] Summarizing: {transcript_path}")
    print(f"[summarize] Transcript length: {len(transcript)} chars\n")

    # Pipe transcript to auggie
    cmd = ["auggie", "--print", "--quiet", prompt]
    try:
        result = subprocess.run(
            cmd,
            input=transcript,
            text=True,
            capture_output=not args.stream,
        )
        if not args.stream and result.stdout:
            print(result.stdout)

        # Optionally save to file
        if args.output:
            output_text = result.stdout if not args.stream else ""
            if output_text:
                with open(args.output, "w") as f:
                    f.write(output_text)
                print(f"\n[summarize] Saved to: {args.output}")

    except KeyboardInterrupt:
        print("\n[summarize] Cancelled")
    except FileNotFoundError:
        print("[summarize] Error: 'auggie' command not found")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="scribe",
        description="Real-time AI meeting transcription with speaker detection, powered by local Whisper",
    )
    sub = parser.add_subparsers(dest="command")

    # --- devices ---
    sub.add_parser("devices", help="List available audio input devices")

    # --- record ---
    rec = sub.add_parser("record", help="Start recording and transcribing")
    rec.add_argument("--device", "-d", default=None,
                     help="Device name (partial match, default: Meeting Input or BlackHole)")
    rec.add_argument("--device-id", type=int, default=None,
                     help="Device ID (from 'devices' command)")
    rec.add_argument("--raw", "-r", default=None,
                     help="Output directory (default: $SCRIBE_RAW_DIR or ./raw)")
    rec.add_argument("--model", "-m", default="small",
                     help="Whisper model: tiny, base, small, medium, large-v3 (default: small)")
    rec.add_argument("--language", "-l", default="en",
                     help="Language code (default: en)")
    rec.add_argument("--window", "-w", type=float, default=30.0,
                     help="Transcription window in seconds (default: 30)")
    rec.add_argument("--step", "-s", type=float, default=15.0,
                     help="Step between transcriptions in seconds (default: 15)")
    rec.add_argument("--mic-channel", type=int, default=0,
                     help="Which input channel is your mic (default: 0)")
    rec.add_argument("--no-speakers", action="store_true",
                     help="Disable speaker detection (plain text mode)")
    rec.add_argument("--debug", action="store_true",
                     help="Print channel energy levels for tuning")

    # --- summarize ---
    summ = sub.add_parser("summarize", help="Summarize a transcript using auggie AI")
    summ.add_argument("file", help="Path to transcript markdown file")
    summ.add_argument("--output", "-o", default=None,
                      help="Save summary to file")
    summ.add_argument("--prompt", "-p", default=None,
                      help="Custom prompt (default: meeting summary template)")
    summ.add_argument("--stream", action="store_true",
                      help="Stream output in real-time (default: wait for full response)")

    args = parser.parse_args()

    if args.command == "devices":
        cmd_devices(args)
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "summarize":
        cmd_summarize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
