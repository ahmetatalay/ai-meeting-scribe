"""CLI entrypoint for ai-meeting-scribe."""

from __future__ import annotations

import argparse
import os
import queue
import signal
import sys
import threading

from .capture import find_device, get_system_audio, list_devices
from .doctor import run_doctor
from .session import (
    STATE_ERROR,
    STATE_IDLE,
    STATE_LOADING,
    STATE_RECORDING,
    RecordingSession,
    SessionConfig,
)


def cmd_devices(args) -> None:
    """List audio devices."""
    for kind in ("input", "output"):
        devices = list_devices(kind)
        print(f"\n{kind.title()} devices")
        print(f"{'ID':<5} {'Name':<45} {'Ch':<5} {'Rate'}")
        print("-" * 70)
        if not devices:
            print("(none found)")
            continue
        for d in devices:
            channels = d.max_input_channels if kind == "input" else d.max_output_channels
            print(f"{d.id:<5} {d.name[:44]:<45} {channels:<5} {int(d.default_samplerate)}")

    print("\nscribe records two independent streams:")
    print("  --mic     your microphone      (default: system default input)")
    print("  --system  meeting audio        (default: BlackHole)")


def _resolve_devices(args) -> tuple[int | None, int | None]:
    """Work out which device index to use for each stream."""
    mic_id = None
    if not args.no_mic:
        if args.mic:
            dev = find_device(args.mic, "input")
            if dev is None:
                print(f"[main] Microphone '{args.mic}' not found. Run: scribe devices")
                sys.exit(1)
            mic_id = dev.id
        else:
            default_in = get_system_audio()["input"]
            if default_in is not None:
                mic_id = default_in.id
                print(f"[main] Microphone: {default_in.name} (system default)")
            else:
                print("[main] No default input device; recording meeting audio only.")

    system_id = None
    if not args.no_system:
        dev = find_device(args.system, "input")
        if dev is None:
            print(f"[main] System audio device '{args.system}' not found.")
            print("       Install BlackHole, or pass --system with another name,")
            print("       or use --no-system to record only your microphone.")
            sys.exit(1)
        system_id = dev.id

    if mic_id is None and system_id is None:
        print("[main] Nothing to record: both --no-mic and --no-system were given.")
        sys.exit(1)

    return mic_id, system_id


def _report_routing() -> bool:
    """Print the current output routing. Returns True if it looks capturable."""
    sys_audio = get_system_audio()
    out = sys_audio["output"]
    out_name = out.name if out else "None"
    print(f"[main] System output: {out_name}")

    looks_routed = bool(out) and any(
        hint in out_name.lower() for hint in ("meeting capture", "blackhole", "multi-output")
    )
    if not looks_routed:
        print("[main] ⚠ System output is not routed through BlackHole.")
        print("       Meeting audio will not be captured until it is.")
    return looks_routed


def _warn_silent_loopback() -> None:
    print()
    print("[main] ⚠ No meeting audio has arrived yet.")
    print("       The most common cause is an app-level output override:")
    print("       in Zoom/Teams set Speaker to 'Same as System' (not your")
    print("       headset directly), with system output on Meeting Capture.")
    print("       Your own microphone is still being recorded.")
    print()


def cmd_record(args) -> None:
    """Capture, transcribe and write a transcript until interrupted."""
    mic_id, system_id = _resolve_devices(args)

    config = SessionConfig(
        mic_device=mic_id,
        system_device=system_id,
        model=args.model,
        language=None if args.language in ("auto", "") else args.language,
        window=args.window,
        step=args.step,
        raw_dir=args.raw or os.environ.get("SCRIBE_RAW_DIR", "./raw"),
        title=args.title,
        mic_margin=args.mic_margin,
        use_speakers=not args.no_speakers,
        keep_filler=args.keep_filler,
    )

    _report_routing()
    session = RecordingSession(config)
    events = session.bus.subscribe()

    stopping = {"flag": False}

    def handle_signal(sig, frame):
        if stopping["flag"]:
            return
        stopping["flag"] = True
        print("\n[main] Stopping...")
        threading.Thread(target=session.stop, daemon=True).start()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    session.start()
    _render_events(session, events)


def _render_events(session: RecordingSession, events: queue.Queue) -> None:
    """Print the session's event stream until it finishes."""
    while True:
        try:
            event = events.get(timeout=0.25)
        except queue.Empty:
            if not session.is_running and session.state != STATE_IDLE:
                break
            continue

        kind = event.get("type")
        if kind == "state":
            _render_state(event)
        elif kind == "segments":
            for seg in event["segments"]:
                print(f"  {seg['speaker']}: {seg['text']}")
        elif kind == "notice":
            if event["level"] == "info":
                print(f"[main] {event['message']}")
            else:
                marker = "✗" if event["level"] == "error" else "⚠"
                print(f"\n[main] {marker} {event['message']}\n")
        elif kind == "session" and event.get("markdown"):
            print(f"[main] Transcript: {event['markdown']}")
        elif kind == "summary":
            _render_summary(event)
            break


def _render_state(event: dict) -> None:
    value = event["value"]
    if value == STATE_LOADING:
        print(f"[main] Loading model '{event.get('model')}'...")
    elif value == STATE_RECORDING:
        speakers = "ON" if event.get("speakers") else "OFF"
        print(f"[main] Speaker detection: {speakers}")
        print("[main] Recording... Press Ctrl+C to stop.\n")
    elif value == STATE_ERROR:
        print(f"[main] Failed: {event.get('error', 'unknown error')}")


def _render_summary(event: dict) -> None:
    stats = event["stats"]
    if not event.get("markdown"):
        print("\n[main] Nothing was transcribed.")
        return

    print(f"\n[main] Transcript: {event['markdown']}")
    print(f"[main] Metadata:   {event['sidecar']}")
    speakers = ", ".join(f"{k} {v}" for k, v in stats["speakers"].items())
    print(f"[main] {stats['segments']} segments, {stats['words']} words"
          + (f" ({speakers})" if speakers else ""))
    if stats["rejected_segments"]:
        print(f"[main] Dropped {stats['rejected_segments']} low-confidence segment(s).")
    if stats["collapsed_segments"]:
        print(f"[main] Collapsed {stats['collapsed_segments']} repetition loop(s).")
    if stats["dropped_seconds"]:
        print(f"[main] Lost {stats['dropped_seconds']}s of audio to slow transcription.")


def cmd_web(args) -> None:
    """Run the local web UI."""
    try:
        from .web.server import DEFAULT_PORT, serve
    except ImportError:
        print("[web] The web UI needs the 'web' extra:")
        print("        uv sync --extra web          (in a checkout)")
        print("        uv tool install '.[web]'     (installed globally)")
        sys.exit(1)

    serve(
        raw_dir=args.raw or os.environ.get("SCRIBE_RAW_DIR", "./raw"),
        port=args.port or DEFAULT_PORT,
        model=args.model,
        open_browser=not args.no_browser,
    )


def _deprecated(parser) -> None:
    """Explain the retired aggregate-device flags instead of 'unrecognized argument'."""
    for flag in ("--device", "--device-id", "--mic-channel"):
        parser.add_argument(flag, dest="_retired", default=argparse.SUPPRESS,
                            help=argparse.SUPPRESS)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="scribe",
        description="Real-time AI meeting transcription with speaker detection, "
                    "powered by local Whisper",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("devices", help="List audio input and output devices")
    sub.add_parser("doctor", help="Diagnose the audio setup")

    web = sub.add_parser("web", help="Open the local web UI (loopback only)")
    web.add_argument("--port", type=int, default=None, help="Port (default: 8765)")
    web.add_argument("--raw", "-r", default=None,
                     help="Output directory (default: $SCRIBE_RAW_DIR or ./raw)")
    web.add_argument("--model", "-m", default="small",
                     help="Default Whisper model offered in the UI")
    web.add_argument("--no-browser", action="store_true",
                     help="Do not open a browser window")

    rec = sub.add_parser("record", help="Start recording and transcribing")
    rec.add_argument("--mic", default=None,
                     help="Microphone name, partial match (default: system default input)")
    rec.add_argument("--system", default="BlackHole",
                     help="System-audio loopback device name (default: BlackHole)")
    rec.add_argument("--no-mic", action="store_true",
                     help="Record meeting audio only")
    rec.add_argument("--no-system", action="store_true",
                     help="Record your microphone only")
    rec.add_argument("--raw", "-r", default=None,
                     help="Output directory (default: $SCRIBE_RAW_DIR or ./raw)")
    rec.add_argument("--title", default=None, help="Meeting title")
    rec.add_argument("--model", "-m", default="small",
                     help="Whisper model: tiny, base, small, medium, large-v3")
    rec.add_argument("--language", "-l", default="en",
                     help="Language code, or 'auto' to detect (default: en)")
    rec.add_argument("--window", "-w", type=float, default=30.0,
                     help="Transcription window in seconds (default: 30)")
    rec.add_argument("--step", "-s", type=float, default=15.0,
                     help="Seconds between transcriptions (default: 15)")
    rec.add_argument("--mic-margin", type=float, default=8.0,
                     help="dB above the mic's own noise floor that counts as your "
                          "speech (default: 8)")
    rec.add_argument("--no-speakers", action="store_true",
                     help="Disable speaker labelling")
    rec.add_argument("--keep-filler", action="store_true",
                     help="Keep low-confidence filler segments ('You', 'Thank you.')")
    rec.add_argument("--debug", action="store_true",
                     help="Print per-segment channel energy")
    _deprecated(rec)

    args = parser.parse_args()

    if getattr(args, "_retired", None) is not None:
        print("--device, --device-id and --mic-channel were removed.")
        print("scribe no longer needs an Aggregate Device: it opens the microphone")
        print("and the loopback as two separate streams.")
        print("  Old: scribe record --device 'Meeting Input' --mic-channel 0")
        print("  New: scribe record --mic 'Plantronics' --system 'BlackHole'")
        sys.exit(2)

    if args.command == "devices":
        cmd_devices(args)
    elif args.command == "doctor":
        run_doctor()
    elif args.command == "record":
        cmd_record(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
