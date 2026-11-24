"""Diagnostics for the capture setup.

The checks no longer require a specific Aggregate Device. scribe opens the
microphone and the loopback as two separate streams, so the only things that
must be true are:

1. A loopback device exists (BlackHole).
2. System output is routed so that the loopback receives a copy of the audio.
3. The loopback actually passes audio.
4. Some microphone is available.

The routing check is the one that matters: a mis-routed output produces a
recording containing only your own voice, which is how a real 36-minute
interview was captured with the other side entirely missing.
"""

from __future__ import annotations

import subprocess
import threading
import time

from pathlib import Path

import numpy as np

from .capture import find_device, get_system_audio, list_devices

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"

LOOPBACK_NAME = "BlackHole"
ROUTING_HINTS = ("blackhole", "meeting capture", "multi-output")


def _test_loopback(device_name: str, seconds: float = 2.0) -> float:
    """Play a system sound and measure what the loopback receives."""
    import sounddevice as sd

    def play():
        time.sleep(0.3)
        subprocess.run(
            ["afplay", "/System/Library/Sounds/Ping.aiff"], capture_output=True
        )

    thread = threading.Thread(target=play, daemon=True)
    thread.start()
    audio = sd.rec(
        int(seconds * 48000), samplerate=48000, channels=2,
        device=device_name, dtype="float32",
    )
    sd.wait()
    thread.join(timeout=2)
    return float(np.sqrt(np.mean(audio**2)))


def measure_microphone(device_id: int, seconds: float = 4.0) -> dict:
    """Measure a microphone's quiet floor against its level while speaking.

    Speaker attribution works on how far a channel rises above its *own* noise
    floor, so this reports the same quantity the detector uses — which makes
    "my voice is not being labelled" answerable without recording a meeting.
    """
    import sounddevice as sd

    audio = sd.rec(int(seconds * 16000), samplerate=16000, channels=1,
                   device=device_id, dtype="float32")
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    n = 3200  # 0.2s slices, matching the detector
    usable = audio[: (audio.size // n) * n].reshape(-1, n)
    rms = np.sqrt(np.mean(np.square(usable, dtype=np.float64), axis=1))

    floor = float(np.percentile(rms, 5))
    loud = float(np.percentile(rms, 95))
    reference = max(floor, 1e-5)
    return {
        "floor": floor,
        "loud": loud,
        "prominence_db": 20.0 * np.log10(max(loud, 1e-12) / reference),
    }


# Long enough that the synthesiser fills a 30-second window on its own, with
# its own sentence pauses. Tiling a shorter clip instead produced thirty
# seconds of unbroken speech, which is not what a meeting sounds like: it made
# `small` measure 21.8s against 4.5s for the same text at realistic density,
# and recommended `tiny` on a machine that runs `small` comfortably.
SPEECH_SAMPLE = (
    "The migration project is going reasonably well so far. We finished the "
    "first phase last night and the team is now looking at the second one. "
    "There are a few open questions about the database layer, particularly "
    "around how we handle the cutover without any downtime at all. "
    "I think we should go through the rollback plan in more detail before "
    "anyone commits to a date, because the last time we attempted something "
    "along these lines it took considerably longer than we expected. "
    "The other thing worth raising is the monitoring story. At the moment we "
    "have very little visibility into the replication lag, and without that "
    "we would not know the cutover had gone wrong until users told us."
)

# Fraction of the window the measurement clip should actually contain speech.
# Meetings have pauses and the voice-activity filter skips them before the
# encoder runs — measured here, a near-silent window costs 0.00s against
# 4.59s for speech — so a wall-to-wall clip overstates the real cost.
SPEECH_DENSITY = 0.8

# Fastest first, so a slow machine has produced a usable recommendation before
# it reaches the models that will disappoint it.
BENCHMARK_MODELS = ("tiny", "small", "large-v3-turbo", "medium", "large-v3")

# Worst to best, which is a *different* order from the one above and must stay
# separate: large-v3-turbo measured faster than medium (7.33s against 9.14s on
# the same window) as well as being the better model, so ranking by speed
# would recommend medium over a model that beats it on both counts.
MODEL_QUALITY = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")


def _speech_clip(seconds: float, sample_rate: int = 16000):
    """Real speech of about ``seconds`` length, synthesised locally.

    Benchmarking on noise is worthless: the VAD strips it before the encoder
    runs, so every model looks instant. Measured here, a near-silent window
    costs 0.00s against 4.59s for speech on `small`. macOS ships a speech
    synthesiser, so a representative clip needs no bundled audio file.
    """
    import subprocess
    import tempfile
    import wave

    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        aiff = Path(tmp) / "s.aiff"
        wav = Path(tmp) / "s.wav"
        subprocess.run(["say", "-o", str(aiff), SPEECH_SAMPLE],
                       check=True, capture_output=True, timeout=60)
        subprocess.run(["afconvert", "-f", "WAVE", "-d",
                        f"LEI16@{sample_rate}", "-c", "1", str(aiff), str(wav)],
                       check=True, capture_output=True, timeout=60)
        with wave.open(str(wav)) as w:
            audio = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                     .astype(np.float32) / 32768.0)

    want = int(seconds * sample_rate)
    speech_frames = int(want * SPEECH_DENSITY)
    if audio.size > speech_frames:
        audio = audio[:speech_frames]
    elif audio.size < speech_frames:
        audio = np.tile(audio, int(np.ceil(speech_frames / audio.size)))
        audio = audio[:speech_frames]

    clip = np.zeros(want, dtype=np.float32)
    clip[:audio.size] = audio
    return clip


def benchmark_models(window: float = 30.0, step: float = 15.0,
                     models=BENCHMARK_MODELS,
                     backend: str = "cpu") -> list[dict]:
    """Time one window per model and say which can keep up on this machine.

    The budget is the step, not the window: a window is produced every step
    seconds, so transcription has that long to finish one. At the defaults
    that is 30 seconds of audio every 15, which is why keeping up needs twice
    realtime and why a model measured at exactly 2.0x does not.
    """
    import time

    from .transcriber import Transcriber

    audio = _speech_clip(window)
    results = []
    for name in models:
        try:
            model = Transcriber(model_size=name, backend=backend)
            model.load()
            model.transcribe_segments(audio)            # warm the graph
            start = time.perf_counter()
            model.transcribe_segments(audio)
            elapsed = time.perf_counter() - start
            results.append({
                "model": name,
                "seconds": elapsed,
                "realtime_factor": window / elapsed if elapsed else float("inf"),
                "keeps_up": elapsed < step,
                "headroom": step / elapsed if elapsed else float("inf"),
            })
        except Exception as exc:
            results.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
    return results


def recommend_model(results: list[dict], safety: float = 1.5) -> str | None:
    """The best model with real headroom, not merely the best that fits.

    A model that only just fits will stop fitting: these numbers come from an
    idle machine, and a passively cooled laptop loses speed to throttling once
    a meeting has been running for a while.
    """
    usable = [
        r for r in results
        if not r.get("error") and r.get("headroom", 0) >= safety
    ]
    if not usable:
        return None
    quality = {name: i for i, name in enumerate(MODEL_QUALITY)}
    return max(usable, key=lambda r: quality.get(r["model"], -1))["model"]


def run_doctor(benchmark: bool = False, window: float = 30.0,
               step: float = 15.0, backend: str = "cpu") -> None:
    print(f"\n{BOLD}scribe doctor{RESET}\n")
    total = 6 if benchmark else 5
    failures = 0

    # 1. Loopback device -----------------------------------------------------
    print(f"{BOLD}[1/{total}] Loopback device{RESET}")
    loopback = find_device(LOOPBACK_NAME, "input")
    if loopback:
        print(f"  {PASS}  {loopback.name} (device {loopback.id}, "
              f"{loopback.max_input_channels} in)")
    else:
        print(f"  {FAIL}  No '{LOOPBACK_NAME}' input device found")
        print("         Install it: brew install blackhole-2ch")
        failures += 1

    # 2. Output routing ------------------------------------------------------
    print(f"\n{BOLD}[2/{total}] System output routing{RESET}")
    output = get_system_audio()["output"]
    out_name = output.name if output else "None"
    print(f"  Output: {BOLD}{out_name}{RESET}")
    routed = bool(output) and any(h in out_name.lower() for h in ROUTING_HINTS)
    if routed:
        print(f"  {PASS}  Output reaches the loopback")
    else:
        print(f"  {WARN}  Output does not reach the loopback — meeting audio")
        print("         will not be captured.")
        print("         Create a Multi-Output Device containing your headset and")
        print("         BlackHole, then select it as the system output.")

    print(f"\n  {BOLD}Also check inside your meeting app:{RESET}")
    print("    Zoom/Teams Speaker must be 'Same as System', not your headset.")
    print("    Selecting the headset there bypasses the loopback entirely and")
    print("    is the most common cause of a one-sided transcript.")

    # 3. Loopback carries audio ---------------------------------------------
    print(f"\n{BOLD}[3/{total}] Loopback passes audio{RESET}")
    if loopback:
        try:
            rms = _test_loopback(loopback.name)
            if rms > 0.0005:
                print(f"  {PASS}  Audio detected (RMS {rms:.5f})")
            else:
                print(f"  {WARN}  Silence (RMS {rms:.5f})")
                print("         Fix the routing above, then re-run.")
        except Exception as exc:
            print(f"  {WARN}  Could not test: {exc}")
    else:
        print(f"  {FAIL}  Skipped — no loopback device")
        failures += 1

    # 4. Microphone ----------------------------------------------------------
    print(f"\n{BOLD}[4/{total}] Microphone{RESET}")
    default_in = get_system_audio()["input"]
    if default_in:
        print(f"  {PASS}  Default input: {default_in.name}")
    else:
        print(f"  {WARN}  No default input device — speaker labelling will be off")

    others = [
        d for d in list_devices("input")
        if LOOPBACK_NAME.lower() not in d.name.lower()
        and (default_in is None or d.id != default_in.id)
    ]
    if others:
        print("  Other inputs available for --mic:")
        for d in others[:8]:
            print(f"    - {d.name}")

    # 5. Microphone level ----------------------------------------------------
    print(f"\n{BOLD}[5/{total}] Microphone level{RESET}")
    if default_in is not None:
        print(f"  {BOLD}Speak normally for 4 seconds, starting now...{RESET}")
        try:
            m = measure_microphone(default_in.id)
            margin = m["prominence_db"]
            print(f"  Quiet floor: {m['floor']:.6f}   While speaking: {m['loud']:.6f}")
            if margin >= 20:
                print(f"  {PASS}  Your voice sits {margin:.0f} dB above the floor")
            elif margin >= 8:
                print(f"  {WARN}  Only {margin:.0f} dB above the floor — labelling will")
                print(f"         work but is marginal. Move the mic closer.")
            else:
                print(f"  {FAIL}  Just {margin:.0f} dB above the floor — your speech will")
                print(f"         not be labelled 'You'.")
                print(f"         Check the headset's inline mute switch and the input")
                print(f"         volume in System Settings > Sound > Input.")
                failures += 1
        except Exception as exc:
            print(f"  {WARN}  Could not measure: {exc}")
    else:
        print(f"  {WARN}  Skipped — no default input device")

    # 6. Transcription speed -------------------------------------------------
    if benchmark:
        print(f"\n{BOLD}[6/{total}] Transcription speed on this machine{RESET}")
        print(f"  Backend: {backend}")
        print(f"  A {window:.0f}s window is produced every {step:.0f}s, so each "
              f"one must finish\n  inside {step:.0f}s to keep up live. "
              f"Measuring, this takes a minute...")
        try:
            results = benchmark_models(window=window, step=step,
                                       backend=backend)
            print()
            print(f"  {'model':16s} {'per window':>11s} {'xRT':>7s}  verdict")
            for r in results:
                if r.get("error"):
                    print(f"  {r['model']:16s} {r['error']}")
                    continue
                verdict = (f"{GREEN}keeps up{RESET}" if r["keeps_up"]
                           else f"{RED}falls behind{RESET}")
                print(f"  {r['model']:16s} {r['seconds']:10.2f}s "
                      f"{r['realtime_factor']:6.1f}x  {verdict}")

            best = recommend_model(results)
            print()
            if best is None:
                print(f"  {WARN}  No model keeps up comfortably here. Recording "
                      f"still loses\n         nothing — audio is spooled to disk "
                      f"and transcription\n         finishes after you stop.")
            else:
                print(f"  {PASS}  Recommended: {BOLD}--model {best}{RESET}")
                print(f"         Chosen with headroom to spare: these numbers "
                      f"come from an\n         idle machine, and a fanless "
                      f"laptop slows down once a\n         meeting has been "
                      f"running for a while.")
        except FileNotFoundError:
            print(f"  {WARN}  Skipped — needs macOS 'say' and 'afconvert'")
        except Exception as exc:
            print(f"  {WARN}  Could not measure: {exc}")

    # Summary ----------------------------------------------------------------
    print(f"\n{BOLD}{'=' * 56}{RESET}")
    if failures == 0 and routed:
        print(f"{GREEN}{BOLD}Ready to record.{RESET}")
    elif failures == 0:
        print(f"{YELLOW}{BOLD}Usable, but meeting audio is not routed yet.{RESET}")
    else:
        print(f"{RED}{BOLD}Setup incomplete — see above.{RESET}")

    print(f"\n{BOLD}Usage{RESET}")
    print("  scribe record                       # system default mic + BlackHole")
    print("  scribe record --mic 'Plantronics'   # pick a microphone explicitly")
    print("  scribe record --no-mic              # meeting audio only")
    print(f"\n{BOLD}If audio behaves strangely after reconnecting a headset{RESET}")
    print("  sudo killall coreaudiod             # restarts automatically")
    print("  brew reinstall blackhole-2ch        # then restart macOS")
    print()
