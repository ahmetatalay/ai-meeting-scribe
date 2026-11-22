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


def run_doctor() -> None:
    print(f"\n{BOLD}scribe doctor{RESET}\n")
    failures = 0

    # 1. Loopback device -----------------------------------------------------
    print(f"{BOLD}[1/5] Loopback device{RESET}")
    loopback = find_device(LOOPBACK_NAME, "input")
    if loopback:
        print(f"  {PASS}  {loopback.name} (device {loopback.id}, "
              f"{loopback.max_input_channels} in)")
    else:
        print(f"  {FAIL}  No '{LOOPBACK_NAME}' input device found")
        print("         Install it: brew install blackhole-2ch")
        failures += 1

    # 2. Output routing ------------------------------------------------------
    print(f"\n{BOLD}[2/5] System output routing{RESET}")
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
    print(f"\n{BOLD}[3/5] Loopback passes audio{RESET}")
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
    print(f"\n{BOLD}[4/5] Microphone{RESET}")
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
    print(f"\n{BOLD}[5/5] Microphone level{RESET}")
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
