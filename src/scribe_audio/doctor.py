"""Diagnostic tool for scribe audio setup.

Checks BlackHole, Aggregate/Multi-Output devices, and audio routing.
"""

import subprocess
import numpy as np
import sounddevice as sd

from .audio import get_system_audio

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
WARN = f"{YELLOW}⚠ WARN{RESET}"


def _find_device(name: str) -> dict | None:
    """Find input device by partial name match (case-insensitive)."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower() and d["max_input_channels"] > 0:
            return {"id": i, **d}
    return None


def _find_output_device(name: str) -> dict | None:
    """Find output device by partial name match."""
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if name.lower() in d["name"].lower() and d["max_output_channels"] > 0:
            return {"id": i, **d}
    return None


def _test_blackhole_loopback() -> bool:
    """Test BlackHole by playing a system sound and reading it back."""
    try:
        import threading
        import time

        def play():
            time.sleep(0.3)
            subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"],
                           capture_output=True)

        t = threading.Thread(target=play)
        t.start()
        audio = sd.rec(int(2 * 48000), samplerate=48000, channels=2,
                       device="BlackHole 2ch", dtype="float32")
        sd.wait()
        t.join()
        rms = float(np.sqrt(np.mean(audio ** 2)))
        return rms > 0.0005
    except Exception:
        return False


def run_doctor():
    """Run all diagnostic checks."""
    print(f"\n{BOLD}scribe doctor — Audio Diagnostic{RESET}\n")
    all_ok = True

    # 0. Current system audio devices
    sys_audio = get_system_audio()
    out_name = sys_audio["output"]["name"] if sys_audio["output"] else "None"
    in_name = sys_audio["input"]["name"] if sys_audio["input"] else "None"
    print(f"{BOLD}System Audio{RESET}")
    print(f"  🔊 Output: {BOLD}{out_name}{RESET}")
    print(f"  🎤 Input:  {BOLD}{in_name}{RESET}")

    # Check if output is a Meeting Capture device
    if "meeting capture" in out_name.lower():
        print(f"  {PASS}  Output is routed through Meeting Capture")
    else:
        print(f"  {WARN}  Output is NOT Meeting Capture — meeting audio won't be captured")
        print(f"         → Set Sound Output to Meeting Capture")

    print()

    # 1. BlackHole driver
    print(f"{BOLD}[1/4] BlackHole 2ch{RESET}")
    bh = _find_device("BlackHole 2ch")
    if bh:
        print(f"  {PASS}  Found: device {bh['id']}, "
              f"{bh['max_input_channels']}in/{bh['max_output_channels']}out")
    else:
        print(f"  {FAIL}  Not found — install: brew install blackhole-2ch")
        all_ok = False

    # 2. BlackHole loopback
    print(f"\n{BOLD}[2/4] BlackHole loopback test{RESET}")
    if bh:
        ok = _test_blackhole_loopback()
        if ok:
            print(f"  {PASS}  BlackHole passes audio")
        else:
            print(f"  {WARN}  No audio detected — make sure system output includes BlackHole")
            print(f"         Set output to Meeting Capture or BlackHole 2ch, then re-run")
    else:
        print(f"  {FAIL}  Skipped (BlackHole not found)")
        all_ok = False

    # 3. Meeting Input (Aggregate Device)
    print(f"\n{BOLD}[3/4] Meeting Input (Aggregate Device){RESET}")
    mi = _find_device("Meeting Input")
    if mi:
        ch = mi["max_input_channels"]
        print(f"  {PASS}  Found: device {mi['id']}, {ch} channels")
        if ch >= 3:
            print(f"  {PASS}  Channel layout: mic(0) + meeting(1,2)")
        elif ch == 2:
            print(f"  {WARN}  Only 2 channels — mic and meeting may overlap")
        else:
            print(f"  {WARN}  Only {ch} channel — speaker detection won't work")
    else:
        print(f"  {FAIL}  Not found — create in Audio MIDI Setup:")
        print(f"         1. Click + → Create Aggregate Device")
        print(f"         2. Tick your mic + BlackHole 2ch")
        print(f"         3. Rename to 'Meeting Input'")
        all_ok = False

    # 4. Meeting Capture (Multi-Output Device)
    print(f"\n{BOLD}[4/4] Meeting Capture (Multi-Output Device){RESET}")
    mc = _find_output_device("Meeting Capture")
    if mc:
        print(f"  {PASS}  Found: device {mc['id']}")
    else:
        print(f"  {FAIL}  Not found — create in Audio MIDI Setup:")
        print(f"         1. Click + → Create Multi-Output Device")
        print(f"         2. Tick your speakers/headset + BlackHole 2ch")
        print(f"         3. Rename to 'Meeting Capture'")
        all_ok = False

    # Summary
    print(f"\n{BOLD}{'=' * 50}{RESET}")
    if all_ok:
        print(f"{GREEN}{BOLD}All checks passed!{RESET}")
    else:
        print(f"{RED}{BOLD}Some checks failed — see above for fixes.{RESET}")

    # Usage hint
    print(f"\n{BOLD}Usage:{RESET}")
    print(f"  1. Set Sound Output → Meeting Capture")
    print(f"  2. scribe record --device 'Meeting Input'")
    print(f"\n{BOLD}Troubleshooting:{RESET}")
    print(f"  BlackHole broken:  brew reinstall blackhole-2ch && restart")
    print(f"  Audio daemon stuck: sudo killall coreaudiod")
    print()
