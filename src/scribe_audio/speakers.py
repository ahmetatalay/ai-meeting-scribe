"""Channel-based speaker detection.

Instead of ML diarization, we use the physical channel separation:
- Mic channel (your headset/mic) = You
- Meeting channels (BlackHole) = Others

For each transcribed segment (with timestamps), we check which channel
had more energy during that time range to determine who was speaking.
"""

import numpy as np

# Labels
SPEAKER_YOU = "You"
SPEAKER_MEETING = "Meeting"


def detect_speaker(
    mic_audio: np.ndarray,
    meeting_audio: np.ndarray,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
    mic_threshold: float = 0.001,
    debug: bool = False,
) -> str:
    """
    Detect who was speaking during a time range based on mic channel energy.

    Logic: if the mic channel has significant energy, YOU are speaking.
    The mic (headset) only picks up your voice. Meeting audio comes
    through BlackHole which doesn't feed into the mic channel.

    Args:
        mic_audio: full window mic channel audio
        meeting_audio: full window meeting channel audio
        start_sec: segment start time in seconds
        end_sec: segment end time in seconds
        sample_rate: sample rate in Hz
        mic_threshold: RMS energy threshold for mic to count as "You speaking"
        debug: print energy levels for tuning

    Returns:
        SPEAKER_YOU or SPEAKER_MEETING
    """
    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)

    # Clamp to buffer bounds
    start_idx = max(0, min(start_idx, len(mic_audio) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(mic_audio)))

    mic_slice = mic_audio[start_idx:end_idx]
    meeting_slice = meeting_audio[start_idx:end_idx]

    # RMS energy per channel
    mic_energy = np.sqrt(np.mean(mic_slice ** 2))
    meeting_energy = np.sqrt(np.mean(meeting_slice ** 2))

    # "You" if mic is active AND either:
    #   - mic is louder than meeting (you're talking over it), OR
    #   - meeting is essentially silent (you're the only one talking)
    is_you = (
        mic_energy > mic_threshold
        and (mic_energy > meeting_energy or meeting_energy < 0.0005)
    )

    if debug:
        print(f"  [{start_sec:.1f}-{end_sec:.1f}s] mic={mic_energy:.4f} "
              f"meeting={meeting_energy:.4f} → "
              f"{'YOU' if is_you else 'MEETING'}")

    return SPEAKER_YOU if is_you else SPEAKER_MEETING


def label_segments(
    text_segments: list[dict],
    mic_audio: np.ndarray,
    meeting_audio: np.ndarray,
    sample_rate: int = 16000,
    debug: bool = False,
) -> list[dict]:
    """
    Label each word with a speaker, then merge consecutive same-speaker words
    into sentence segments.

    Args:
        text_segments: word-level segments [{"start": float, "end": float, "text": str}, ...]
        mic_audio: mic channel audio for the full window
        meeting_audio: meeting channel audio for the full window
        sample_rate: sample rate in Hz

    Returns:
        [{"speaker": "You", "text": "hello world", "start": 0.5}, ...]
    """
    if not text_segments:
        return []

    # Label each word
    labeled_words = []
    for seg in text_segments:
        speaker = detect_speaker(
            mic_audio, meeting_audio,
            seg["start"], seg["end"],
            sample_rate,
            debug=debug,
        )
        labeled_words.append({
            "speaker": speaker,
            "text": seg["text"].strip(),
            "start": seg["start"],
        })

    # Smooth out isolated speaker flips:
    # If a word has a different speaker than both its neighbors, flip it
    # to match the neighbors (it's noise, not a real speaker change)
    for i in range(1, len(labeled_words) - 1):
        prev_spk = labeled_words[i - 1]["speaker"]
        next_spk = labeled_words[i + 1]["speaker"]
        curr_spk = labeled_words[i]["speaker"]
        if prev_spk == next_spk and curr_spk != prev_spk:
            labeled_words[i]["speaker"] = prev_spk

    # Second pass: smooth runs of up to 2 isolated words
    for i in range(1, len(labeled_words) - 2):
        prev_spk = labeled_words[i - 1]["speaker"]
        curr_spk = labeled_words[i]["speaker"]
        next_next_spk = labeled_words[i + 2]["speaker"] if i + 2 < len(labeled_words) else None
        if (curr_spk != prev_spk
                and labeled_words[i + 1]["speaker"] == curr_spk
                and next_next_spk == prev_spk):
            # Two isolated words surrounded by the other speaker — flip them
            labeled_words[i]["speaker"] = prev_spk
            labeled_words[i + 1]["speaker"] = prev_spk

    # Merge consecutive words from the same speaker
    merged = []
    current = {
        "speaker": labeled_words[0]["speaker"],
        "text": labeled_words[0]["text"],
        "start": labeled_words[0]["start"],
    }

    for word in labeled_words[1:]:
        if word["speaker"] == current["speaker"]:
            # Same speaker — append word
            current["text"] += " " + word["text"]
        else:
            # Speaker changed — flush and start new segment
            merged.append(current)
            current = {
                "speaker": word["speaker"],
                "text": word["text"],
                "start": word["start"],
            }

    merged.append(current)  # flush last segment
    return merged
