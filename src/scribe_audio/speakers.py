"""Speaker attribution from per-channel energy.

Two physically separate signals are available — the microphone (you) and the
system-audio loopback (everyone else) — so no ML diarization is needed to tell
your speech from theirs.

The previous implementation compared raw RMS against a hardcoded 0.001 and
decided per *word*. Measured against real captures that mislabelled almost
everything: in one 36-minute call, 97 of 97 chunks came out as "You", including
the entire other side. Two causes, both fixed here:

1. **An absolute threshold cannot work.** 0.001 RMS is below the noise floor of
   an ordinary microphone in a quiet room, so the "is the mic active" test was
   true essentially always and the decision collapsed to a raw loudness
   comparison between channels with completely different gains. Each channel is
   now calibrated against *its own* measured noise floor, and activity means
   "clearly above that floor".

2. **Comparing absolute levels across channels is meaningless.** A near-field
   headset mic is simply louder than post-mixer loopback audio, so the mic won
   every comparison. What matters is which channel is more prominent *relative
   to its own floor*. This also handles the inverse failure, where a built-in
   mic picks up the laptop speakers: that bleed is attenuated, so its prominence
   stays below the loopback's.

Decisions are made per Whisper segment rather than per word. Word-level RMS over
a 0.2-0.5s slice is mostly noise, which is why transcripts previously flipped
speaker mid-sentence ("... Have you" / "received any post" / "feedback? Yeah.").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SPEAKER_YOU = "You"
SPEAKER_MEETING = "Meeting"

# Below this RMS a channel is treated as digital silence: no signal at all,
# rather than a very quiet one. Without this, a channel of exact zeros would
# have a zero floor and any dither would look infinitely prominent.
ABSOLUTE_SILENCE = 1e-5

# Floor estimates come from the quietest slices of a window. This has to be a
# low percentile rather than a middling one: someone talking steadily can be
# loud for 80-90% of a 30s window, and a 20th percentile would then sit inside
# their speech and inflate the floor. Pauses between words reliably leave a few
# quiet slices behind.
FLOOR_PERCENTILE = 5.0
FLOOR_SLICE_SECONDS = 0.2

# The running floor follows observations asymmetrically: quickly downwards (a
# quieter baseline is believed at once) and slowly upwards (so sustained speech
# cannot drag the floor up behind it). This is the standard way noise-floor
# trackers avoid being captured by the signal they are meant to exclude.
FLOOR_DECAY_DOWN = 0.5
FLOOR_DECAY_UP = 0.05


def _db(value: float) -> float:
    return 20.0 * np.log10(max(float(value), 1e-12))


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _noise_floor(audio: np.ndarray, sample_rate: int) -> float:
    """Low percentile of short-slice RMS — an estimate of the quiet baseline."""
    n = max(1, int(sample_rate * FLOOR_SLICE_SECONDS))
    if audio.size < n * 2:
        return _rms(audio)
    usable = audio[: (audio.size // n) * n].reshape(-1, n)
    slice_rms = np.sqrt(np.mean(np.square(usable, dtype=np.float64), axis=1))
    return float(np.percentile(slice_rms, FLOOR_PERCENTILE))


@dataclass
class Decision:
    """One attribution, with the numbers behind it for debugging and the UI."""

    speaker: str
    mic_rms: float
    meeting_rms: float
    mic_prominence_db: float
    meeting_prominence_db: float
    reason: str

    def describe(self, start: float, end: float) -> str:
        return (
            f"  [{start:.1f}-{end:.1f}s] "
            f"mic={self.mic_rms:.5f} (+{self.mic_prominence_db:.1f}dB) "
            f"meeting={self.meeting_rms:.5f} (+{self.meeting_prominence_db:.1f}dB) "
            f"-> {self.speaker} [{self.reason}]"
        )


@dataclass
class SpeakerDetector:
    """Attributes speech to one side using calibrated per-channel energy.

    ``margin_db`` is how far above its own noise floor a channel must rise to
    count as carrying speech. ``separation_db`` is how much more prominent one
    channel must be than the other before the decision flips.
    """

    margin_db: float = 8.0
    separation_db: float = 3.0
    mic_floor: float | None = None
    meeting_floor: float | None = None
    _last_speaker: str = field(default=SPEAKER_MEETING, repr=False)

    def observe(self, mic: np.ndarray, meeting: np.ndarray, sample_rate: int) -> None:
        """Update the noise-floor estimates from a freshly captured window."""
        self.mic_floor = self._update_floor(self.mic_floor, _noise_floor(mic, sample_rate))
        self.meeting_floor = self._update_floor(
            self.meeting_floor, _noise_floor(meeting, sample_rate)
        )

    @staticmethod
    def _update_floor(current: float | None, observed: float) -> float:
        if current is None:
            return observed
        rate = FLOOR_DECAY_DOWN if observed < current else FLOOR_DECAY_UP
        return (1.0 - rate) * current + rate * observed

    def _prominence_db(self, rms: float, floor: float | None) -> float:
        """How far this level sits above the channel's own quiet baseline."""
        if rms < ABSOLUTE_SILENCE:
            return 0.0
        reference = max(floor if floor is not None else 0.0, ABSOLUTE_SILENCE)
        return _db(rms) - _db(reference)

    def classify(
        self,
        mic: np.ndarray,
        meeting: np.ndarray,
        start_sec: float,
        end_sec: float,
        sample_rate: int,
    ) -> Decision:
        """Attribute the audio between ``start_sec`` and ``end_sec``."""
        start_idx = max(0, int(start_sec * sample_rate))
        end_idx = min(len(mic), max(start_idx + 1, int(end_sec * sample_rate)))

        mic_slice = mic[start_idx:end_idx]
        meeting_slice = meeting[start_idx:min(len(meeting), end_idx)]

        mic_rms = _rms(mic_slice)
        meeting_rms = _rms(meeting_slice)

        mic_prom = self._prominence_db(mic_rms, self.mic_floor)
        meeting_prom = self._prominence_db(meeting_rms, self.meeting_floor)

        mic_live = mic_rms >= ABSOLUTE_SILENCE and mic_prom >= self.margin_db
        meeting_live = meeting_rms >= ABSOLUTE_SILENCE and meeting_prom >= self.margin_db

        if mic_live and not meeting_live:
            speaker, reason = SPEAKER_YOU, "mic only"
        elif meeting_live and not mic_live:
            speaker, reason = SPEAKER_MEETING, "meeting only"
        elif mic_live and meeting_live:
            # Both carry signal — cross-talk, or bleed from one into the other.
            # The more prominent channel relative to its own floor wins.
            delta = mic_prom - meeting_prom
            if delta >= self.separation_db:
                speaker, reason = SPEAKER_YOU, "mic more prominent"
            elif -delta >= self.separation_db:
                speaker, reason = SPEAKER_MEETING, "meeting more prominent"
            else:
                speaker, reason = self._last_speaker, "ambiguous, held"
        else:
            # Neither channel is clearly active, yet something was transcribed.
            # Hold the previous speaker rather than inventing a change.
            speaker, reason = self._last_speaker, "below floor, held"

        self._last_speaker = speaker
        return Decision(
            speaker=speaker,
            mic_rms=mic_rms,
            meeting_rms=meeting_rms,
            mic_prominence_db=mic_prom,
            meeting_prominence_db=meeting_prom,
            reason=reason,
        )


def label_segments(
    segments: list[dict],
    mic: np.ndarray,
    meeting: np.ndarray,
    sample_rate: int,
    detector: SpeakerDetector,
    debug: bool = False,
) -> list[dict]:
    """Attribute each segment and merge consecutive runs by the same speaker.

    ``segments`` are Whisper segments — ``{"start", "end", "text"}`` — not
    individual words.
    """
    if not segments:
        return []

    labelled = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        decision = detector.classify(
            mic, meeting, seg["start"], seg["end"], sample_rate
        )
        if debug:
            print(decision.describe(seg["start"], seg["end"]))
        labelled.append({
            "speaker": decision.speaker,
            "text": text,
            "start": seg["start"],
            "end": seg["end"],
            "reason": decision.reason,
            # Carried through so a UI can answer "why was this labelled that?"
            # without re-running the analysis.
            "mic_db": round(decision.mic_prominence_db, 1),
            "meeting_db": round(decision.meeting_prominence_db, 1),
        })

    return merge_runs(labelled)


def merge_runs(labelled: list[dict]) -> list[dict]:
    """Collapse consecutive segments by the same speaker into one entry."""
    if not labelled:
        return []
    merged = [dict(labelled[0])]
    for seg in labelled[1:]:
        if seg["speaker"] == merged[-1]["speaker"]:
            merged[-1]["text"] = f"{merged[-1]['text']} {seg['text']}".strip()
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged
