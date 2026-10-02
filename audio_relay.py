"""
audio_relay.py — Acoustic Audio Pipeline for the AI Telephone Game.
Translates text to audio using neural TTS, applies acoustic noise and whispering filters via ffmpeg,
and transcribes what was heard using OpenAI Whisper.
"""

import asyncio
import os
import shutil
import subprocess

_WHISPER_MODEL = None

VOICE_LIST = [
    "en-US-ChristopherNeural",
    "en-US-JennyNeural",
    "en-GB-RyanNeural",
    "en-AU-WilliamNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-CA-LiamNeural",
    "en-US-AriaNeural",
]


def is_audio_available() -> tuple[bool, str]:
    """Check if all required audio dependencies are available."""
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg is not installed or not in PATH."
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        return False, "edge-tts is not installed. Install with: pip install edge-tts"
    try:
        import whisper  # noqa: F401
    except ImportError:
        return False, "whisper is not installed. Install with: pip install openai-whisper"
    return True, "Audio dependencies are ready."


def get_whisper_model(model_name: str = "tiny"):
    """Load and cache the Whisper model."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        import whisper
        _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL


async def _synthesize_edge_tts(text: str, voice: str, speed_pct: int, output_path: str):
    """Internal helper to call edge_tts asynchronously."""
    import edge_tts

    rate_str = f"{speed_pct:+d}%" if speed_pct != 0 else "+0%"
    communicate = edge_tts.Communicate(text, voice, rate=rate_str)
    await communicate.save(output_path)


def relay_audio_step(
    text: str,
    agent_num: int,
    output_audio_path: str,
    speed_pct: int = 20,
    volume_factor: float = 0.85,
    noise_level: float = 0.08,
    muffle: bool = True,
    whisper_model_name: str = "tiny",
) -> tuple[str, str]:
    """
    Execute one acoustic whisper relay step:
    1. Text -> Spoken Audio via edge-tts (with voice selection and speed).
    2. Audio -> ffmpeg acoustic degradation (volume, lowpass filter, noise).
    3. Degraded Audio -> Speech Recognition via Whisper.

    Returns:
        (transcribed_text, output_audio_path)
    """
    if not text.strip():
        return "", output_audio_path

    voice = VOICE_LIST[(agent_num - 1) % len(VOICE_LIST)]
    raw_mp3 = output_audio_path + ".raw.mp3"

    try:
        # Step 1: Synthesize with edge-tts
        asyncio.run(_synthesize_edge_tts(text, voice, speed_pct, raw_mp3))

        # Step 2: Apply acoustic filters using ffmpeg
        filters = []
        if muffle:
            filters.append("lowpass=f=1400")
        if volume_factor != 1.0:
            filters.append(f"volume={max(0.1, min(2.0, volume_factor)):.2f}")

        filter_str = ",".join(filters) if filters else "anull"

        if noise_level > 0:
            # Mix pink room/whisper noise
            noise_val = max(0.01, min(0.35, noise_level))
            cmd = [
                "ffmpeg", "-y", "-i", raw_mp3,
                "-filter_complex",
                f"anoisesrc=d=60:c=pink:r=24000:a={noise_val:.3f} [noise]; [0:a]{filter_str} [clean]; [clean][noise] amix=inputs=2:duration=first",
                "-ac", "1",
                output_audio_path,
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", raw_mp3,
                "-af", filter_str,
                "-ac", "1",
                output_audio_path,
            ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Step 3: Transcribe with Whisper
        model = get_whisper_model(whisper_model_name)
        result = model.transcribe(output_audio_path, fp16=False)
        transcribed_text = result.get("text", "").strip()

        return transcribed_text, output_audio_path

    finally:
        if os.path.exists(raw_mp3):
            try:
                os.remove(raw_mp3)
            except OSError:
                pass
