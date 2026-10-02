#!/usr/bin/env python3
"""
AI Telephone Game (Whisper Game)
================================
Chain AI agents in a telephone/whisper game experiment.
The first agent receives a paragraph and tries to repeat it,
then the next agent receives that output, and so on.

Supports Ollama (local) and Gemini API providers,
with toggleable difficulty modes.

Usage:
    python3 whisper_game.py                          # defaults (Ollama, 5 agents, recall mode)
    python3 whisper_game.py --provider gemini         # use Gemini API
    python3 whisper_game.py --num-agents 10           # 10 agents in chain
    python3 whisper_game.py --generate-start-text     # let AI generate the starting text
    python3 whisper_game.py --mode-redact --mode-distraction --temperature 1.5  # hard mode
    python3 whisper_game.py --no-mode-recall          # verbatim copy attempt
    python3 whisper_game.py --start-text "Your text"  # custom starting text
"""

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time

import requests

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULTS = {
    # Provider settings
    "provider": "ollama",           # "ollama" or "gemini"
    "ollama_model": "llama3.2:3b",  # Ollama model name
    "ollama_url": "http://localhost:11434",  # Ollama API base URL
    "gemini_model": "gemini-3.8-flash",     # Gemini model name

    # Chain settings
    "num_agents": 5,                # Number of agents in the chain

    # Starting text
    "generate_start_text": False,   # Let the AI generate the starting paragraph
    "start_text_topic": "a historical event",  # Topic hint for generated text
    "start_text": (
        "On July 20, 1969, astronaut Neil Armstrong became the first human to walk "
        "on the Moon during the Apollo 11 mission. He was joined by Buzz Aldrin, while "
        "Michael Collins orbited above in the command module Columbia. Armstrong's famous "
        "words, 'That's one small step for man, one giant leap for mankind,' were broadcast "
        "to an estimated 600 million people watching on television around the world. The "
        "mission lasted eight days and was a defining achievement of the Space Race between "
        "the United States and the Soviet Union."
    ),

    # Difficulty modes (toggleable)
    "mode_recall": True,            # "Recall from memory" prompt
    "mode_summarize_expand": False, # Summarize then expand
    "expand_word_limit": None,      # Limit words when expanding in summarize_expand mode (int or None)
    "mode_redact": False,           # Randomly mask ~20% of words with [???]
    "mode_distraction": False,      # Surround text with distracting noise

    # Export
    "export_path": None,            # Path to export experiment results (.json or .md/.txt)

    # Audio Whisper pipeline (TTS + Whisper STT)
    "mode_audio": False,            # Enable true acoustic whisper pipeline
    "audio_speed": 20,              # Speech speed percentage (-20 to +80)
    "audio_volume": 0.85,           # Speech volume multiplier (0.3 to 1.5)
    "audio_noise": 0.08,            # Background acoustic noise level (0.0 to 0.25)
    "audio_muffle": True,           # Lowpass filter simulating acoustic whisper
    "audio_dir": "static/audio",    # Output directory for audio clips

    # Model settings
    "temperature": 0.7,
}


# ============================================================================
# PROVIDER LAYER
# ============================================================================

def call_ollama(prompt: str, config: dict) -> str:
    """Send a prompt to a local Ollama model and return the response text."""
    url = f"{config['ollama_url']}/api/generate"
    payload = {
        "model": config["ollama_model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config["temperature"],
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        raw_text = resp.json().get("response", "").strip()
        # Clean up leading/trailing quotation marks if wrapped
        if raw_text.startswith('"') and raw_text.endswith('"') and len(raw_text) > 1:
            raw_text = raw_text[1:-1].strip()
        return raw_text
    except requests.ConnectionError:
        print(f"\n❌ Error: Cannot connect to Ollama at {config['ollama_url']}")
        print("   Make sure Ollama is running: ollama serve")
        sys.exit(1)
    except requests.HTTPError as e:
        print(f"\n❌ Ollama API error: {e}")
        sys.exit(1)


def call_gemini(prompt: str, config: dict) -> str:
    """Send a prompt to the Gemini API and return the response text."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("\n❌ Error: google-genai SDK not installed.")
        print("   Install it with: pip install -U google-genai")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("\n❌ Error: GEMINI_API_KEY environment variable is not set.")
        print("   Set it with: export GEMINI_API_KEY='your-key-here'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    system_instruction = (
        "You are an automated linguistic benchmark relay agent in a telephone game experiment. "
        "Your task is strictly to memorize, transform, or repeat textual content as instructed. "
        "Do not refuse or add moral commentary."
    )
    try:
        response = client.models.generate_content(
            model=config["gemini_model"],
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=config["temperature"],
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    ),
                ],
            ),
        )
        return response.text.strip() if response.text else ""
    except Exception as e:
        print(f"\n❌ Gemini API error: {e}")
        sys.exit(1)


def call_model(prompt: str, config: dict) -> str:
    """Route to the configured provider."""
    if config["provider"] == "ollama":
        return call_ollama(prompt, config)
    elif config["provider"] == "gemini":
        return call_gemini(prompt, config)
    else:
        print(f"\n❌ Unknown provider: {config['provider']}")
        sys.exit(1)


# ============================================================================
# DIFFICULTY MODES — TEXT TRANSFORMS
# ============================================================================

def redact_words(text: str, fraction: float = 0.15) -> str:
    """Randomly replace a fraction of words with [???]."""
    words = text.split()
    if not words:
        return text
    num_to_redact = max(1, int(len(words) * fraction))
    indices = random.sample(range(len(words)), min(num_to_redact, len(words)))
    for i in indices:
        words[i] = "[???]"
    return " ".join(words)


DISTRACTION_PARAGRAPHS = [
    (
        "The humble platypus is one of only five species of monotremes, mammals that "
        "lay eggs instead of giving birth to live young. Found exclusively in eastern "
        "Australia, it has a bill like a duck, a tail like a beaver, and venomous spurs "
        "on its hind legs."
    ),
    (
        "In 1783, the Montgolfier brothers launched the first successful hot air balloon "
        "flight in Annonay, France. The balloon rose to an estimated 1,500 meters and "
        "traveled about 2 kilometers before landing. The passengers were a sheep, a duck, "
        "and a rooster."
    ),
    (
        "The world's largest known living organism is a honey fungus in Oregon's Blue "
        "Mountains, spanning approximately 2,385 acres. Known as the Humongous Fungus, "
        "it is estimated to be between 2,400 and 8,650 years old and mostly lives "
        "underground."
    ),
    (
        "The shortest war in recorded history lasted only 38 to 45 minutes. It occurred "
        "between Britain and Zanzibar on August 27, 1896, after Sultan Khalid bin Barghash "
        "refused to step down. The British bombarded the palace and the sultan fled."
    ),
    (
        "A group of flamingos is called a 'flamboyance.' These vibrant pink birds get "
        "their distinctive color from carotenoid pigments found in the algae and crustaceans "
        "they eat. Baby flamingos are born white or grey and develop their pink coloring "
        "over several years."
    ),
]


def wrap_with_distractions(text: str) -> str:
    """Surround the target text with random distracting paragraphs."""
    before = random.choice(DISTRACTION_PARAGRAPHS)
    after = random.choice(DISTRACTION_PARAGRAPHS)
    return (
        f"--- Background Information ---\n{before}\n\n"
        f"--- Important: Target Text ---\n{text}\n\n"
        f"--- Additional Context ---\n{after}"
    )


# ============================================================================
# AUDIO RELAY & TEXT CLEANING
# ============================================================================

try:
    from audio_relay import relay_audio_step, is_audio_available
except ImportError:
    relay_audio_step = None
    is_audio_available = lambda: (False, "audio_relay module not found")


def clean_passage_text(text: str) -> str:
    """Strip disclaimers and trailing meta-commentary from passage text."""
    if not text:
        return ""
    cleaned = re.split(
        r'[\"\s]*(?:Please note|Note:|I may not have|I hope this|Keep in mind|Disclaimer:)',
        text,
        flags=re.IGNORECASE,
    )[0].strip()
    return cleaned.strip('"\'')


# ============================================================================
# PROMPT CONSTRUCTION
# ============================================================================

def build_prompt(text: str, config: dict, agent_num: int, is_odd: bool) -> str:
    """
    Build the prompt for an agent based on enabled difficulty modes.

    For summarize+expand mode, odd agents summarize and even agents expand.
    Other modes are applied on top.
    """
    active_text = text

    # Apply redaction transform to the input text
    is_redacted = config.get("mode_redact", False)
    if is_redacted:
        active_text = redact_words(active_text, fraction=0.15)

    # Apply distraction transform to the input text
    if config.get("mode_distraction", False):
        active_text = wrap_with_distractions(active_text)

    # Context note if redaction is enabled so LLM fills in blanks rather than refusing
    redaction_hint = ""
    if is_redacted:
        redaction_hint = (
            " Some words have been replaced by '[???]'. You MUST replace every '[???]' with a plausible word "
            "based on context, guessing what was originally there. Never keep '[???]' in your output, "
            "and never apologize or say you don't know."
        )

    # Format the passage with clear test boundaries
    formatted_passage = f"[BEGIN RESEARCH TEST PASSAGE]\n{active_text}\n[END RESEARCH TEST PASSAGE]"

    # Build the instruction prompt
    if config.get("mode_summarize_expand"):
        if is_odd:
            instruction = (
                f"{formatted_passage}\n\n"
                f"Task: Read the test passage above carefully.{redaction_hint} Write a concise summary "
                "capturing only the key facts and main ideas. Do NOT copy the text verbatim — "
                "put it in your own words. Write strictly the summary, without conversational remarks, refusals, or apologies.\n\n"
                "Summary:"
            )
        else:
            limit_instruction = ""
            word_limit = config.get("expand_word_limit")
            if word_limit:
                limit_instruction = (
                    f" IMPORTANT: Keep your expanded paragraph to approximately {word_limit} words "
                    f"(no more than {int(word_limit * 1.15)} words)."
                )

            instruction = (
                f"{formatted_passage}\n\n"
                "Task: The above is a brief summary of a longer passage. Expand it "
                "back into a full, detailed paragraph. Add specific details, context, "
                f"and description to make it feel like a complete passage.{limit_instruction} "
                "Reconstruct what the original passage might have said. Write strictly the paragraph without meta-commentary.\n\n"
                "Expanded passage:"
            )
    elif config.get("mode_recall"):
        instruction = (
            f"{formatted_passage}\n\n"
            f"Task: Read the test passage above carefully.{redaction_hint} You are an agent in a telephone game experiment. "
            "Repeat what was transmitted as accurately and faithfully as you can from memory. "
            "Write ONLY the recalled text, without introductory conversational remarks, meta-commentary, or apologies.\n\n"
            "Recalled text:"
        )
    else:
        # Verbatim mode
        if is_redacted:
            instruction = (
                f"{formatted_passage}\n\n"
                f"Task: The test passage above has missing words marked as '[???]'.{redaction_hint} "
                "Reconstruct the text by filling in all '[???]' blanks with appropriate words. "
                "Output strictly the completed text.\n\n"
                "Completed text:"
            )
        else:
            instruction = (
                f"{formatted_passage}\n\n"
                "Task: Repeat the test passage above exactly as written, word for word. "
                "Do not add any commentary, explanation, or modifications.\n\n"
                "Exact repetition:"
            )

    return instruction


def build_generation_prompt(topic: str) -> str:
    """Build a prompt to generate the starting text."""
    return (
        f"Write a single paragraph (50-80 words) about {topic}. "
        "Include specific facts, names, dates, and numbers. "
        "Write only the paragraph, no title or introduction."
    )


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

BOX_WIDTH = 60

def print_header(config: dict):
    """Print the experiment header box."""
    provider_info = config["provider"]
    if config["provider"] == "ollama":
        provider_info += f" ({config['ollama_model']})"
    else:
        provider_info += f" ({config['gemini_model']})"

    modes = []
    if config["mode_recall"]:
        modes.append("recall")
    if config["mode_summarize_expand"]:
        modes.append("summarize+expand")
    if config["mode_redact"]:
        modes.append("redact")
    if config["mode_distraction"]:
        modes.append("distraction")
    if not modes:
        modes.append("verbatim")

    mode_str = ", ".join(modes)

    print()
    print("╔" + "═" * BOX_WIDTH + "╗")
    print("║" + " AI TELEPHONE GAME".center(BOX_WIDTH) + "║")
    print("║" + "".center(BOX_WIDTH) + "║")
    print("║" + f"  Provider: {provider_info}".ljust(BOX_WIDTH) + "║")
    print("║" + f"  Agents: {config['num_agents']}  |  Temp: {config['temperature']}".ljust(BOX_WIDTH) + "║")
    print("║" + f"  Modes: {mode_str}".ljust(BOX_WIDTH) + "║")
    print("╚" + "═" * BOX_WIDTH + "╝")
    print()


def print_round(label: str, text: str):
    """Print a single round's output."""
    print(f"── {label} " + "─" * max(1, BOX_WIDTH - len(label) - 4))
    word_count = len(text.split())
    # Wrap long lines for readability
    wrapped = textwrap.fill(text, width=BOX_WIDTH + 10)
    print(wrapped)
    print(f"({word_count} words)")
    print()


def print_summary(rounds: list[tuple[str, str]]):
    """Print the summary table at the end."""
    print("═" * (BOX_WIDTH + 2))
    print(" SUMMARY".center(BOX_WIDTH + 2))
    print("═" * (BOX_WIDTH + 2))
    print(f"  {'Round':<10} {'Words':>8} {'Change':>8}")
    print(f"  {'─' * 10} {'─' * 8} {'─' * 8}")

    prev_count = None
    for label, text in rounds:
        word_count = len(text.split())
        if prev_count is None:
            change = "  -"
        else:
            diff = word_count - prev_count
            change = f"{diff:+d}" if diff != 0 else "  0"
        print(f"  {label:<10} {word_count:>8} {change:>8}")
        prev_count = word_count

    print()


def export_results(filepath: str, config: dict, rounds: list[tuple[str, str]]) -> None:
    """Export experiment results to JSON or Markdown/Text."""
    ext = os.path.splitext(filepath)[1].lower()
    
    # Calculate word counts and metadata
    round_data = []
    prev_count = None
    for label, text in rounds:
        w_count = len(text.split())
        diff = (w_count - prev_count) if prev_count is not None else 0
        round_data.append({
            "label": label,
            "text": text,
            "word_count": w_count,
            "word_change": diff
        })
        prev_count = w_count

    if ext == ".json":
        data = {
            "config": {
                k: v for k, v in config.items() if k != "start_text" or not config.get("generate_start_text")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "rounds": round_data,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"📁 Results exported to JSON: {filepath}")
    else:
        # Default to Markdown / Text format
        lines = [
            "# AI Telephone Game — Experiment Results",
            "",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Provider:** {config['provider']} ({config.get('ollama_model') if config['provider'] == 'ollama' else config.get('gemini_model')})",
            f"**Agents:** {config['num_agents']} | **Temperature:** {config['temperature']}",
            "",
            "## Rounds",
            ""
        ]
        for r in round_data:
            lines.append(f"### {r['label']} ({r['word_count']} words)")
            lines.append("")
            lines.append(r["text"])
            lines.append("")
        
        lines.append("## Summary")
        lines.append("")
        lines.append("| Round | Words | Change |")
        lines.append("|---|---|---|")
        for r in round_data:
            change_str = "-" if r["label"] == "Original" else f"{r['word_change']:+d}"
            lines.append(f"| {r['label']} | {r['word_count']} | {change_str} |")
        lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"📁 Results exported to Markdown: {filepath}")


# ============================================================================
# MAIN CHAIN RUNNER
# ============================================================================

def run_telephone_game(config: dict):
    """Run the full telephone game experiment."""
    print_header(config)

    # Step 1: Get the starting text
    if config["generate_start_text"]:
        print("🎲 Generating starting text...", flush=True)
        prompt = build_generation_prompt(config["start_text_topic"])
        start_text = call_model(prompt, config)
        if not start_text:
            print("❌ Failed to generate starting text.")
            sys.exit(1)
    else:
        start_text = config["start_text"]

    # If expand_word_limit is "original", resolve to actual word count
    if config.get("expand_word_limit") == "original":
        config["expand_word_limit"] = len(start_text.split())

    # Track all rounds for the summary
    rounds: list[tuple[str, str]] = [("Original", start_text)]
    print_round("ORIGINAL TEXT", start_text)

    # Step 2: Run the chain
    current_text = start_text
    for i in range(1, config["num_agents"] + 1):
        is_odd = (i % 2 == 1)
        prompt = build_prompt(current_text, config, i, is_odd)

        label = f"AGENT {i}"
        if config.get("mode_summarize_expand"):
            label += " (summarize)" if is_odd else " (expand)"

        if config.get("mode_audio"):
            # Acoustic Audio Whisper Mode (TTS + Whisper STT)
            audio_dir = config.get("audio_dir", "static/audio")
            os.makedirs(audio_dir, exist_ok=True)
            audio_path = os.path.join(audio_dir, f"round_{i}.mp3")

            print(f"🎙️  Agent {i}/{config['num_agents']} whispering audio...", end="", flush=True)
            start_time = time.time()
            transcribed, _ = relay_audio_step(
                current_text,
                agent_num=i,
                output_audio_path=audio_path,
                speed_pct=int(config.get("audio_speed", 20)),
                volume_factor=float(config.get("audio_volume", 0.85)),
                noise_level=float(config.get("audio_noise", 0.08)),
                muffle=bool(config.get("audio_muffle", True)),
            )
            elapsed = time.time() - start_time
            print(f" heard ({elapsed:.1f}s)")
            current_text = transcribed
        else:
            # Standard LLM agent execution
            print(f"🔗 Agent {i}/{config['num_agents']} thinking...", end="", flush=True)
            start_time = time.time()
            raw_response = call_model(prompt, config)
            elapsed = time.time() - start_time
            print(f" done ({elapsed:.1f}s)")

            if not raw_response:
                print(f"⚠️  Agent {i} returned empty response, stopping chain.")
                break

            current_text = clean_passage_text(raw_response)

        rounds.append((f"Agent {i}", current_text))
        print_round(label, current_text)

    # Step 3: Print summary
    print_summary(rounds)

    # Step 4: Export if requested
    if config.get("export_path"):
        export_results(config["export_path"], config, rounds)

    print("✅ Telephone game complete!\n")


# ============================================================================
# CLI ARGUMENT PARSING
# ============================================================================

def parse_args() -> dict:
    """Parse command-line arguments and merge with defaults."""
    parser = argparse.ArgumentParser(
        description="AI Telephone Game — chain AI agents in a whisper game experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s                                    # defaults (Ollama, 5 agents, recall mode)
              %(prog)s --mode-audio                       # acoustic audio whisper game (TTS + Whisper STT)
              %(prog)s --mode-audio --audio-noise 0.12    # noisy whisper audio
              %(prog)s --provider gemini                  # use Gemini API
              %(prog)s --num-agents 10                    # 10 agents in chain
              %(prog)s --temperature 1.5                  # high temperature for more chaos
        """),
    )

    # Provider settings
    provider_group = parser.add_argument_group("provider settings")
    provider_group.add_argument(
        "--provider", choices=["ollama", "gemini"], default=DEFAULTS["provider"],
        help=f"Model provider (default: {DEFAULTS['provider']})",
    )
    provider_group.add_argument(
        "--ollama-model", default=DEFAULTS["ollama_model"],
        help=f"Ollama model name (default: {DEFAULTS['ollama_model']})",
    )
    provider_group.add_argument(
        "--ollama-url", default=DEFAULTS["ollama_url"],
        help=f"Ollama API base URL (default: {DEFAULTS['ollama_url']})",
    )
    provider_group.add_argument(
        "--gemini-model", default=DEFAULTS["gemini_model"],
        help=f"Gemini model name (default: {DEFAULTS['gemini_model']})",
    )

    # Chain settings
    chain_group = parser.add_argument_group("chain settings")
    chain_group.add_argument(
        "--num-agents", type=int, default=DEFAULTS["num_agents"],
        help=f"Number of agents in the chain (default: {DEFAULTS['num_agents']})",
    )
    chain_group.add_argument(
        "--temperature", type=float, default=DEFAULTS["temperature"],
        help=f"Model temperature — higher = more creative/lossy (default: {DEFAULTS['temperature']})",
    )

    # Starting text
    text_group = parser.add_argument_group("starting text")
    text_group.add_argument(
        "--start-text", default=DEFAULTS["start_text"],
        help="Custom starting paragraph text",
    )
    text_group.add_argument(
        "--generate-start-text", action="store_true", default=DEFAULTS["generate_start_text"],
        help="Let the AI generate the starting paragraph",
    )
    text_group.add_argument(
        "--start-text-topic", default=DEFAULTS["start_text_topic"],
        help=f"Topic for generated starting text (default: '{DEFAULTS['start_text_topic']}')",
    )

    # Difficulty modes (each is a --mode-X / --no-mode-X toggle)
    mode_group = parser.add_argument_group("difficulty modes (toggleable)")
    mode_group.add_argument(
        "--mode-recall", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_recall"],
        help="'Recall from memory' prompt (default: on)",
    )
    mode_group.add_argument(
        "--mode-summarize-expand", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_summarize_expand"],
        help="Alternate summarize/expand between agents (default: off)",
    )
    mode_group.add_argument(
        "--expand-word-limit", default=None,
        help="Word limit when expanding in summarize+expand mode: an integer or 'original' (default: unlimited)",
    )
    mode_group.add_argument(
        "--mode-redact", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_redact"],
        help="Randomly mask ~20%% of words with [???] (default: off)",
    )
    mode_group.add_argument(
        "--mode-distraction", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_distraction"],
        help="Surround target text with distracting paragraphs (default: off)",
    )

    # Audio Whisper pipeline settings
    audio_group = parser.add_argument_group("acoustic audio whisper settings (TTS + Whisper STT)")
    audio_group.add_argument(
        "--mode-audio", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_audio"],
        help="Enable acoustic audio whisper pipeline with TTS + Whisper STT (default: off)",
    )
    audio_group.add_argument(
        "--audio-speed", type=int, default=DEFAULTS["audio_speed"],
        help="Speech speed adjustment percentage (-20 to +80, default: 20)",
    )
    audio_group.add_argument(
        "--audio-volume", type=float, default=DEFAULTS["audio_volume"],
        help="Speech volume/power multiplier (0.3 to 1.5, default: 0.85)",
    )
    audio_group.add_argument(
        "--audio-noise", type=float, default=DEFAULTS["audio_noise"],
        help="Background acoustic pink noise level (0.0 to 0.25, default: 0.08)",
    )
    audio_group.add_argument(
        "--audio-muffle", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["audio_muffle"],
        help="Apply acoustic lowpass muffle filter (default: on)",
    )

    # Export options
    export_group = parser.add_argument_group("export options")
    export_group.add_argument(
        "--export", dest="export_path", default=None,
        help="File path to export results (.json or .md/.txt)",
    )

    args = parser.parse_args()

    # Parse expand_word_limit (could be "original", an int, or None)
    expand_limit = args.expand_word_limit
    if expand_limit and expand_limit != "original":
        try:
            expand_limit = int(expand_limit)
        except ValueError:
            print(f"⚠️  Warning: Invalid --expand-word-limit '{expand_limit}', ignoring.")
            expand_limit = None

    # Convert to config dict (using underscores)
    config = {
        "provider": args.provider,
        "ollama_model": args.ollama_model,
        "ollama_url": args.ollama_url,
        "gemini_model": args.gemini_model,
        "num_agents": args.num_agents,
        "temperature": args.temperature,
        "start_text": args.start_text,
        "generate_start_text": args.generate_start_text,
        "start_text_topic": args.start_text_topic,
        "mode_recall": args.mode_recall,
        "mode_summarize_expand": args.mode_summarize_expand,
        "expand_word_limit": expand_limit,
        "mode_redact": args.mode_redact,
        "mode_distraction": args.mode_distraction,
        "mode_audio": args.mode_audio,
        "audio_speed": args.audio_speed,
        "audio_volume": args.audio_volume,
        "audio_noise": args.audio_noise,
        "audio_muffle": args.audio_muffle,
        "export_path": args.export_path,
    }

    return config


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = parse_args()
    run_telephone_game(config)
