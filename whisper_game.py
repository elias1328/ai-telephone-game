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
    "ollama_model": "llama3.2",     # Ollama model name
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
    "mode_redact": False,           # Randomly mask ~20% of words with [???]
    "mode_distraction": False,      # Surround text with distracting noise

    # Model settings
    "temperature": 0.7,
}


# ============================================================================
# PROVIDER LAYER
# ============================================================================

def call_ollama(prompt: str, config: dict) -> str:
    """Send a prompt to a local Ollama model and return the response text."""
    url = f"{config['ollama_url']}/api/chat"
    payload = {
        "model": config["ollama_model"],
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": config["temperature"],
        },
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
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
    try:
        response = client.models.generate_content(
            model=config["gemini_model"],
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=config["temperature"],
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

def redact_words(text: str, fraction: float = 0.20) -> str:
    """Randomly replace a fraction of words with [???]."""
    words = text.split()
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
    if config["mode_redact"]:
        active_text = redact_words(active_text)

    # Apply distraction transform to the input text
    if config["mode_distraction"]:
        active_text = wrap_with_distractions(active_text)

    # Build the instruction prompt
    if config["mode_summarize_expand"]:
        if is_odd:
            instruction = (
                "Read the following text carefully, then write a concise summary "
                "capturing only the key facts and main ideas. Do NOT copy the text "
                "verbatim — put it in your own words.\n\n"
                f"{active_text}\n\n"
                "Your summary:"
            )
        else:
            instruction = (
                "The following is a brief summary of a longer passage. Expand it "
                "back into a full, detailed paragraph. Add specific details, context, "
                "and description to make it feel like a complete passage. Do your best "
                "to reconstruct what the original might have said.\n\n"
                f"{active_text}\n\n"
                "Your expanded paragraph:"
            )
    elif config["mode_recall"]:
        instruction = (
            "You briefly saw the following text a while ago and now need to recall it "
            "from memory. Try to reproduce the text as accurately as you can, but "
            "don't worry if you can't remember every detail perfectly — just do your "
            "best to reconstruct it from what you remember. Write ONLY the recalled "
            "text, nothing else.\n\n"
            f"The text you saw:\n{active_text}\n\n"
            "Your recall of the text:"
        )
    else:
        # Verbatim mode — no difficulty modifiers on the prompt itself
        instruction = (
            "Repeat the following text exactly as written, word for word. "
            "Do not add any commentary, explanation, or modifications.\n\n"
            f"{active_text}\n\n"
            "Your exact repetition:"
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

    # Track all rounds for the summary
    rounds: list[tuple[str, str]] = [("Original", start_text)]
    print_round("ORIGINAL TEXT", start_text)

    # Step 2: Run the chain
    current_text = start_text
    for i in range(1, config["num_agents"] + 1):
        is_odd = (i % 2 == 1)
        prompt = build_prompt(current_text, config, i, is_odd)

        label = f"AGENT {i}"
        if config["mode_summarize_expand"]:
            label += " (summarize)" if is_odd else " (expand)"

        print(f"🔗 Agent {i}/{config['num_agents']} thinking...", end="", flush=True)
        start_time = time.time()
        response = call_model(prompt, config)
        elapsed = time.time() - start_time
        print(f" done ({elapsed:.1f}s)")

        if not response:
            print(f"⚠️  Agent {i} returned empty response, stopping chain.")
            break

        current_text = response
        rounds.append((f"Agent {i}", current_text))
        print_round(label, current_text)

    # Step 3: Print summary
    print_summary(rounds)

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
              %(prog)s --provider gemini                  # use Gemini API
              %(prog)s --num-agents 10                    # 10 agents in chain
              %(prog)s --generate-start-text              # let AI generate starting text
              %(prog)s --mode-redact --mode-distraction   # enable extra difficulty
              %(prog)s --no-mode-recall                   # disable recall mode (verbatim)
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
        help="'Recall from memory' prompt — forces paraphrasing (default: on)",
    )
    mode_group.add_argument(
        "--mode-summarize-expand", action=argparse.BooleanOptionalAction,
        default=DEFAULTS["mode_summarize_expand"],
        help="Alternate summarize/expand between agents (default: off)",
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

    args = parser.parse_args()

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
        "mode_redact": args.mode_redact,
        "mode_distraction": args.mode_distraction,
    }

    return config


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    config = parse_args()
    run_telephone_game(config)
