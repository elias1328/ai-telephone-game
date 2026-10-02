#!/usr/bin/env python3
"""
AI Telephone Game — Web UI
===========================
Flask server that provides a web interface for the telephone game.
Reuses the model-calling and text-transform logic from whisper_game.py.

Usage:
    python3 whisper_game_web.py
    python3 whisper_game_web.py --port 8080
"""

import json
import os
import sys
import time
import webbrowser
import argparse
from threading import Timer

import requests
from flask import Flask, render_template, request, Response, jsonify

# Import the existing game logic
from whisper_game import (
    call_model,
    redact_words,
    wrap_with_distractions,
    build_prompt,
    build_generation_prompt,
)

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models", methods=["GET"])
def get_models():
    """Return available Ollama models."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            return jsonify(resp.json())
        return jsonify({"models": []})
    except Exception as e:
        return jsonify({"models": [], "error": str(e)})


@app.route("/api/run", methods=["POST"])
def run_game():
    """Run the telephone game and stream results via SSE."""
    config = request.json

    def generate():
        def emit(event, data):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            provider = config.get("provider", "ollama")
            num_agents = config.get("num_agents", 5)

            # Expand word limit parsing
            expand_limit = config.get("expand_word_limit")
            if expand_limit and expand_limit != "original":
                try:
                    expand_limit = int(expand_limit)
                except (ValueError, TypeError):
                    expand_limit = None

            # Build the config dict expected by whisper_game.call_model
            game_config = {
                "provider": provider,
                "ollama_model": config.get("ollama_model", "llama3.2"),
                "ollama_url": config.get("ollama_url", "http://localhost:11434"),
                "gemini_model": config.get("gemini_model", "gemini-3.8-flash"),
                "temperature": float(config.get("temperature", 0.7)),
                "mode_recall": config.get("mode_recall", True),
                "mode_summarize_expand": config.get("mode_summarize_expand", False),
                "expand_word_limit": expand_limit,
                "mode_redact": config.get("mode_redact", False),
                "mode_distraction": config.get("mode_distraction", False),
            }

            model_name = (
                game_config["ollama_model"]
                if provider == "ollama"
                else game_config["gemini_model"]
            )

            # Emit header
            yield emit(
                "header",
                {
                    "provider": provider,
                    "model": model_name,
                    "num_agents": num_agents,
                },
            )

            # Step 1: Get the starting text
            current_text = config.get("start_text", "")

            if config.get("generate_start_text"):
                topic = config.get("start_text_topic", "a historical event")
                gen_prompt = build_generation_prompt(topic)
                yield emit("thinking", {"agent_num": 0, "total": num_agents})
                try:
                    current_text = call_model(gen_prompt, game_config)
                except Exception as e:
                    yield emit(
                        "error",
                        {"message": f"Failed to generate start text: {str(e)}"},
                    )
                    return

            if not current_text.strip():
                yield emit("error", {"message": "No starting text provided."})
                return

            # Emit original text
            original_words = len(current_text.split())
            if game_config.get("expand_word_limit") == "original":
                game_config["expand_word_limit"] = original_words

            yield emit(
                "original", {"text": current_text, "word_count": original_words}
            )

            summary_data = [{"agent": 0, "word_count": original_words, "time": 0}]

            # Step 2: Run the chain
            for i in range(1, num_agents + 1):
                yield emit("thinking", {"agent_num": i, "total": num_agents})

                start_time = time.time()
                is_odd = i % 2 == 1

                # Build the prompt using the existing logic (handles all modes)
                prompt = build_prompt(current_text, game_config, i, is_odd)

                try:
                    current_text = call_model(prompt, game_config)
                except Exception as e:
                    yield emit("error", {"message": f"Agent {i} failed: {str(e)}"})
                    return

                elapsed = time.time() - start_time
                word_count = len(current_text.split())

                summary_data.append(
                    {"agent": i, "word_count": word_count, "time": elapsed}
                )

                yield emit(
                    "result",
                    {
                        "agent_num": i,
                        "text": current_text,
                        "word_count": word_count,
                        "time_taken": elapsed,
                    },
                )

            # Step 3: Summary
            yield emit("summary", {"rounds": summary_data})
            yield emit("done", {})

        except Exception as e:
            yield emit("error", {"message": str(e)})

    return Response(generate(), mimetype="text/event-stream")


def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Telephone Game — Web UI")
    parser.add_argument("--port", type=int, default=5000, help="Port to run on (default: 5000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    if not args.no_browser:
        Timer(1.5, open_browser, args=[args.port]).start()

    print(f"\n🎮 AI Telephone Game Web UI")
    print(f"   http://127.0.0.1:{args.port}")
    print(f"   Press Ctrl+C to stop\n")
    app.run(port=args.port, debug=False)
