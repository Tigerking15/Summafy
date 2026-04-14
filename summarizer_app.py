"""
SUMNIGA — Neural Text Summarizer
Main entry point. Loads data, initializes model, builds UI, and launches.

Run:
    python summarizer_app.py
"""

import os
from engine import Summarizer, load_squad_contexts, build_title_index
from ui import build_ui

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SQUAD_PATH = os.path.join(BASE_DIR, "train-v2.0.json")
CSS_PATH = os.path.join(BASE_DIR, "styles.css")

# ── Load Dataset ───────────────────────────────────────────────────────────────
print("Loading SQuAD v2.0 dataset…")
contexts = load_squad_contexts(SQUAD_PATH)
if not contexts:
    print("⚠️  Warning: No usable contexts found or train-v2.0.json missing.")
    contexts = [{"title": "Example", "context": "Please provide your own text here."}]
print(f"Loaded {len(contexts):,} usable contexts.")

title_index = build_title_index(contexts)
all_titles = sorted(title_index.keys())

# ── Load Model ─────────────────────────────────────────────────────────────────
summarizer = Summarizer(model_name="facebook/bart-large-cnn")

# ── Load CSS ───────────────────────────────────────────────────────────────────
with open(CSS_PATH, "r", encoding="utf-8") as f:
    css = f.read()

# ── Build & Launch ─────────────────────────────────────────────────────────────
demo = build_ui(summarizer, contexts, all_titles, title_index)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=css,
    )