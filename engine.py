"""
engine.py — Dataset loading, model management, and summarization logic.
"""

import json
import os
import re
import random
import traceback
import torch
from transformers import BartForConditionalGeneration, BartTokenizer


# ── Style Presets ──────────────────────────────────────────────────────────────
STYLE_PRESETS = {
    "Concise": {
        "length_penalty": 0.6,
        "no_repeat_ngram_size": 3,
        "num_beams": 6,
        "word_ratio": 0.3,
    },
    "Balanced": {
        "length_penalty": 1.0,
        "no_repeat_ngram_size": 3,
        "num_beams": 5,
        "word_ratio": 0.5,
    },
    "Detailed": {
        "length_penalty": 2.0,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "word_ratio": 0.75,
    },
}


# ── Dataset ────────────────────────────────────────────────────────────────────
def load_squad_contexts(path: str, min_words: int = 80):
    """Load paragraph contexts from SQuAD v2.0 JSON."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contexts = []
    for article in data["data"]:
        title = article["title"].replace("_", " ")
        for para in article["paragraphs"]:
            ctx = para["context"].strip()
            if len(ctx.split()) >= min_words:
                contexts.append({"title": title, "context": ctx})
    return contexts


def build_title_index(contexts):
    """Build title → list-of-contexts mapping."""
    index: dict[str, list[str]] = {}
    for item in contexts:
        index.setdefault(item["title"], []).append(item["context"])
    return index


# ── Model ──────────────────────────────────────────────────────────────────────
class Summarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        print(f"Loading {model_name}…")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}.")

    @staticmethod
    def _words_to_tokens(word_count: int) -> int:
        return max(10, int(word_count * 1.3))

    @staticmethod
    def _trim_to_last_sentence(text: str) -> str:
        text = text.strip()
        if re.search(r'[.!?]\s*$', text):
            return text
        match = re.match(r'(.*[.!?])', text, flags=re.DOTALL)
        if match and len(match.group(1).split()) >= 3:
            return match.group(1).strip()
        return text

    def summarize(self, text: str, desired_words: int, style: str):
        """Generate an abstractive summary with full error handling."""
        try:
            text = (text or "").strip()
            if not text:
                return "⚠️ Please enter or load some text first.", "—", "—"

            orig_words = len(text.split())
            if orig_words < 10:
                return "⚠️ Input too short — need at least 10 words.", "—", "—"

            preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Balanced"])

            # Guard: cap target to 90% of input
            effective_words = int(desired_words)
            capped = False
            if effective_words >= orig_words:
                effective_words = max(10, int(orig_words * 0.9))
                capped = True

            target_tokens = self._words_to_tokens(effective_words)
            min_tokens = max(10, int(target_tokens * 0.85))
            max_tokens = target_tokens + 30

            # Hard cap: never exceed input token count
            input_token_count = len(
                self.tokenizer.encode(text, max_length=1024, truncation=True)
            )
            max_tokens = min(max_tokens, input_token_count)
            if min_tokens >= max_tokens:
                min_tokens = max(5, max_tokens - 10)

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_tokens,
                    min_length=min_tokens,
                    length_penalty=preset["length_penalty"],
                    num_beams=preset["num_beams"],
                    no_repeat_ngram_size=preset["no_repeat_ngram_size"],
                    encoder_no_repeat_ngram_size=8,
                    repetition_penalty=1.5,
                    do_sample=False,
                    early_stopping=True,
                )

            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summary = self._trim_to_last_sentence(summary)

            summ_words = len(summary.split())
            compression = (
                round((1 - summ_words / orig_words) * 100, 1)
                if orig_words > 0
                else 0.0
            )

            cap_note = "  ⚠ CAPPED" if capped else ""
            wc = (
                f"Original: {orig_words}w → Summary: {summ_words}w | "
                f"Target: ~{effective_words}w ({style}){cap_note}"
            )
            cr = f"{compression}% compressed • {100 - compression:.1f}% retained"
            return summary, wc, cr

        except Exception as e:
            traceback.print_exc()
            return f"❌ Error: {str(e)}", "—", "—"

    def summarize_stream(self, text: str, desired_words: int, style: str):
        """Generator: yields (partial_text, wc, cr) word-by-word for typewriter effect.
        BART beam search generates the full output first; we then stream it visually.
        wc and cr are only populated on the final yield.
        """
        import time

        # Run full generation
        summary, wc, cr = self.summarize(text, desired_words, style)

        # Error / warning — emit immediately
        if summary.startswith("⚠️") or summary.startswith("❌"):
            yield summary, wc, cr
            return

        # Stream word-by-word at ~22 words/sec
        words = summary.split()
        for i, word in enumerate(words):
            partial = " ".join(words[: i + 1])
            is_last = (i == len(words) - 1)
            yield partial, (wc if is_last else ""), (cr if is_last else "")
            time.sleep(0.045)


