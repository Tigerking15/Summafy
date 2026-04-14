"""
ui.py — Gradio UI layout and event wiring for Summify.
Features: Streaming typewriter output, history sidebar (last 5 summaries).
"""

import random
import gradio as gr
from engine import STYLE_PRESETS

MAX_HISTORY = 5


def _live_word_count(text: str):
    wc = len(text.split()) if text and text.strip() else 0
    return f"📝 {wc} words" if wc > 0 else ""


def update_slider_from_style(style: str, text: str):
    preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Balanced"])
    text = (text or "").strip()
    orig_words = len(text.split()) if text else 0
    if orig_words < 10:
        defaults = {"Concise": 40, "Balanced": 80, "Detailed": 150}
        return gr.update(value=defaults.get(style, 80))
    target = max(20, int(orig_words * preset["word_ratio"]))
    target = round(target / 10) * 10
    return gr.update(value=max(20, min(target, 400)))


def _build_history_html(history: list) -> str:
    """Render history list as styled HTML cards."""
    if not history:
        return """
        <div class="hist-empty">
            <span>No summaries yet.</span><br>
            <small>Your last 5 summaries will appear here.</small>
        </div>
        """
    cards = ""
    for i, item in enumerate(reversed(history)):
        label_color = {"Concise": "#FFB86C", "Balanced": "#B8E3FF", "Detailed": "#C1F5B0"}.get(
            item["style"], "#D8BFFF"
        )
        cards += f"""
        <div class="hist-card" onclick="document.dispatchEvent(new CustomEvent('load_history', {{detail: {i}}}))">
            <div class="hist-card-header" style="background:{label_color}">
                <span class="hist-num">#{len(history) - i}</span>
                <span class="hist-style">{item['style']}</span>
                <span class="hist-wc">{item['words']}w</span>
            </div>
            <div class="hist-card-body">{item['preview']}</div>
        </div>
        """
    return f'<div class="hist-list">{cards}</div>'


def build_ui(summarizer, contexts, all_titles, title_index):
    """Build and return the Gradio Blocks app."""

    # ── Inner helpers ──────────────────────────────────────────────────────────
    def _compute_slider(text: str, style: str) -> int:
        """Calculate slider value for a given text and style."""
        preset = STYLE_PRESETS.get(style, STYLE_PRESETS["Balanced"])
        orig_words = len(text.split()) if text and text.strip() else 0
        if orig_words < 10:
            return {"Concise": 40, "Balanced": 80, "Detailed": 150}.get(style, 80)
        target = max(20, int(orig_words * preset["word_ratio"]))
        return max(20, min(round(target / 10) * 10, 400))

    def _random_sample(style):
        item = random.choice(contexts)
        text = item["context"]
        return text, f"📚 {item['title']}", _compute_slider(text, style)

    def _sample_by_title(title, style):
        if title and title in title_index:
            ctx = random.choice(title_index[title])
            return ctx, f"📚 {title}", _compute_slider(ctx, style)
        return "", "", gr.update()

    def _stream_and_update_history(text, words, style, history):
        """Generator that streams the summary and updates history on completion."""
        final_summary = ""
        final_wc = ""
        final_cr = ""

        for partial, wc, cr in summarizer.summarize_stream(text, words, style):
            final_summary = partial
            final_wc = wc
            final_cr = cr
            # yield: output_box, wc_box, cr_box, history_state, history_html
            yield partial, wc, cr, history, _build_history_html(history)

        # On completion, add to history (skip errors)
        if final_summary and not final_summary.startswith(("⚠️", "❌")):
            entry = {
                "style": style,
                "words": len(final_summary.split()),
                "preview": (final_summary[:120] + "…") if len(final_summary) > 120 else final_summary,
                "full": final_summary,
            }
            new_history = (history + [entry])[-MAX_HISTORY:]
        else:
            new_history = history

        yield final_summary, final_wc, final_cr, new_history, _build_history_html(new_history)

    # ── Gradio layout ──────────────────────────────────────────────────────────
    with gr.Blocks() as demo:

        # Persistent state
        history_state = gr.State([])

        # ── Header ──
        gr.HTML(f"""
        <div class="nb-header">
            <div class="nb-header-deco-circle"></div>
            <div class="nb-header-deco-diamond"></div>
            <h1 class="nb-title">SUMM<span class="nb-title-highlight">IFY</span></h1>
            <p class="nb-subtitle">Abstractive neural summarization powered by BART-Large-CNN</p>
            <div class="nb-badge">⚡ {len(contexts):,} contexts • {len(all_titles)} articles • SQuAD v2.0</div>
        </div>
        """)

        # ── Controls Strip ──
        with gr.Column(elem_classes=["nb-controls"]):
            with gr.Row():
                word_slider = gr.Slider(
                    minimum=20, maximum=400, value=80, step=10,
                    label="Target Word Count",
                    info="Approximate number of words in the summary",
                    scale=3,
                )
                style_radio = gr.Radio(
                    choices=list(STYLE_PRESETS.keys()),
                    value="Balanced",
                    label="Summary Style",
                    elem_classes=["nb-radio"],
                    scale=2,
                )
                gen_btn = gr.Button("SUMMARIZE →", variant="primary", scale=0, min_width=180)

        # ── Dataset Browser ──
        with gr.Column(elem_classes=["nb-source"]):
            with gr.Row():
                title_dd = gr.Dropdown(
                    choices=all_titles,
                    label=f"Browse Articles ({len(all_titles)})",
                    value=None,
                    allow_custom_value=False,
                    scale=4,
                )
                rand_btn = gr.Button("⚡ RANDOM", variant="secondary", scale=0, min_width=120)
                custom_btn = gr.Button("✏️ CUSTOM", variant="secondary", scale=0, min_width=120)
            source_lbl = gr.Textbox(
                value="",
                show_label=False,
                interactive=False,
                placeholder="Select an article or click Random…",
                elem_classes=["nb-source-label"],
            )

        # ── Main Area: Input | Output | History ──
        with gr.Row(elem_classes=["nb-panels-row"]):

            # Input Panel
            with gr.Column(elem_classes=["nb-panel-input"], scale=5):
                gr.HTML('<div class="nb-label-input">● INPUT TEXT</div>')
                with gr.Column(elem_classes=["nb-panel-body"]):
                    input_box = gr.Textbox(
                        placeholder="Paste any text here, or load an article from above…",
                        lines=16,
                        show_label=False,
                    )
                    input_wc = gr.Textbox(
                        show_label=False,
                        interactive=False,
                        placeholder="",
                        elem_classes=["nb-live-wc"],
                    )

            # Output Panel
            with gr.Column(elem_classes=["nb-panel-output"], scale=5):
                gr.HTML('<div class="nb-label-output">● GENERATED SUMMARY</div>')
                with gr.Column(elem_classes=["nb-panel-body"]):
                    output_box = gr.Textbox(
                        placeholder="Summary streams here word by word…",
                        lines=18,
                        show_label=False,
                        interactive=False,
                    )

            # History Sidebar
            with gr.Column(elem_classes=["nb-panel-history"], scale=2):
                gr.HTML('<div class="nb-label-history">🕐 RECENT</div>')
                with gr.Column(elem_classes=["nb-panel-body"]):
                    history_html = gr.HTML(
                        value=_build_history_html([]),
                        elem_classes=["nb-history-container"],
                    )
                    clear_hist_btn = gr.Button(
                        "CLEAR HISTORY", variant="secondary", size="sm"
                    )

        # ── Stats Bar ──
        with gr.Row(elem_classes=["nb-stats"]):
            wc_box = gr.Textbox(label="Word Count", interactive=False)
            cr_box = gr.Textbox(label="Compression", interactive=False)

        # ── Footer ──
        gr.HTML("""
        <div class="nb-footer">
            Summify • BART-Large-CNN • Hugging Face Transformers • SQuAD v2.0 • Neobrutalism
        </div>
        """)

        # ── Event Wiring ──────────────────────────────────────────────────────
        gen_btn.click(
            fn=_stream_and_update_history,
            inputs=[input_box, word_slider, style_radio, history_state],
            outputs=[output_box, wc_box, cr_box, history_state, history_html],
        )

        input_box.change(fn=_live_word_count, inputs=[input_box], outputs=[input_wc])

        # Random / title load — update input + label + slider, do NOT touch output_box
        rand_btn.click(
            fn=_random_sample,
            inputs=[style_radio],
            outputs=[input_box, source_lbl, word_slider],
        )
        title_dd.change(
            fn=_sample_by_title,
            inputs=[title_dd, style_radio],
            outputs=[input_box, source_lbl, word_slider],
        )
        custom_btn.click(
            fn=lambda style: ("", "✏️ Custom Input Mode",
                              {"Concise": 40, "Balanced": 80, "Detailed": 150}.get(style, 80)),
            inputs=[style_radio],
            outputs=[input_box, source_lbl, word_slider],
        )

        style_radio.change(
            fn=update_slider_from_style,
            inputs=[style_radio, input_box],
            outputs=[word_slider],
        )

        clear_hist_btn.click(
            fn=lambda: ([], _build_history_html([])),
            inputs=[],
            outputs=[history_state, history_html],
        )

    return demo
