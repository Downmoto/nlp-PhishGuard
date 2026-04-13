"""Gradio web interface for PhishGuard."""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from phishguard.config import load_config
from phishguard.inference.predictor import PhishGuardPredictor

# Loaded lazily on first use
_predictor: PhishGuardPredictor | None = None


def _get_predictor() -> PhishGuardPredictor:
    global _predictor
    if _predictor is None:
        config = load_config()
        ckpt = config["output"]["best_checkpoint_dir"]
        if not Path(ckpt).exists():
            raise RuntimeError(
                f"Checkpoint not found at {ckpt}. "
                "Run scripts/train.py first."
            )
        _predictor = PhishGuardPredictor(
            checkpoint_dir=ckpt,
            max_length=config["model"]["max_seq_length"],
        )
    return _predictor


def _classify(email_text: str) -> tuple[str, str, str]:
    """Gradio callback: return (verdict, confidence, highlighted tokens)."""
    if not email_text.strip():
        return "—", "—", "Please enter some email text."

    predictor = _get_predictor()
    result = predictor.predict(email_text)

    label = result["label"].upper()
    confidence = f"{result['confidence']:.1%}"

    # Build a simple highlighted-token string (top-20 by score)
    token_scores = result["token_scores"]
    top_tokens = sorted(token_scores, key=lambda x: x["score"], reverse=True)[:20]
    highlights = "  ".join(
        f"**{t['token']}** ({t['score']:.3f})" for t in top_tokens
        if not t["token"].startswith("[") or t["token"] in ("[URL]", "[EMAIL]", "[PHONE]")
    )

    return label, confidence, highlights or "—"


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks application."""
    with gr.Blocks(title="PhishGuard — Phishing Email Detector") as demo:
        gr.Markdown("# PhishGuard\nPaste an email body below to analyse it for phishing indicators.")

        with gr.Row():
            with gr.Column(scale=2):
                email_input = gr.Textbox(
                    label="Email Text",
                    placeholder="Paste email content here…",
                    lines=12,
                )
                analyse_btn = gr.Button("Analyse", variant="primary")
            with gr.Column(scale=1):
                verdict_out = gr.Textbox(label="Verdict", interactive=False)
                confidence_out = gr.Textbox(label="Confidence", interactive=False)
                tokens_out = gr.Markdown(label="Top Attention Tokens")

        analyse_btn.click(
            fn=_classify,
            inputs=[email_input],
            outputs=[verdict_out, confidence_out, tokens_out],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
