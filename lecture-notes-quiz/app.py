"""
app.py
------
Gradio frontend for the Multilingual Lecture-to-Notes & Quiz Generator.

Tabs
----
1. Transcript   – full ASR output with per-chunk view
2. Translation  – transcript translated into the selected language
3. Notes        – Markdown notes in the selected language
4. Quiz         – interactive MCQs in the selected language

Run
---
    # MOCK mode (no GPU required):
    MOCK_MODELS=1 python app.py

    # Real mode:
    HF_TOKEN=hf_... python app.py

    # Colab (see notebooks/colab_run.ipynb):
    python app.py --share
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr

# Ensure the project root is on sys.path when running from any directory
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from src.pipeline import run_pipeline
from src.pdf_export import save_notes_pdf, save_quiz_pdf
from src.utils import get_logger, is_mock_mode, load_config

logger = get_logger("app")
cfg = load_config()

# ── Language options ──────────────────────────────────────────────────────────

LANG_DISPLAY: dict[str, str] = cfg.get("supported_languages", {"hi": "Hindi"})
ASR_LANG_CODES = set(cfg.get("asr_supported_languages", LANG_DISPLAY))
TRANS_DISPLAY: dict[str, str] = cfg.get(
    "translation_targets",
    {"en": "English", "hi": "Hindi", "kn": "Kannada", "ta": "Tamil"},
)

LANG_CHOICES = [
    (f"{name} ({code})", code)
    for code, name in LANG_DISPLAY.items()
    if code in ASR_LANG_CODES
]
TRANS_CHOICES = [("(No translation)", "")] + [
    (f"{name} ({code})", code) for code, name in TRANS_DISPLAY.items()
]

DEFAULT_LANG = cfg.get("default_language", "hi")
DEFAULT_TRANSLATION = cfg.get("default_translation_language", "en")

# ── Timing banner helper ──────────────────────────────────────────────────────

def _timing_html(timings: dict[str, float], total: float) -> str:
    rows = "".join(
        f"<tr><td style='padding:2px 12px'>{stage.upper()}</td>"
        f"<td style='padding:2px 8px;text-align:right'>{lat:.2f} s</td></tr>"
        for stage, lat in timings.items()
    )
    return (
        "<table style='font-family:monospace;font-size:0.85em;border-collapse:collapse'>"
        f"{rows}"
        f"<tr style='border-top:1px solid #ccc'>"
        f"<td style='padding:2px 12px'><b>TOTAL</b></td>"
        f"<td style='padding:2px 8px;text-align:right'><b>{total:.2f} s</b></td></tr>"
        "</table>"
    )


# ── Quiz HTML renderer ────────────────────────────────────────────────────────

def _render_quiz_html(mcqs: list[dict]) -> str:
    """Render MCQs with a native, keyboard-accessible answer reveal."""
    if not mcqs:
        return "<p style='color:#4b5563'>No questions generated.</p>"

    items_html = []
    for i, q in enumerate(mcqs):
        qn = i + 1
        question = html.escape(str(q.get("question", "")))
        options: dict = q.get("options", {})
        answer = html.escape(str(q.get("answer", "")), quote=True)
        explanation = html.escape(str(q.get("explanation", "")))
        evidence = html.escape(str(q.get("evidence", "")))

        opts_html = ""
        for letter, text in options.items():
            safe_letter = html.escape(str(letter), quote=True)
            safe_text = html.escape(str(text))
            opts_html += (
                "<label style='display:block;margin:4px 0;cursor:pointer;"
                "color:#1f2937'>"
                f"<input type='radio' name='q{qn}' value='{safe_letter}'>"
                f" <b style='color:#111827 !important'>{safe_letter}.</b> "
                f"<span style='color:#1f2937 !important'>{safe_text}</span></label>"
            )

        items_html.append(f"""
<div id="q{qn}" style="
  border:1px solid #ddd; border-radius:8px; padding:14px 18px;
  margin-bottom:16px; background:#fff; color:#1f2937">
  <p style="font-weight:600;margin:0 0 8px 0;color:#111827">Q{qn}. {question}</p>
  {opts_html}
  <details style="margin-top:10px">
    <summary style="cursor:pointer;color:#1f2937;font-weight:600">
      Show answer and explanation
    </summary>
    <div style="margin-top:8px;padding:8px 12px;border-radius:6px;
      background:#e8f5e9;border-left:4px solid #4caf50;color:#1f2937">
      <b style="color:#111827 !important">✔ Correct Answer: {answer}</b><br>
      <span style="font-size:0.9em;color:#1f2937 !important">{explanation}</span><br>
      <span style="font-size:0.8em;color:#374151 !important;margin-top:4px;display:block">
        📖 Evidence: <i style="color:#374151 !important">{evidence}</i>
      </span>
    </div>
  </details>
</div>
""")

    return (
        "<div style='max-width:700px;background:#fff;color:#1f2937'>"
        + "".join(items_html)
        + "</div>"
    )


# ── Download helpers ──────────────────────────────────────────────────────────

def _save_notes(notes_md: str) -> str:
    """Write notes to a temp file and return the path (Gradio download)."""
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w", encoding="utf-8"
    )
    tmp.write(notes_md)
    tmp.close()
    return tmp.name


def _save_quiz(mcqs: list[dict]) -> str:
    """Write MCQs to a temp JSON file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".json", mode="w", encoding="utf-8"
    )
    json.dump(mcqs, tmp, ensure_ascii=False, indent=2)
    tmp.close()
    return tmp.name


# ── Core pipeline callback ────────────────────────────────────────────────────

def generate(
    audio_input,        # file path from gr.Audio (filepath mode)
    lang_code: str,
    translate_to: str,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Main Gradio callback. Returns an 11-tuple:
      (status_html, transcript_text, chunks_text, notes_md,
       quiz_html, translated_transcript, timing_html, notes_file, notes_pdf,
       quiz_file, quiz_pdf)
    """
    mock_banner = (
        "<span style='background:#fff3cd;padding:2px 8px;border-radius:4px;"
        "font-size:0.85em'>⚠ MOCK MODE – models not loaded</span>"
        if is_mock_mode() else ""
    )

    if audio_input is None:
        err = "❌ Please upload an audio file or record via microphone."
        return (
            err, "", "", "", "<p>–</p>", "", "", None, None, None, None
        )

    try:
        result = run_pipeline(
            audio_path=audio_input,
            lang=lang_code,
            translate_to=translate_to if translate_to else None,
            progress_callback=lambda stage, fraction: progress(
                fraction, desc=stage
            ),
        )
    except Exception as exc:
        logger.exception("Pipeline crashed")
        return (
            f"❌ Unexpected error: {html.escape(str(exc))}",
            "", "", "", "<p>–</p>", "", "", None, None, None, None,
        )

    d = result.to_dict()

    # ── Format transcript tab ─────────────────────────────────────────────
    transcript_full = d["transcript"]["full_text"]
    chunks_lines = []
    for c in d["transcript"]["chunks"]:
        chunks_lines.append(
            f"[{c['start_s']:.1f}s – {c['end_s']:.1f}s]\n{c['text']}"
        )
    chunks_text = "\n\n---\n".join(chunks_lines)

    # ── Notes ─────────────────────────────────────────────────────────────
    notes_md: str = d["notes"]["markdown"]
    notes_file = _save_notes(notes_md) if notes_md else None

    mr_note = ""
    if d["notes"].get("used_map_reduce"):
        mr_note = "\n\n> ℹ️ *Map-reduce was used (transcript was long).*"
    notes_pdf_file = save_notes_pdf(notes_md + mr_note) if notes_md else None

    # ── Quiz ──────────────────────────────────────────────────────────────
    mcqs = d["quiz"]["mcqs"]
    quiz_html = _render_quiz_html(mcqs)
    quiz_warnings = d["quiz"].get("warnings", [])
    if quiz_warnings:
        warn_str = "\n".join(f"⚠ {w}" for w in quiz_warnings)
        quiz_html += f"<pre style='color:#856404;background:#fff3cd;padding:8px'>{warn_str}</pre>"
    quiz_file = _save_quiz(mcqs) if mcqs else None
    quiz_pdf_file = save_quiz_pdf(mcqs) if mcqs else None

    # ── Translation ───────────────────────────────────────────────────────
    translated_text: str = d["translation"]["text"]

    # ── Timing ────────────────────────────────────────────────────────────
    timing_html = _timing_html(d["timings"], d["total_latency_s"])

    if result.success:
        if d["warnings"]:
            warning = html.escape("; ".join(d["warnings"]))
            status = f"⚠️ Completed with warning: {warning}  {mock_banner}"
        else:
            status = f"✅ Done!  {mock_banner}"
    else:
        stage = html.escape((result.error_stage or "unknown").upper())
        message = html.escape(result.error_message or "No additional error details.")
        status = f"❌ Pipeline failed at stage <b>{stage}</b>: {message}"

    return (
        status,
        transcript_full,
        chunks_text,
        notes_md + mr_note,
        quiz_html,
        translated_text,
        timing_html,
        notes_file,
        notes_pdf_file,
        quiz_file,
        quiz_pdf_file,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    mock_badge = (
        "🟡 **MOCK MODE** – returns canned outputs (set `MOCK_MODELS=0` for real inference)"
        if is_mock_mode()
        else "🟢 **Live Mode** – real models will be loaded on first run"
    )

    with gr.Blocks(
        title="Lecture-to-Notes & Quiz Generator",
        theme=gr.themes.Soft(),
        css="""
        .tab-content { padding: 12px 0; }
        #status-box { font-size: 1.05em; min-height: 32px; }
        """,
    ) as demo:
        gr.Markdown(
            "# 🎓 Multilingual Lecture-to-Notes & Quiz Generator\n"
            "Upload a Hindi/Kannada/Tamil lecture recording → get structured notes "
            "and 5 MCQs instantly.\n\n"
            f"> {mock_badge}"
        )

        with gr.Row():
            with gr.Column(scale=1):
                # ── Inputs ────────────────────────────────────────────────
                audio_in = gr.Audio(
                    label="📤 Upload lecture audio (wav / mp3 / m4a)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                lang_dd = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value=DEFAULT_LANG,
                    label="🌐 Lecture language",
                )

                trans_dd = gr.Dropdown(
                    choices=TRANS_CHOICES,
                    value=DEFAULT_TRANSLATION,
                    label="🔄 Translate transcript and generate notes/quiz in",
                )

                generate_btn = gr.Button(
                    "🚀 Generate Notes & Quiz",
                    variant="primary",
                    size="lg",
                )

                status_box = gr.HTML(
                    label="Status",
                    elem_id="status-box",
                )

            with gr.Column(scale=2):
                # ── Output tabs ───────────────────────────────────────────
                with gr.Tabs():
                    with gr.Tab("📝 Transcript"):
                        transcript_box = gr.Textbox(
                            label="Full transcript",
                            lines=8,
                            show_copy_button=True,
                        )
                        with gr.Accordion("Per-chunk transcript", open=False):
                            chunks_box = gr.Textbox(
                                label="Chunks with timestamps",
                                lines=10,
                                show_copy_button=True,
                            )
                        timing_box = gr.HTML(label="⏱ Per-stage latency")

                    with gr.Tab("🌍 Translation"):
                        translated_box = gr.Textbox(
                            label="Translated transcript",
                            lines=8,
                            show_copy_button=True,
                        )

                    with gr.Tab("📚 Notes"):
                        notes_box = gr.Markdown(label="Structured Notes")
                        notes_dl = gr.File(
                            label="⬇ Download notes (.md)",
                            visible=True,
                        )
                        notes_pdf_dl = gr.File(
                            label="⬇ Download notes (.pdf)",
                            visible=True,
                        )

                    with gr.Tab("❓ Quiz"):
                        quiz_html_box = gr.HTML(label="Interactive Quiz")
                        quiz_dl = gr.File(
                            label="⬇ Download quiz (.json)",
                            visible=True,
                        )
                        quiz_pdf_dl = gr.File(
                            label="⬇ Download quiz (.pdf)",
                            visible=True,
                        )

        # ── Wire up ───────────────────────────────────────────────────────
        generate_btn.click(
            fn=generate,
            inputs=[audio_in, lang_dd, trans_dd],
            outputs=[
                status_box,
                transcript_box,
                chunks_box,
                notes_box,
                quiz_html_box,
                translated_box,
                timing_box,
                notes_dl,
                notes_pdf_dl,
                quiz_dl,
                quiz_pdf_dl,
            ],
            show_progress="full",
        )

        gr.Markdown(
            "---\n"
            "**Models**: "
            "[IndicConformer 600M](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) · "
            "[BharatGen Param-1](https://huggingface.co/bharatgenai) · "
            "[IndicTrans2](https://github.com/AI4Bharat/IndicTrans2) | "
            "CS3235 – CIE3 Lab Assessment"
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Gradio app")
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    logger.info("Starting app (MOCK_MODELS=%s)", os.getenv("MOCK_MODELS", "0"))

    demo = build_ui()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0",
    )
