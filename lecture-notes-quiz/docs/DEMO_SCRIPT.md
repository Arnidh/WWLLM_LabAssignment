# DEMO_SCRIPT.md — 3-Minute Demo Guide (CS3235 CIE3)

**Total time: ~3 minutes**

---

## Slide 0 — Introduction (20 s)
> "I built a system that takes a Hindi lecture recording, transcribes it using India's sovereign AI model IndicConformer, generates structured notes and 5 MCQs using BharatGen Param-1, and optionally translates everything using IndicTrans2."

## Slide 1 — Problem (15 s)
> "Students often miss lectures or struggle to take notes in their native language. This tool automates structured note-taking and quiz generation entirely in Indian languages, using open-source Indian AI models."

## Slide 2 — Architecture (20 s)
*(Point to the Mermaid diagram in README)*
> "Audio goes through four stages: ASR → Notes → Quiz → Translation. Each stage is isolated, lazy-loaded, and can free GPU memory before the next stage. There's a MOCK mode for development without a GPU."

---

## Live Demo (90 s)

1. **Open the Gradio app** (already running at `localhost:7860`)
2. **Upload** the sample Hindi lecture clip (`data/audio/hindi_clean_01.wav`)
3. **Select** language: Hindi (hi)
4. **Select** translate to: English
5. Click **Generate Notes & Quiz**
6. **Tab: Transcript** — show the ASR output with timestamps
7. **Tab: Notes** — show the structured Markdown (Title, Key Points, Key Terms, Summary)
8. **Tab: Quiz** — click an option on Q1 to reveal the correct answer and evidence
9. **Tab: Translated Notes** — show the English translation
10. **Download** notes (.md) and quiz (.json)
11. **Point to timings panel** — "Each stage logged its latency"

---

## Slide 3 — Evaluation (25 s)
> "I tested on 30 clips across 7 categories: clean speech, noisy speech, fast speech, code-mixed, numbers, short/long clips, and edge cases. WER on clean Hindi was TODO%. MCQ auto-validity was TODO/5 on average. Failure analysis is in results/failure_analysis.md."

## Slide 4 — Limitations & Ethics (20 s)
> "Coverage is currently best for Hindi; Kannada and Tamil need more testing. ASR degrades on noisy or fast speech. All prompts explicitly instruct the model to avoid hallucination. The system should never be used as a replacement for expert-written study material."

## Slide 5 — Q&A (10 s)
> "Questions?"

---

## Backup / Fallback Plan
- If live GPU is unavailable: run in `MOCK_MODELS=1` mode — UI and all tabs work identically with canned outputs.
- If internet is unavailable: models must be pre-cached in `~/.cache/huggingface/`.

