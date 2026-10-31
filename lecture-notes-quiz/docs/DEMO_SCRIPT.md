# DEMO_SCRIPT.md — 3-Minute Demo Guide (CS3235 CIE3)

**Total time: ~3 minutes**

---

## Slide 0 — Introduction (20 s)
> "I built a system that takes a Hindi lecture recording, keeps its original transcript, translates it using IndicTrans2, and generates structured notes and 5 MCQs in the selected language using BharatGen Param-1."

## Slide 1 — Problem (15 s)
> "Students often miss lectures or struggle to take notes in their native language. This tool automates structured note-taking and quiz generation entirely in Indian languages, using open-source Indian AI models."

## Slide 2 — Architecture (20 s)
*(Point to the Mermaid diagram in README)*
> "Audio goes through four stages: ASR → Translation → Notes → Quiz. The original transcript remains available, while notes and questions use the translated transcript. There's a MOCK mode for development without a GPU."

---

## Live Demo (90 s)

1. **Open the Gradio app** at `localhost:7860`
2. **Upload** a Hindi WAV clip (for example, one listed in `data/test_cases.csv` after dataset preparation)
3. **Select** language: Hindi (hi)
4. **Select** translate to: English
5. Click **Generate Notes & Quiz**
6. **Tab: Transcript** — show the ASR output with timestamps
7. **Tab: Translation** — show the English transcript
8. **Tab: Notes** — show the English structured notes (Title, Key Points, Key Terms, Summary)
9. **Tab: Quiz** — click an option on Q1 to reveal the correct answer and evidence
10. **Download** notes and quiz as PDF (or Markdown/JSON)
11. **Point to timings panel** — "Each stage logged its latency"

---

## Slide 3 — Evaluation (25 s)
> "I prepared 30 held-out FLEURS read-speech clips—10 each in Hindi, Kannada, and Tamil. I stopped local inference before the model finished loading due to limited memory, so no WER/CER is claimed. On a T4 runtime, the reproducible commands generate per-language results. FLEURS is not lecture audio and does not evaluate notes or quizzes."

## Slide 4 — Limitations & Ethics (20 s)
> "The ASR benchmark covers a small read-speech subset, not noisy, fast, code-switched, or lecture speech. Notes and quiz quality still need human review. The system should never be used as a replacement for expert-written study material."

## Slide 5 — Q&A (10 s)
> "Questions?"

---

## Backup / Fallback Plan
- If live GPU is unavailable: run in `MOCK_MODELS=1` mode — UI and all tabs work identically with canned outputs.
- If internet is unavailable: models must be pre-cached in `~/.cache/huggingface/`.
