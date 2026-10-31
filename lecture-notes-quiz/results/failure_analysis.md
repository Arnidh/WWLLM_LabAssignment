# failure_analysis.md — ASR / Notes / MCQ Failure Analysis

## Top Failure Categories

> TODO: Fill in after running evaluation. Categories below are templates based on expected failure modes.

1. **Noisy audio → high WER** — Background noise degrades IndicConformer output significantly.
2. **Fast speech → word merging** — Model merges adjacent words when speech rate is high.
3. **Code-mixed text → LLM confusion** — Param-1 may hallucinate when switching between Hindi and English technical terms.
4. **Long transcript → map-reduce context loss** — Summarisation of individual chunks may lose inter-chunk connections.
5. **MCQ evidence not found** — Model generates plausible-sounding evidence not actually in transcript.

---

## Concrete Failure Examples

> TODO: Replace each example below with a real failure from your evaluation run.

### Example 1 — Noisy Speech ASR Error
- **Input**: `data/audio/hindi_noisy_01.wav`
- **Reference**: `न्यूटन का पहला नियम जड़त्व का नियम है`
- **ASR Output**: `TODO`
- **WER**: `TODO`
- **Why it failed**: TODO (e.g., road noise at 65 dB caused [inaudible] segments)
- **Root cause stage**: ASR
- **Possible fix**: Pre-process audio with noise reduction (noisereduce library) before ASR.

### Example 2 — Code-Mixed MCQ Hallucination
- **Input transcript**: `TODO`
- **Generated MCQ**: `TODO`
- **Issue**: Evidence field references a sentence not in the transcript.
- **Root cause stage**: MCQ generation (Param-1)
- **Possible fix**: Stricter evidence validation; increase temperature penalty for evidence field.

### Example 3 — Map-Reduce Note Loss
- **Input**: `data/audio/hindi_long_01.wav` (>10 min)
- **Expected key point**: `TODO`
- **Generated notes**: (key point missing)
- **Root cause stage**: Notes (map-reduce)
- **Possible fix**: Increase chunk overlap; add a verification pass that checks all expected terms appear.

### Example 4 — Fast Speech Word Merging
- **Input**: `data/audio/hindi_fast_01.wav`
- **Reference**: `TODO`
- **ASR Output**: `TODO`
- **Root cause stage**: ASR
- **Possible fix**: Slow audio by 0.8× using `librosa.effects.time_stretch` before ASR.

### Example 5 — Silent / Non-Lecture Audio
- **Input**: `data/audio/silent_01.wav`
- **ASR Output**: `""` or random noise tokens
- **Notes Output**: System correctly says "Transcript appears empty"
- **Root cause stage**: ASR (expected; input is edge case)
- **Possible fix**: Add audio energy check before running ASR; warn user if energy < threshold.

---

## Root Cause Distribution (Placeholder)

| Root Cause | Count | % of Failures |
|------------|-------|---------------|
| ASR | TODO | TODO |
| Notes (Param-1) | TODO | TODO |
| MCQ (Param-1) | TODO | TODO |
| Translation | TODO | TODO |
| Infrastructure | TODO | TODO |

---

## Possible Fixes (Priority Order)

1. **Noise reduction pre-processing** — Quick win for noisy speech.
2. **Evidence post-processing** — Reject MCQs where evidence is not a substring of transcript.
3. **Chunk overlap increase** — Reduce map-reduce context loss.
4. **Instruct format tuning** — Verify Param-1 chat template; may significantly improve output quality.
5. **Language-specific test data** — Add Kannada and Tamil clips for proper evaluation.

