# ASR Failure Analysis

## Benchmark scope

The current benchmark is a seeded 30-sample subset of Google FLEURS test data:
10 read-speech clips each in Hindi, Kannada, and Tamil. It is not lecture audio
and does not test noise, fast speech, code-switching, long-form context,
summarization, translation, or quiz quality.

## Results

No real ASR inference or WER/CER result was produced in this workspace. The
authorized checkpoint is accessible, but its custom loader downloads the full
2.56 GB repository; only about 1.6 GB of RAM was free, so the smoke test was
stopped before inference to avoid risking resource exhaustion. No actual
out-of-memory failure or ASR error was observed. Run
`python eval/run_asr_inference.py` followed by
`python eval/wer_eval.py` on a better-provisioned runtime to produce results.

No error cause is inferred from WER alone. Investigate the worst per-sample
rows alongside their audio and reference before assigning a root cause.
Notes and quiz failure analysis requires separate reference material and human
ratings; those results are intentionally not fabricated from ASR transcripts.
