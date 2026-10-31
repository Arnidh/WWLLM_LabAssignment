# Model Notes — Verified Checkpoints & Usage

ASR details below were checked against the official Hugging Face model cards
and repository metadata on **2026-10-30**. Param-1 details below were verified
against its model card and Hub config on **2026-10-30**. Notes/translation model
details outside these sections remain unverified.

---

## 1. ASR — AI4Bharat IndicConformer

| Field | Value |
|-------|-------|
| **Primary HF ID** | `ai4bharat/indic-conformer-600m-multilingual` |
| **Verified date** | 2026-10-30 |
| **License** | MIT |
| **Languages** | Assamese (`as`), Bengali (`bn`), Bodo (`brx`), Dogri (`doi`), Gujarati (`gu`), Hindi (`hi`), Kannada (`kn`), Konkani (`kok`), Kashmiri (`ks`), Maithili (`mai`), Malayalam (`ml`), Manipuri (`mni`), Marathi (`mr`), Nepali (`ne`), Odia (`or`), Punjabi (`pa`), Sanskrit (`sa`), Santali (`sat`), Sindhi (`sd`), Tamil (`ta`), Telugu (`te`), Urdu (`ur`) |
| **Architecture** | 600M multilingual Conformer hybrid CTC + RNNT |
| **trust_remote_code** | Required; the card uses `AutoModel` with custom code |
| **HF Token required** | Hugging Face API metadata reports `gated: auto`; provide an `HF_TOKEN` with model access |
| **Input format** | 16 kHz mono waveform tensor (the wrapper prepares `float32` samples) |
| **Output** | Transcript string from the documented CTC or RNNT call |
| **GPU / precision** | Custom ONNX-backed implementation; card provides no fp16/dtype argument. Use `onnxruntime-gpu` to expose CUDA, otherwise CPU ONNX Runtime. No unsupported fp16 override is applied. |

### Verified model-card usage

The official card uses `AutoModel`, not `AutoModelForCTC`/`AutoProcessor` or a
generic Transformers ASR pipeline:

```python
import os
import torch
import torchaudio
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "ai4bharat/indic-conformer-600m-multilingual",
    token=os.getenv("HF_TOKEN"),
    trust_remote_code=True,
)
wav, sample_rate = torchaudio.load("audio.wav")
wav = torch.mean(wav, dim=0, keepdim=True)
if sample_rate != 16000:
    wav = torchaudio.transforms.Resample(sample_rate, 16000)(wav)
transcript = model(wav, "hi", "ctc")
```

Official sources: [model card](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual)
and [IndicConformerASR repository](https://github.com/AI4Bharat/IndicConformerASR).
The model-card installation lists `transformers`, `torchaudio`, `onnx==1.20.1`,
`onnxruntime==1.20.1`, and `onnxruntime-gpu==1.20.1`; for a given environment,
install the CPU or GPU ONNX Runtime package as appropriate.

### Fallback — BharatGen Shrutam-2

| Field | Value |
|-------|-------|
| **Model ID** | `bharatgenai/Shrutam-2` |
| **License** | BharatGen non-commercial license |
| **Languages** | 12: Hindi (`hi`), Marathi (`mr`), Tamil (`ta`), Telugu (`te`), Malayalam (`ml`), Kannada (`kn`), Odia (`or`), Bengali (`bn`), Urdu (`ur`), Assamese (`as`), Gujarati (`gu`), Punjabi (`pa`) |
| **Architecture** | Conformer speech encoder + LLM decoder with MoE projection |
| **trust_remote_code** | Required |
| **HF Token required** | Hugging Face API metadata reports public and not gated |
| **Documented runtime** | Model-card setup uses Python 3.10.14 |
| **Verified usage** | `AutoModel.from_pretrained(..., trust_remote_code=True)`, `AutoTokenizer.from_pretrained(...)`, `model.to(device)`, then `model.transcribe(audio_path, prompts=..., tokenizer=tokenizer)` |
| **Notes** | Used only when IndicConformer fails to load. App writes each prepared chunk to a temporary WAV for the documented file-based call. |

Official source: [Shrutam-2 model card](https://huggingface.co/bharatgenai/Shrutam-2).

### Known Issues / Deviations
- IndicConformer HF metadata reports automatic gating; if access is denied, use
  an accepted HF token. The app preserves the Hub's error and also reports any
  fallback-loading error.
- The gated ONNX weight endpoint returned an authorized response, but this is
  not proof that local inference succeeds. The model's custom
  `from_pretrained` implementation calls `snapshot_download` without filters;
  the pinned repository has 405 files totaling about 2.56 GB. A local load was
  stopped before inference because only about 1.6 GB of RAM was free; an actual
  out-of-memory failure was not observed.
- The HF checkpoint's documented ONNX path does not expose a dtype/fp16 option.
  The wrapper does not invent a precision override; CUDA acceleration requires
  a CUDA-enabled ONNX Runtime installation.
- The fallback's license is non-commercial. Check the linked license before
  using it outside this academic, non-commercial project.

---

## 2. Notes + MCQ — BharatGen Param-1

| Field | Value |
|-------|-------|
| **HF ID** | `bharatgenai/Param-1-2.9B-Instruct` |
| **Verified date** | 2026-10-30 |
| **Model** | Param 1-2.9B-Instruct; early SFT checkpoint; bilingual Hindi and English |
| **Architecture** | `ParamBharatGenForCausalLM`; custom Transformers model code |
| **Context window** | `max_position_embeddings: 8192` in the published `config.json` |
| **License** | **Unresolved conflict:** Hub API metadata labels it Apache-2.0; the model card prose says MIT. The card links `LICENSE`, but the linked file returned HTTP 404 and the Hub file listing did not include it at verification time. Do not assume either license; check with BharatGen before redistribution or other uses where the license matters. |
| **trust_remote_code** | `AutoModelForCausalLM` requires `trust_remote_code=True`; card uses `trust_remote_code=False` for `AutoTokenizer` |
| **HF Token required** | No; Hub metadata reports public and not gated. The application still accepts `HF_TOKEN` from the environment. |
| **Precision** | Card describes training at `bf16-mixed`; app selects fp16 on CUDA and fp32 on CPU. 4-bit/8-bit modes use BitsAndBytes when configured and CUDA is available. |
| **Model-card generation example** | `tokenizer.apply_chat_template(conversation=..., return_tensors="pt", add_generation_prompt=True)` followed by `model.generate(inputs, ...)` |

### Chat template and verified loading pattern

The card demonstrates a `conversation` list of `{role, content}` messages with
`system` then `user` roles and calls
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`. The published
`chat_template.jinja` formats messages using `<|im_start|>{role}\n{content}<|im_end|>`
and opens the assistant turn as `<|im_start|>assistant\n`. The app uses the
tokenizer's template rather than hard-coding an assumed chat format.

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "bharatgenai/Param-1-2.9B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token=os.getenv("HF_TOKEN"),
    trust_remote_code=False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=os.getenv("HF_TOKEN"),
    trust_remote_code=True,
    device_map="auto",
)
conversation = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the BharatGen Mission?"},
]
inputs = tokenizer.apply_chat_template(
    conversation=conversation,
    return_tensors="pt",
    add_generation_prompt=True,
).to(model.device)
with torch.no_grad():
    output = model.generate(
        inputs,
        max_new_tokens=300,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.6,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
answer = tokenizer.decode(
    output[0][inputs.shape[-1]:],
    skip_special_tokens=True,
)
```

The official example uses temperature `0.6`; the project intentionally caps it
at `0.3` to follow the task's grounded-generation requirement. Quantized
BitsAndBytes loading is an application option, not a feature demonstrated in
the model card. It requires a compatible CUDA/BitsAndBytes environment and has
not been tested against this checkpoint on a T4.

Sources: [Param-1 model card](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct),
[Hub config.json](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct/raw/main/config.json),
[Hub tokenizer_config.json](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct/raw/main/tokenizer_config.json),
and [Hub chat_template.jinja](https://huggingface.co/bharatgenai/Param-1-2.9B-Instruct/raw/main/chat_template.jinja).

---

## 3. Translation — AI4Bharat IndicTrans2 (configured; not inference-verified)

| Direction | HF ID | Verified |
|-----------|--------|---------|
| Indic → English | `ai4bharat/indictrans2-indic-en-dist-200M` | Checkpoint is configured; real loading/translation not verified |
| English → Indic | `ai4bharat/indictrans2-en-indic-dist-200M` | Checkpoint is configured; real loading/translation not verified |
| Indic → Indic | `ai4bharat/indictrans2-indic-indic-dist-320M` | Checkpoint is configured; real loading/translation not verified |

| Field | Value |
|-------|-------|
| **Toolkit** | `IndicTransToolkit` is referenced by the wrapper; installation and import path are not verified here |
| **Language codes** | The wrapper configures Flores-200-style codes; mappings have not been checked by a real translation call |
| **License / token / remote-code behavior** | Not independently verified for these checkpoints |
| **Inference** | No verified usage snippet is included because a real model call has not been run |

The translation checkpoints and wrapper are part of the project, but no
translation score or quality claim should be reported until model access,
runtime compatibility, licensing, and a real translation example are checked
against the official [IndicTrans2 repository](https://github.com/AI4Bharat/IndicTrans2).

---

## Verification Checklist

- [x] IndicConformer: verify exact HF model ID, MIT license, and 22 language codes
- [x] IndicConformer: verify language code format (`"hi"`, `"kn"`, etc.)
- [x] IndicConformer: confirm `trust_remote_code=True` and documented `AutoModel` API
- [ ] IndicConformer: run a real inference smoke test on a machine with sufficient RAM (not completed locally)
- [x] BharatGen fallback: verify Shrutam-2 ID, documented API, coverage, and license
- [x] Param-1: verify exact instruct checkpoint ID and 8,192 positional limit
- [x] Param-1: verify tokenizer chat-template API and message roles
- [x] Param-1: document conflict between Hub license metadata and card prose
- [ ] Param-1: resolve license conflict with BharatGen before redistribution
- [ ] Param-1: test normal and quantized loading on T4 GPU
- [ ] IndicTrans2: install IndicTransToolkit and verify `IndicProcessor` import path
- [ ] IndicTrans2: confirm Flores-200 codes for all target languages
- [x] ASR: record HF token gating requirements and model licenses
- [ ] Notes and translation: record exact license (relevant for academic submission)
