# Base models, their licences and their cutoffs

**Chosen, 2026-09-19: Gemma 4 E2B, Gemma 4 E4B and OLMo 3 7B.** The split's cutoff is
**2025-01-31**, the latest of the three. Revisions and weight digests were resolved and
recorded on 2026-09-21, so this gate is clear.

A gate, not a placeholder. Three things have to be recorded here before a training run
starts, and two of them decide whether the project's headline number means anything.

## What goes here

| Column | Why it is not optional |
|---|---|
| Family, exact revision and parameter count | Three sizes, roughly 1.5B, 4B and 8B |
| **Licence**, quoted and linked | Only families whose terms permit publishing adapters *and* merged weights. A base whose licence forbids redistribution cannot be one of the three, however good it is |
| **Documented training cutoff**, with the source | The split depends on it. `assign_splits` takes it as `cutoff`, and the post-cutoff test set is the headline |
| Context length, tokeniser, chat template | The prompt is built against these and the load test runs at the context length recorded here |
| Where the weights were obtained and their digest | So the run is reproducible from the recipe alone |

## The cutoff is the load-bearing one

The headline number is accuracy on filings published after the base model's training
cutoff, and the pre-cutoff set is reported beside it as the contamination gap. If a
model's cutoff is undocumented, or documented only as a month with no day, that has to be
recorded here as the vagueness it is, and the split takes the *later* end of the stated
range so the post-cutoff set stays honest. A base whose cutoff cannot be established at all
is not usable for this project's central claim, and using it anyway while reporting a
post-cutoff number would be the kind of quiet dishonesty the whole repository is arranged
against.

Where the three cutoffs differ, the split uses the latest of them, so that one held-out set
is after every base model's cutoff and all three sizes are measured on identical items.

## Chosen

| | Small: Gemma 4 E2B | Middle: Gemma 4 E4B | Large: OLMo 3 7B |
|---|---|---|---|
| Base checkpoint | [`google/gemma-4-E2B`](https://huggingface.co/google/gemma-4-E2B) | [`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) | [`allenai/Olmo-3-1025-7B`](https://huggingface.co/allenai/Olmo-3-1025-7B) |
| Untuned baseline | `google/gemma-4-E2B-it` | `google/gemma-4-E4B-it` | `allenai/Olmo-3-7B-Instruct` |
| Parameters | 2.3B effective, 5.1B with per-layer embeddings | 4.5B effective, 8B with per-layer embeddings | 7B |
| Licence | Apache 2.0, ungated | Apache 2.0, ungated | Apache 2.0 |
| Documented cutoff | "a cutoff date of January 2025" ([card](https://ai.google.dev/gemma/docs/core/model_card_4)) | same | "Date cutoff: Dec 2024" ([card](https://huggingface.co/allenai/Olmo-3-1025-7B)) |
| Context | 128k | 128k | 65k |
| Reasoning | Opt-in by a `<\|think\|>` token in the system prompt | same | The base has none; a separate Think variant exists and is not used |
| Training data | Not released | Not released | **Released (Dolma 3)**, so contamination can be checked against the corpus directly |
| Base revision | `d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f` | `411aa17b749aa952df1359d2dcea73917a544d9a` | `a81bae42db3975be1671e27b9c9a56da1a9f980f` |
| Weights on disk, bf16 | 10.25 GB, one file | 15.99 GB, one file | 14.60 GB, three files |
| Untuned baseline revision | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` | `ee0ef6023621cff504d758262d4e04895a5af4a2` | `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |

Licences and cutoffs were read from the model cards and Hugging Face pages on 2026-09-19.
Month-only cutoffs take the month's last day, so the split uses 2025-01-31, after all three.

**Revisions and digests, resolved 2026-09-21** from the Hugging Face API, which also
confirmed all six repositories ungated and Apache 2.0 on the day. A run passes the
revision to `from_pretrained`, so a later push to any of these repositories cannot change
what a recorded run trained on, and `RunRecord` stores it. The sha256 of each weight file
at the base revision, for checking a download:

| Base | File | sha256 |
|---|---|---|
| Gemma 4 E2B | `model.safetensors` | `76dc84a5a805a2c8b91e9ccc00b8dbf8f4a99bf0d56ab25832f6e6addd4f7f57` |
| Gemma 4 E4B | `model.safetensors` | `43fb96cec3045b72852c787540300dc5b258634b7a025f7c80355ac0788b9651` |
| OLMo 3 7B | `model-00001-of-00003.safetensors` | `0490d6668e613a29b23367e3a7aa9cc6aced3d162694445bb969ed7622b3c4e2` |
| OLMo 3 7B | `model-00002-of-00003.safetensors` | `e127ea479fb6e208fe9d48d23b11212b5722f4873f6eef9c009b7a855866c641` |
| OLMo 3 7B | `model-00003-of-00003.safetensors` | `f3ddff10052ffe5de5c6b4cad45c422c0d898acc6beb21b1b8531244adfb3c70` |

The on-disk sizes say what a card has to hold. Gemma's "E2B" is 10 GB in bf16 because the
per-layer embeddings are counted in the file and not in the name; in 4-bit for QLoRA the
three load at roughly a quarter of those sizes, so all three train on a 24 GB card, which
is the class the GPU decision is between.

**Reasoning.** The fine-tunes train the base checkpoints to emit the JSON directly, with no
thinking tokens, so reasoning costs no latency and no money. The untuned Gemma baselines are
measured with thinking off and on, which says what reasoning buys on this task before any
fine-tuning.

**What mixing families costs.** The three sizes are not one family, so the size axis of the
curve mixes architecture with scale, and the per-layer embeddings make Gemma's "effective"
size smaller than what serving memory pays for. The cards and the Pareto chart state both
counts. No single family with a documented cutoff offered all three sizes on 2026-09-19.

## Shortlist, 2026-09-19

The remaining two sizes are chosen from this list, and every licence and cutoff is re-read
from its model card on the day each is chosen. The constraint that bites is not quality or
licence, it is whether the training cutoff is documented at all.

| Family | Sizes near 1.5B, 4B, 8B | Licence | Documented cutoff | Notes |
|---|---|---|---|---|
| [OLMo 3](https://huggingface.co/allenai/Olmo-3-1025-7B) (Ai2) | 7B (and 32B) | Apache 2.0 | "Date cutoff: Dec 2024" | **Training data is released (Dolma 3).** Contamination can be checked against the corpus directly rather than inferred from a date, which no other family allows. No small size in this generation. Base checkpoints published. 65k context |
| [Gemma 4](https://ai.google.dev/gemma/docs/core/model_card_4) (Google) | E2B, E4B (then 12B) | Apache 2.0, no separate Google terms | "cutoff date of January 2025" | Month only, so the split takes 2025-01-31. "E" sizes are effective parameters; the weights on disk are larger, which matters for serving memory and is read from the card on the day. 128k context |
| [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B) (Alibaba) | 0.8B, 2B, 4B, 9B | Apache 2.0 | **Not stated on the model card** | The natural single family for three sizes, with base checkpoints. Hybrid Gated DeltaNet and sparse MoE, with a vision encoder. Unusable for the headline as things stand, by the rule above, unless a cutoff is documented by the day |
| [Granite 4.1](https://huggingface.co/ibm-granite/granite-4.1-8b) (IBM) | 3B, 8B | Apache 2.0 | **Not stated on the model card** | Dense, released 2026-04-29, 128k context. Same problem |
| Ministral 3 (Mistral) | 3B, 8B | Apache 2.0 | Not checked yet | To read on the day |

Sources read 2026-09-19: the model cards linked above, and a
[survey of 2026 small models](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
for the list of families to check.

**What this pointed to before E4B was chosen.** A mixed trio with documented
cutoffs: Gemma 4 E2B and E4B for the two small sizes and OLMo 3 7B for the large one. All
three are Apache 2.0 and publish base checkpoints. The latest cutoff is January 2025, so
the post-cutoff test set is every test-pool filing from February 2025 on, which is most of
2025 and 2026 rather than one quarter of 2027, and the contamination gap can be
cross-checked on the OLMo size against its released training data. The cost is that the
three sizes are not one family, so the size axis of the curve mixes architecture with
scale; the card for each size says so. If Qwen or Granite document their cutoffs by the
day, a single-family trio becomes possible and is preferable for that reason.

## Model cards

Every published adapter and every merged or quantised weight file gets a card under
[`cards/`](cards/), in the same commit as the upload. Each card carries the per-field
accuracy table and the named failure modes from `field_report`: how often the model reads
the prior-year column, drops the reporting scale, inverts a loss in parentheses, invents a
value the filing does not report, or fails to emit parseable JSON at all. A card with only
a headline number is not a card.
