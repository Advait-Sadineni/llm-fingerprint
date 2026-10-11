#import "template.typ": paper, fig, wide

#show: paper.with(
  title: [Who Wrote It? Identifying Large Language Models from Their Responses],
  author: [Advait Sadineni (team leader and sole member)],
  affiliation: [CMPSC 448, The Pennsylvania State University],
  date: [October 11, 2026 · code and data: #link("https://github.com/Advait-Sadineni/llm-fingerprint")[github.com/Advait-Sadineni/llm-fingerprint]],
  abstract: [
    Can a classifier tell which large language model wrote a response? I treat LLM attribution as supervised text classification and study it on two datasets: 20,000 responses from five 2023 model families drawn from UltraFeedback with an identical prompt mix per family, and a new set I generated in which 1,118 prompts were each answered by five 2026 families (Nemotron 3, gpt-oss, Kimi K3, Llama 3.2, DiffusionGemma). The two required models, a TextCNN and a BiLSTM, reach 0.736 and 0.725 macro-F1 on the 2023 families (chance 0.20) and 0.852 and 0.819 on the 2026 families; a fine-tuned DistilBERT adds under two points, and an ensemble reaches 0.781 and 0.860. The user's prompt carries no information about the model once prompt assignment is controlled, and prompt-only accuracy above chance in uncontrolled data is shown to be an artefact of how prompts were distributed. Fingerprints transfer across task domains except to one-word-answer tasks, need only a few hundred examples per model, and do not survive a model generation: a classifier trained on Llama-2 labels 2% of Llama 3.2 responses as Llama. Ablations, filter probing and saliency show the fingerprint is style rather than content: keeping only function words, punctuation and line breaks does not hurt the BiLSTM at all, and on 2026 models a tenth of the accuracy comes from typographic choices such as non-breaking hyphens and em dashes. A paraphrase attack quantifies how much of the signal survives rewriting. All code, data and a browser demo are public.
  ],
)

= Introduction and problem definition

Large language models now write a large share of the text people read, and different models are trained on different data with different post-training recipes. If those differences leave a measurable trace in the text, then LLM output is attributable: given a response, a classifier can say which model family produced it. This project treats attribution as supervised text classification and asks four questions.

- *RQ1.* Given only a response, can we identify the model family that generated it?
- *RQ2.* Does the user's prompt help, and if prompt-only classification works, why?
- *RQ3.* Do the learned fingerprints generalise across task domains, across model generations, and how much data do they need?
- *RQ4.* What are the fingerprints made of: content, style, or formatting?

The required models are a convolutional network (TextCNN) and a recurrent network (BiLSTM). I add a classical TF-IDF baseline below them and a fine-tuned DistilBERT above them, plus two hybrids, so every number has context. Every headline result is the mean of three random seeds, every split is grouped by prompt so no prompt leaks between training and test, and every table cell in this report is produced by one command in the repository.

= Data

== Dataset A: UltraFeedback (2023 models)

UltraFeedback @ultrafeedback is a public, MIT-licensed preference dataset in which 63,967 instructions from six sources were each answered by four models sampled from a pool of seventeen. Because it stores the model name next to every completion, it is exactly the `(LLM_name, LLM_input, LLM_output)` triple the project asks for, and because each prompt was answered by several models it supports controlled comparisons. I flattened the 256k completions, dropped empty, very short (under 20 characters) and exact-duplicate responses, and kept five clearly distinct model families: *GPT* (gpt-4, gpt-3.5-turbo), *Llama-2-chat* (7B, 13B, 70B), *Bard*, *Falcon-40B-instruct* and *MPT-30B-chat*. Llama fine-tunes in the pool (WizardLM, Vicuna, UltraLM, Alpaca) share Llama's base weights and were excluded from the family task; they return in the 16-way fine-grained experiment.

*Controlled source mix.* The six prompt sources (Flan, ShareGPT, Evol-Instruct, UltraChat, TruthfulQA, FalseQA) differ in style and length, and the families are not equally represented in each source (Bard answered only 701 Flan prompts, GPT-4 answered 5,456). A classifier could therefore learn "this prompt looks like Flan, Flan prompts are rarely Bard" instead of a model fingerprint. To rule this out, the main dataset (`uf`) samples *exactly the same number of responses per source for every family*: 1,000 Evol-Instruct, 1,000 ShareGPT, 800 UltraChat, 600 Flan, 450 FalseQA and 150 TruthfulQA, for 4,000 responses per family and 20,000 in total. A second variant (`uf_natural`) draws 4,000 per family with each family's natural source mix, and is used only to show what happens when the confound is left in (RQ2).

*Grouped split.* Prompts, not responses, are assigned to train, validation and test (80/10/10), so the 2–4 responses to one prompt always land in the same split. Without this, a model could memorise prompt-specific content. The split script asserts zero prompt overlap and the per-source balance of every split.

*Confound worth stating.* UltraFeedback generated each completion under a randomly chosen system prompt (a "principle" such as honesty or helpfulness). This adds within-family variance; it does not favour any family.

== Dataset B: a modern set I generated (2026 models)

The 2023 pool does not contain today's models, and in UltraFeedback a given prompt was answered by only four of the seventeen models. I therefore built a second dataset with a complete block design: the *same* 1,200 prompts, sampled from the UltraFeedback *test* prompts with the same source quotas, were each sent to five current model families through NVIDIA's OpenAI-compatible endpoint (`integrate.api.nvidia.com`): *Nemotron-3* (nvidia/nemotron-3-super-120b-a12b), *gpt-oss* (openai/gpt-oss-20b), *Kimi K3* (moonshotai/kimi-k3), *Llama 3.2* (meta/llama-3.2-11b-vision-instruct) and *DiffusionGemma* (google/diffusiongemma-26b-a4b-it, a non-autoregressive diffusion language model). Sampling used temperature 0.7 and a 450-token visible-answer budget (reasoning models received a larger budget for hidden reasoning, which was discarded); the classifiers read at most 512 tokens anyway, and truncation applies equally to every family. Generation took about eight hours of wall-clock time at the endpoint's 40 requests-per-minute free-tier limit, most of it spent discovering which models the endpoint would actually serve at a usable rate. Three further families (Gemma 4 31B, GLM 5.3, Llama 3.2 90B) were started and abandoned because the endpoint queued them at under one response per minute; DeepSeek v4.1 never returned within four minutes. Only prompts answered by every family are kept, so prompt-only classification is at chance by construction, which is the clean control RQ2 needs. The prompts are public; nothing private was sent to the API.

#wide[#figure(
  table(columns: 4, align: (left, right, right, left),
    [*Dataset*], [*Classes*], [*Rows (train / val / test)*], [*Purpose*],
    [`uf`], [5 families], [20,000 (16,015 / 1,991 / 1,994)], [main; identical source mix per family],
    [`uf_natural`], [5 families], [20,000 (15,983 / 2,005 / 2,012)], [natural source mix; RQ2 confound probe],
    [`uf17`], [16 models], [24,000 (19,219 / 2,375 / 2,406)], [fine-grained: can we separate Llama-2-7B from 70B?],
    [`modern`], [5 families], [5,590 (4,470 / 560 / 560)], [2026 models; 1,118 prompts, every one answered by all five families],
  ),
  caption: [The four datasets. All splits are grouped by prompt; `uf` and `uf_natural` draw from disjoint random samples of the same pool.],
)]

= Models and training

== Text representation

The tokenizer is a regular expression that keeps *case*, *punctuation* and *newlines* as tokens. This is deliberate: a lower-casing, punctuation-stripping tokenizer would erase exactly the signals (markdown bullets, bold markers, capitalised openers, line breaks) that RQ4 shows the models rely on. The vocabulary is the 30,000 most frequent training tokens with frequency at least two; the rest map to `<unk>`. Responses are truncated to 512 tokens, prompts to 128, and the input+output view concatenates them around a `<sep>` token (640 tokens). The Char-CNN reads the first 1,024 characters with a 97-symbol alphabet instead.

== The six classifiers

#figure(
  table(columns: 3, align: (left, left, right),
    [*Model*], [*Architecture*], [*Params*],
    [TF-IDF + LR], [word 1–2-grams and character 3–5-grams (200k features each, sublinear tf, no lower-casing) → multinomial logistic regression, $C=5$], [—],
    [TextCNN (required)], [embedding 128 → three parallel 1-D convolutions (widths 3, 4, 5; 128 channels each) → ReLU → global max-pool → dropout 0.5 → linear @kim2014], [4.04 M],
    [BiLSTM (required)], [embedding 128 → bidirectional LSTM (128 hidden per direction) → concatenation of masked max-pool and mean-pool → dropout 0.5 → linear], [4.10 M],
    [CNN-BiLSTM], [embedding 128 → width-3 convolution (128 ch) → BiLSTM (128) → max-pool → linear; the "combine models" variant], [4.15 M],
    [Char-CNN], [character embedding 32 → six 1-D convolutions (256 ch, widths 7,7,3,3,3,3 with three max-pools of 3) → 512-unit dense → linear @zhang2015], [5.77 M],
    [DistilBERT], [`distilbert-base-cased` fine-tuned end to end, 512-token input, linear head; cased so capitalisation survives], [66 M],
  ),
  caption: [The model ladder. The two required models sit in the middle; the baseline below and the transformer above give them context.],
)

== Training details

All neural models are trained in PyTorch 2.11 on one RTX 5080 laptop GPU with bf16 autocast. TextCNN, BiLSTM, CNN-BiLSTM and Char-CNN use AdamW (learning rate $10^(-3)$, weight decay $10^(-2)$), batch size 64, gradient clipping at 1.0, cross-entropy loss, at most 15 epochs with early stopping on validation macro-F1 (patience 3) and the best-validation checkpoint restored. DistilBERT uses AdamW at $2 dot 10^(-5)$ with 6% linear warm-up then linear decay, batch 16, three epochs. Embeddings are learned from scratch (no pretrained vectors), which keeps the required models honest about what they learn from 16k examples. One TextCNN run takes about 25 s, one BiLSTM run about 45 s, DistilBERT about 20 minutes (sharing the GPU with other runs). Every configuration in Sections 4 and 5 was run with seeds 42, 1 and 2; tables report mean ± standard deviation. The metric is macro-F1 (classes are balanced, so accuracy is nearly identical and is reported alongside).

== Hyper-parameter selection (validation only)

The defaults above are the textbook settings. To check they are not leaving accuracy on the table, I ran a small grid for each required model on the *validation* split of `uf` (seed 42, output view): TextCNN over embedding size {128, 300} × channels {128, 256} × filter widths {(3,4,5), (2,3,4,5)}, and BiLSTM over embedding {128, 300} × hidden {128, 256} × layers {1, 2}, sixteen runs in all (@tab:sweep). The rule, fixed before the sweep ran, was: adopt a configuration as "tuned" only if its validation macro-F1 beats the default by at least 0.01; the test set is never used for selection. Both models cleared the bar. The tuned TextCNN uses 300-dimensional embeddings and adds a width-2 filter (validation 0.750 vs 0.733); the tuned BiLSTM uses 300-dimensional embeddings and 256 hidden units per direction (0.763 vs 0.726). Larger embeddings help both; a second LSTM layer and more CNN channels do not. The tuned models were then retrained with three seeds and appear as separate rows in the result tables, so the default and tuned numbers can be compared directly.

#figure(
  include "tables/sweep.typ",
  caption: [Validation macro-F1 for every configuration in the sweep (seed 42). Defaults: TextCNN emb 128 / 128 channels / widths 3,4,5; BiLSTM emb 128 / hidden 128 / 1 layer.],
) <tab:sweep>

#fig("/figures/training_curves.png", [Training loss and validation macro-F1 per epoch (seed 42, output view, `uf`). TextCNN keeps improving on validation long after the LSTM plateaus; both overfit the training set within ten epochs.]) <fig:train>

= Results: RQ1 and RQ2

#include "sections/results.typ"

= In-depth analyses

#include "sections/analyses.typ"

= Lessons and experience

#include "sections/lessons.typ"

= Limitations

#include "sections/limitations.typ"

= Appendix

#wide[#figure(
  include "tables/artifacts_uf.typ",
  caption: [Share of UltraFeedback responses containing each formatting or typographic pattern, by family.],
) <tab:artifacts-uf>]

#wide[#fig("/figures/rq4_saliency_modern.png", [BiLSTM gradient × input saliency on one correctly classified modern test response per family. Markdown markers, colons and list structure dominate, matching the ablation in @tab:strip-special.]) <fig:saliency-modern>]

*Reproducibility.* Every number in this report is produced by `python -m src.aggregate && python -m src.tables` from `results/*.json`, and every figure by `python -m src.plots`; `bash experiments.sh all` and `bash modern_stage.sh` regenerate the results from the committed datasets on one GPU in about three hours. `python -m src.check_report --final` verifies that the PDF, figures and result files agree before every build.

#bibliography("refs.bib", style: "ieee")
