# Talking points — so you can defend every number

Read this with the PDF open. Each item is a claim in the report, where it comes from, and the one-sentence reason it is true.

## Data

- **Why UltraFeedback?** Public, MIT, 64k prompts × 4 models from a 17-model pool, with the model name stored per completion. It is literally `(LLM_name, LLM_input, LLM_output)`. File: `data/build.py`.
- **Why these 5 families?** GPT, Llama-2-chat, Bard, Falcon, MPT are distinct base models. WizardLM/Vicuna/UltraLM/Alpaca are Llama fine-tunes, so "which family" would be ambiguous. They come back in the 16-way experiment.
- **Why identical source quotas per family?** So the prompt carries no information about the family. Bard answered only 701 Flan prompts vs 5,456 for GPT-4; without quotas a classifier can predict the family from "this looks like a Flan prompt". The `uf_natural` set shows what happens when you don't control it (input-only rises from 0.21 to 0.24; the source-only oracle goes from 18.5% to 28%).
- **Why split by prompt?** 2–4 responses share each prompt. If they land in different splits, the model can memorise prompt content. `build.py` asserts zero overlap.
- **Modern set:** 1,200 UltraFeedback *test* prompts (so no overlap with training), each sent to 5 current families via NVIDIA's OpenAI-compatible endpoint, temperature 0.7, 450-token visible cap. Same prompts for everyone, so prompt-only classification is at chance by construction. Dropped Gemma 4 / GLM 5.3 / Llama 90B because the endpoint queued them at <1 response/min; DeepSeek never answered. `data/generate_modern.py`.

## Models

- **Tokenizer keeps case, punctuation, newlines.** Formatting *is* the fingerprint (ablations prove it). Vocab 30k, response ≤512 tokens, prompt ≤128.
- **TextCNN** = Kim 2014: embedding 128 → conv widths 3/4/5 × 128 channels → ReLU → global max-pool → dropout 0.5 → linear. 4.0M params. "A bank of phrase detectors."
- **BiLSTM** = embedding 128 → bidirectional LSTM 128/direction → concat(max-pool, mean-pool) over time → dropout → linear. 4.1M params. Pooling instead of final hidden state so a phrase at token 20 doesn't have to survive 480 recurrence steps.
- **Training:** AdamW 1e-3, wd 1e-2, batch 64, ≤15 epochs, early stop on val macro-F1 (patience 3), bf16 autocast, grad clip 1. DistilBERT: lr 2e-5, 3 epochs, batch 16, warm-up 6%. 3 seeds (42, 1, 2) for every neural config. `src/train.py`, `src/bert.py`.
- **Why macro-F1?** Classes are balanced so accuracy ≈ macro-F1, but macro-F1 is the standard for multi-class and surfaces per-class weakness (MPT).

## Hyper-parameter sweep (table in §3.4)

- Validation-only, seed 42, 16 configs. Rule fixed in advance: adopt if val F1 improves ≥ 0.01. TextCNN: emb 300 + width-2 filter (val 0.750 vs 0.733). BiLSTM: emb 300 + hidden 256 (0.763 vs 0.726). Test never used for selection. Tuned (3 seeds): CNN 0.741 ± 0.007, LSTM 0.751 ± 0.002. "At defaults the CNN wins; after tuning the LSTM catches up." `results/sweep.csv`.

## RQ1 (table 3, figure 2)

- CNN 0.736 ± 0.007, LSTM 0.725 ± 0.009, TF-IDF 0.710, hybrid 0.728, Char-CNN 0.602, DistilBERT 0.756 ± 0.003, **ensemble (TF-IDF+CNN+LSTM+BERT prob. average) 0.781 ± 0.003** — beats DistilBERT, McNemar p = 0.007. Chance 0.20.
- **CNN > LSTM is significant:** McNemar on paired test predictions, 210 vs 154 discordant, p = 0.004. `results/stats.json`.
- **Why is DistilBERT barely better?** The signal is local phrases; max-pooled 3–5-gram filters are the right bias. Pretraining buys meaning; the fingerprint isn't meaning.
- **Easiest/hardest families:** Bard 0.86, Llama-2 0.84 (consistent opener "Hello! I'm here to help"), GPT/Falcon 0.72, MPT 0.58 (least stylistically committed).

## RQ2 (tables 3, 5; figure 4)

- **Input-only ≈ chance (0.21–0.22)** on the controlled set. By design.
- **Input+output is 1–3 points *worse* than output-only.** The prompt uses up to 128 of 640 tokens with zero label information and dilutes the pooled features. DistilBERT (attention) loses only 0.4.
- **"If input-only works surprisingly well, why?"** Because the dataset's prompt→model assignment is skewed, not because prompts reveal models. Shown by `uf_natural` + the source oracle (28%).
- Modern set: input-only exactly chance by construction (same prompts for all families).

## RQ3 (tables 6, 7; figures 5, 6)

- **Leave-one-source-out:** train on 5 sources, test on the 6th. Small losses (2–5 points) for chat-like sources; **Flan costs 18 points** (0.736 → 0.557) because Flan prompts demand one-word answers and there is no room for style. Fingerprint = style; style needs room.
- **Learning curve:** 800 examples → CNN 0.567; 4,000 → 0.701. CNN is more data-efficient than LSTM (7 points at 5%).
- **Cross-generation:** the 2023 TextCNN applied to all 5,590 modern responses labels only **1.9% of Llama 3.2 as Llama-2**; modern models get sorted into GPT/Bard (the markdown-heavy 2023 families): Nemotron → GPT 74%, DiffusionGemma → Bard 66%. Mean confidence 0.80 while being wrong. Takeaway: fingerprint = post-training recipe of a release, not lineage. `results/crossgen.json`.
- **Modern LOSO:** held-out Flan 0.42 (CNN) vs 0.85 in-domain; every other held-out source ≥ 0.85. One-line answers have no formatting to fingerprint.

## Modern set (table 7, figures 5, 6, 9)

- 1,118 prompts × 5 families = 5,590 rows (4,470/560/560). Complete block: every prompt answered by all five.
- TextCNN 0.852 ± 0.007, BiLSTM 0.819 ± 0.013, tuned 0.848 / 0.840, DistilBERT 0.838, ensemble 0.860 ± 0.008. Chance 0.20. CNN > LSTM McNemar p = 0.04; ensemble vs CNN not significant (p = 0.29).
- Input-only accuracy exactly 0.200 (by construction; macro-F1 lower because models collapse to few classes).
- Why easier than 2023: formatting intensity. gpt-oss median 500 words / 40 line breaks / 13 bold spans; Llama 3.2 uses headers in 7% of responses vs 47–69% for others. Stylometry-only GBM 71.4%.
- **Typographic artifacts (table 8):** 79% of gpt-oss responses contain U+2011 non-breaking hyphen; em dash in 72% Nemotron / 57% Kimi / 0.1% Llama; 17% of Kimi responses contain a leaked `<|close|>` token (endpoint defect, kept and measured). Ablation `strip_special` (remove template tokens, ASCII-fold dashes/quotes): CNN −0.083, LSTM −0.074. "A tenth of the accuracy is punctuation; train attributors on ASCII-folded text."
- Dropped families and why: Gemma 4 31B, GLM 5.3, Llama 3.2 90B queued at <1 response/min on the endpoint; DeepSeek v4.1 never returned. Documented in §2.2.

## RQ4 (tables 8; figures 7, 8, 9)

- **Stylometry:** 26 hand features. GBM on features alone = 64.1% accuracy. Top by ANOVA: type-token ratio, exclamation marks, length, ends-with-question, blank lines, bold.
- **Ablations (retrain on transformed text):** strip markdown −4.6; first 100 tokens −5; first 50 −10/−11; shuffle word order −8/−10 (but still 0.64, so bag-of-words carries most of it); **function words only: LSTM +0.013, CNN −0.031**. Content words are unnecessary for the LSTM and mildly distracting.
- **CNN filter probing:** for each filter, the test n-gram that maximises it, weighted by FC weight. Llama-2 → "I'd be happy to help", Falcon → "As an AI language model", GPT → code fences and numbered lists, Bard → "\n\nHere", "However,", bold, MPT → ". \n 2".
- **LSTM saliency** = gradient × input on the embeddings, one correct example per family. Matches the n-grams.
- **Paraphrase attack:** 300 UltraFeedback test responses (60/family) rewritten by **Llama 3.2** ("keep every fact, change wording/structure/formatting freely"); every saved classifier scored before/after. Nemotron-3 was the first choice but its endpoint returned 429 all evening. Numbers: see `results/paraphrase_uf.json` and §5 of the report.

## Fine-grained (table 9, figure 10)

- 16 models, 1,500 each, chance 0.0625. CNN 0.346, DistilBERT 0.334, LSTM 0.33, TF-IDF 0.304.
- Llama-2-7B is the single most identifiable model (F1 0.80); 7B/13B/70B rarely confused (70B → 13B only 8%). GPT-3.5 vs GPT-4 nearly inseparable (a third of 3.5 labelled 4). MPT-30B last (0.10).
- Takeaway: post-training recipe changes the fingerprint more than scale does.

## Calibration

- ECE: TF-IDF 0.04, CNN 0.08, DistilBERT 0.08, LSTM 0.16. LSTM confidences are not probabilities.

## If asked "what did Claude do?"

Claude Code was used as a pair programmer: scaffolding scripts, debugging, and drafting prose from the result tables. Every experimental decision (dataset design, family choice, controlled quotas, ablation set, modern-set protocol) was mine, I ran and inspected every result, and I can explain every line of `src/` and every number in the PDF. The course allows LLM assistance; the responsibility for understanding is mine, which is why this document exists.
