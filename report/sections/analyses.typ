#import "../template.typ": fig, wide

== RQ3a: do fingerprints transfer across task domains?

Each classifier was retrained six times on five prompt sources and tested on every response from the sixth, which it had never seen (@tab:loso, @fig:loso). Fingerprints transfer, with a cost that depends on how different the held-out domain is. Holding out ShareGPT, UltraChat or Evol-Instruct (open-ended chat and instruction following, the bulk of the data) costs the CNN two to three points. Holding out TruthfulQA costs five. Holding out Flan, whose prompts are short, templated NLP tasks that often demand a one-word answer, costs eighteen points (0.736 → 0.557): when a model is asked to output "yes" or a single entity there is little room for style, and the stylistic cues the classifier learned elsewhere (openers, list formatting, sign-offs) are absent. Falcon and MPT suffer most on held-out Flan; Llama-2 and Bard remain identifiable everywhere, consistent with their per-class F1 in-domain. So the classifier is learning something closer to a general fingerprint than to task-specific patterns, but the fingerprint lives in the parts of a response that short-answer tasks suppress.

#wide[#figure(
  include "../tables/loso_uf.typ",
  caption: [Leave-one-source-out macro-F1 on UltraFeedback (seed 42). Each column is a classifier trained on the other five sources and tested on all responses from the named source.],
) <tab:loso>]

The modern set tells the same story more sharply (@tab:loso-modern, right panel of @fig:loso). Held-out ShareGPT, UltraChat, TruthfulQA and FalseQA are classified *at least as well* as in-domain (0.86–0.92 for the CNN against 0.85 in-domain, partly because the held-out test is larger and the in-domain test contains Flan), while held-out Flan drops to 0.42 for the CNN and 0.35 for the LSTM. The 2026 fingerprint is formatting-heavy, and a one-line answer has no formatting.

#wide[#figure(
  include "../tables/loso_modern.typ",
  caption: [Leave-one-source-out macro-F1 on the modern set (seed 42).],
) <tab:loso-modern>]

#wide[#fig("/figures/rq3_loso.png", [Held-out-source macro-F1 on both datasets. Flan is the hard domain for every model.], width: 90%) <fig:loso>]

== RQ3b: how much data does a fingerprint need?

#figure(
  include "../tables/curve.typ",
  caption: [Learning curve, output view, mean ± std over three seeds. The full training set is 16,015 responses.],
) <tab:curve>

With only 800 training responses (160 per family) the CNN already reaches 0.567 and the LSTM 0.492; with 4,000 the CNN is within four points of its final score (@tab:curve, @fig:curves). The CNN is consistently more data-efficient than the LSTM, by seven points at 5% of the data, which fits the ablation findings below: the signal is in short local patterns that a few hundred examples suffice to reveal, and an LSTM has to learn to carry them through long sequences before it can use them.

#fig("/figures/rq3_learning_curve.png", [Test macro-F1 against training-set size (log scale). Shaded bands are one standard deviation over seeds.]) <fig:curves>

#include "modern_crossgen.typ"

== RQ4: what is the fingerprint made of?

*Stylometry.* Twenty-six hand-computed features per response (length, sentence statistics, type-token ratio, line breaks, markdown markers, hedges, punctuation) differ sharply across families (@fig:stylometry). Falcon writes the shortest responses (median 98 words against 196–262 for the others) with the highest lexical diversity, because short texts repeat fewer words. Bard uses bullet lists 2.5× as often as anyone else and bold markup 8× as often. Llama-2 hedges and apologises twice as often as the other families ("I cannot", "As an AI", "It's important to note"). A gradient-boosted classifier trained on these 26 numbers alone, with no access to the words, reaches 64.1% accuracy: two thirds of the way from chance to the CNN on a feature vector a human can read. The strongest single features by ANOVA F are type-token ratio, exclamation marks, length, whether the response ends in a question, blank lines and bold markup. Length and formatting matter more than vocabulary.

#wide[#fig("/figures/rq4_stylometry_uf.png", [Six of the 26 stylometric features by family on the controlled UltraFeedback set. Boxes show quartiles; bars are means.], width: 85%) <fig:stylometry>]

*Ablations.* The most direct way to ask what a classifier uses is to take a signal away and retrain (@tab:ablation, @fig:ablation). Stripping markdown and line breaks costs both required models 4.6 points. Keeping only the first 100 tokens costs 5; the first 50, 10–11. Shuffling the words, which destroys word order while keeping the bag of words, costs 8–10 points but leaves 0.64, so a large share of the fingerprint survives as a bag of tokens. The surprising row is the fourth: replacing every content word with a placeholder and keeping only function words, punctuation and line breaks *does not hurt the LSTM at all* (0.739 vs 0.725, +0.013) and costs the CNN three points. Everything the models need to know is in how a response is put together: its articles, pronouns, conjunctions, commas, list markers and paragraph breaks. Topic words are not merely unnecessary; for the LSTM they are a distraction.

#wide[#figure(
  include "../tables/ablation.typ",
  caption: [Ablations on the output view, mean ± std over three seeds, with the change from full text in parentheses. Each row retrains the model on transformed text.],
) <tab:ablation>]

#fig("/figures/rq4_ablation.png", [Change in macro-F1 when a signal is removed.]) <fig:ablation>

*What the CNN filters fire on.* For every convolutional filter I recorded the n-gram in the test set that activates it most, then weighted filters by their output-layer weight for each class. The top n-grams per family are not subtle. Llama-2: "I'd be happy to help", "I'm here to help", "Hello! I'm here", "Sure, I'd be happy". Falcon: "As an AI language model", "an AI assistant", "anything else I can", "Here are some". GPT: code fences (three backticks preceded by two line breaks), numbered-list continuations (". ⏎⏎ 4."), "I'm happy to help". Bard: paragraph breaks followed by "Here", "The", "However,", "I hope", and bold markers. MPT: numbered lists with single line breaks (". ⏎ 2") and "For example,". The CNN has learned each family's house style: how it opens, how it hedges, and how it lays out a list.

*What the LSTM looks at.* Gradient × input saliency on one correctly classified example per family (@fig:saliency) agrees. For Llama-2 the mass sits on "Hello", "!", "I'm here to help", "question". For Falcon it sits on the sign-off "help you with." and "Let me know". For Bard the opener "Sure, here is an example" and the table line breaks light up. For GPT and MPT, which have no catchphrase, saliency is spread across ordinary content words and sentence boundaries, which is exactly where their lower per-class F1 comes from.

#wide[#fig("/figures/rq4_saliency_uf.png", [BiLSTM gradient × input saliency on one correctly classified test response per family (first 70 tokens). Darker tokens contributed more to the predicted class.]) <fig:saliency>]

#include "paraphrase.typ"

== Fine-grained: sixteen models, not five families

Collapsing to families hides a harder question: can the classifier tell Llama-2-7B from Llama-2-70B, or GPT-3.5 from GPT-4? On a balanced 16-way task (1,500 responses per model, Pythia-12B dropped for lack of data) the TextCNN reaches 0.356 ± 0.009 macro-F1 over three seeds against a chance level of 0.0625, the BiLSTM 0.345 ± 0.014, and DistilBERT does no better than either (0.334; @tab:uf17, @fig:uf17). The seed-42 confusion matrix is informative. Llama-2-7B is the single most identifiable model of all (F1 0.80), ahead of 13B (0.67) and 70B (0.55), and the three sizes are rarely confused with each other (70B is mistaken for 13B only 8% of the time): the smaller model's style is a more exaggerated version of the family's. GPT-3.5-turbo and GPT-4, by contrast, are nearly inseparable: a third of GPT-3.5 responses are labelled GPT-4, and GPT-3.5 has the second-lowest F1 (0.14). Among the Llama fine-tunes, WizardLM-13B and -70B blur into each other and into UltraLM, and MPT-30B is again last (0.10). Model scale, then, changes a fingerprint less than the post-training recipe does, and OpenAI's two models share a recipe closely enough that style cannot separate them.

#figure(
  include "../tables/uf17.typ",
  caption: [16-way model identification on UltraFeedback (`uf17`, test n = 2,406, chance 0.0625). TextCNN and BiLSTM are means over three seeds; TF-IDF and DistilBERT single runs.],
) <tab:uf17>

#fig("/figures/confusion_uf17.png", [Row-normalised 16-way confusion matrix for the TextCNN. The three Llama-2 sizes form clean diagonal blocks; GPT-3.5 and GPT-4 merge; the Llama fine-tunes (WizardLM, UltraLM, Vicuna) overlap.]) <fig:uf17>

== Calibration

Expected calibration error on the five-family test set is 0.04 for TF-IDF, 0.08 for the CNN and DistilBERT, and 0.16 for the BiLSTM, whose max-pooled features drive it to confident predictions early in training. The LSTM's confidence should not be read as probability; the CNN's is usable with a small temperature correction.
