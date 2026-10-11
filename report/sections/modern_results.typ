#import "../template.typ": fig, wide

== The same questions on 2026 models

The modern set repeats RQ1 and RQ2 on five current families with the cleanest possible design: every one of the 1,118 prompts was answered by all five models. The answers are sharper on both questions (@tab:modern-views, @fig:confusion-modern).

*RQ1.* Today's models are *easier* to tell apart than the 2023 ones. The TextCNN reaches 0.852 ± 0.007 macro-F1 (five-way, chance 0.20) on 4,470 training responses, a quarter of the UltraFeedback training set, and the BiLSTM 0.819 ± 0.013; tuning narrows the gap again (0.848 vs 0.840). DistilBERT scores 0.838, below the CNN, and the four-model ensemble 0.860 ± 0.008. gpt-oss is the most recognisable family (per-class F1 0.93), DiffusionGemma the least (0.83); the main confusions run between Llama 3.2 and the two NVIDIA-hosted reasoning models. The CNN again beats the LSTM at default settings (McNemar $p = 0.04$), and the difference between the ensemble and the CNN alone is not significant ($p = 0.29$): on this data one well-chosen small model is essentially as good as everything combined.

*RQ2.* With identical prompts for every family the input-only view is at chance by construction and the classifiers confirm it: accuracy exactly 0.200, macro-F1 between 0.07 and 0.16 as the models collapse to one or two classes. Adding the prompt to the response costs 2–3 points for the CNN and LSTM, as on UltraFeedback. The combination of this result with @tab:uf-natural settles the project's question (iii): prompt-only accuracy above chance is a property of how a dataset assigned prompts to models, never of the models.

#wide[#figure(
  include "../tables/modern_views.typ",
  caption: [RQ1 and RQ2 on the modern set (5 families, test n = 560; chance 0.20). Mean ± std over three seeds; DistilBERT and TF-IDF single runs. Input-only accuracy is exactly 0.200 for every model because every prompt was answered by all five families.],
) <tab:modern-views>]

#fig("/figures/confusion_modern.png", [Row-normalised confusion matrices on the modern test set, seed 42.]) <fig:confusion-modern>

*Why are the 2026 models easier?* The stylometry in @fig:stylometry-modern says most of it: the modern families differ enormously in *how much* they write and *how they format it*. gpt-oss writes a median of 500 words with 40 line breaks and thirteen bold spans per response; DiffusionGemma, Llama 3.2 and Nemotron-3 write about 250 words; Llama 3.2 almost never uses markdown headers (7% of responses, against 47–69% for the others) and is the only modern family that still hedges ("It's important to note", 0.32 per response). A gradient-boosted classifier on the 26 stylometric features alone reaches 71.4% accuracy on the modern set, against 64.1% on UltraFeedback.

#wide[#fig("/figures/rq4_stylometry_modern.png", [Stylometric profile of the five 2026 families. Formatting intensity, not vocabulary, separates them.], width: 85%) <fig:stylometry-modern>]

*Typographic fingerprints, and a data-quality caveat.* Probing the modern CNN's filters (the method of Section 5.3) surfaces signals that are real but almost embarrassingly shallow (@tab:artifacts-modern). 79% of gpt-oss responses contain a Unicode *non-breaking hyphen* (U+2011, as in "step‑by‑step"), which no other family emits; 72% of Nemotron-3 and 57% of Kimi K3 responses contain an em dash, against 0.1% for Llama 3.2; Nemotron-3 and gpt-oss separate paragraphs with `---` rules half the time; and 17% of Kimi K3 responses contain a leaked chat-template token such as `<|close|>`, usually at the head of a degenerate tail of text, which is a defect of that endpoint's serving stack rather than a property of the model. These are exactly the "signals the classifier appears to use" that RQ4 asks about, so I kept them in the data and measured their contribution instead: retraining with template tokens removed and all typographic dashes and quotes folded to ASCII (@tab:strip-special) costs the TextCNN 8.3 points (0.852 → 0.769 ± 0.018) and the BiLSTM 7.4 (0.819 → 0.744 ± 0.015). Even without them both required models stay far above chance and above their UltraFeedback scores, so the modern fingerprint is not *only* punctuation; but nearly a tenth of the accuracy on 2026 models comes from characters a copy-editor would normalise away. Any deployed attributor should be trained on ASCII-folded text, or it will be defeated by a find-and-replace.

#wide[#figure(
  include "../tables/artifacts_modern.typ",
  caption: [Share of modern responses containing each formatting or typographic pattern. The 2023 equivalents are in the appendix (@tab:artifacts-uf).],
) <tab:artifacts-modern>]

#figure(
  include "../tables/modern_strip_special.typ",
  caption: [Modern set, output view, three seeds: effect of removing leaked template tokens and ASCII-folding typographic punctuation before training.],
) <tab:strip-special>
