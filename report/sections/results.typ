#import "../template.typ": fig, wide

== RQ1: can we identify the model from its response?

Yes, and comfortably. On the controlled five-family UltraFeedback task, every classifier is far above the 0.20 chance level when it sees only the response (@tab:uf-views, "Output" columns; @fig:rq1). The required TextCNN reaches 0.736 ± 0.007 macro-F1 over three seeds and the required BiLSTM 0.725 ± 0.009. The TF-IDF baseline is only a few points behind at 0.710, the CNN-BiLSTM hybrid sits between its two parents, and the character-level CNN is the weakest neural model at 0.602. Fine-tuning DistilBERT, with sixteen times the parameters and pretraining on billions of words, buys 0.756 ± 0.003: two points over a CNN with randomly initialised embeddings that trains in 25 seconds. The tuned configurations from Section 3.4 move both required models up, the LSTM more than the CNN (0.751 ± 0.002 against 0.741 ± 0.007), and averaging the probabilities of TF-IDF, TextCNN, BiLSTM and DistilBERT gives the best number in the project, 0.781 ± 0.003, which beats DistilBERT alone on the paired test (McNemar $p = 0.007$). Classical, convolutional, recurrent and transformer models make partly different mistakes.

#wide[#figure(
  include "../tables/uf_views.typ",
  caption: [RQ1 and RQ2 on the controlled UltraFeedback set (5 families, test n = 1,994). Mean ± std over seeds 42, 1, 2; DistilBERT and TF-IDF are single runs (TF-IDF is deterministic). Chance is 0.20. Bold marks the best model per view.],
) <tab:uf-views>]

#fig("/figures/rq1_models.png", [Macro-F1 of each classifier when it sees only the response. Error bars are one standard deviation over three seeds.]) <fig:rq1>

*TextCNN beats BiLSTM at default settings, and the difference is real.* On the seed-42 test set the CNN is right on 210 responses the LSTM gets wrong, and the LSTM is right on 154 the CNN gets wrong; an exact McNemar test gives $p = 0.004$. The two models disagree on about 18% of responses, so they are not interchangeable, which is why the hybrid does not simply inherit the better of the two. The CNN also trains faster per epoch (2 s vs 3 s at batch 64) and keeps improving on the validation set for longer (@fig:train in Section 3). With the tuned settings the order flips by one point (0.751 vs 0.741); the honest summary is that the CNN is the better model out of the box and the two are equivalent once the LSTM is given larger embeddings and a wider hidden state.

*Which families are easy?* @tab:per-class and the confusion matrices in @fig:confusion tell the same story for every model: Bard (F1 0.86) and Llama-2 (0.84) are the most recognisable, GPT and Falcon are in the middle (0.72), and MPT-30B is the hardest (0.58). MPT's errors go to Falcon and GPT in roughly equal measure, and in the fine-grained experiment of Section 5 it is the least identifiable of all sixteen models; its responses are the least stylistically committed of the five. Llama-2 is rarely confused with anything: its chat tuning produces a very consistent opener ("Hello! I'm here to help you with your question") that the n-gram probe in Section 5 recovers almost verbatim.

#figure(
  include "../tables/per_class_uf.typ",
  caption: [Per-family F1 on the controlled UltraFeedback test set (seed 42, output view).],
) <tab:per-class>

#fig("/figures/confusion_uf.png", [Row-normalised confusion matrices, output view, seed 42. The MPT row is the only one with large off-diagonal mass.]) <fig:confusion>

== RQ2: does the prompt help?

*Prompt alone: no.* With the input view, in which the classifier sees only the user's prompt, every model scores 0.21–0.22 macro-F1, indistinguishable from chance, and the Char-CNN collapses to predicting one class (F1 0.07). This is what the dataset was built to make true: each family answers exactly the same number of prompts from each source, and UltraFeedback assigned models to prompts at random, so there is no information about the model in the prompt. A "source oracle" that predicts the majority family of each prompt source from the training split confirms it: 18.5% accuracy on the test set.

*Prompt plus response: slightly worse than response alone.* Concatenating the prompt in front of the response ("Both" columns) costs every neural model one to three points. This is not a bug in the data; it is capacity. The prompt occupies up to 128 of the 640 tokens with text that carries no label information, the vocabulary now has to cover prompt words, and the max-pooled features are diluted. DistilBERT, which can attend selectively, loses only 0.4 points. The practical answer to the project's question is therefore: the prompt does not help identify the model, and a classifier that is given it has to learn to ignore it.

*When input-only "works", it is the dataset talking.* The `uf_natural` variant keeps each family's natural mix of prompt sources, so Bard is over-represented on UltraChat and Evol-Instruct prompts and under-represented on Flan. Now the input-only classifiers climb to 0.23–0.24 (@tab:uf-natural), and the source oracle alone reaches 28.0% accuracy. The classifier has learned nothing about the models; it has learned which kinds of prompts each model happened to be asked. This is the mechanism behind question (iii) in the project description: in any dataset where models were not asked the same prompts, prompt-only accuracy above chance measures the sampling procedure, not the model. The modern set in Section 5, where every family answered the identical 1,200 prompts, is the cleanest version of this control.

#wide[#figure(
  include "../tables/uf_natural_views.typ",
  caption: [The same three views on `uf_natural`, where each family keeps its natural prompt-source distribution. Input-only rises above chance only because prompt source is now correlated with family (source-oracle accuracy 28.0% vs 18.5% on the controlled set).],
) <tab:uf-natural>]

#fig("/figures/rq2_views.png", [Input, output and input+output views across datasets. The dashed line is chance.]) <fig:rq2>

#include "modern_results.typ"
