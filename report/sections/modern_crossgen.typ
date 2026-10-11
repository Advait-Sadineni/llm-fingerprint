#import "../template.typ": fig, wide

== RQ3c: do fingerprints survive a model generation?

If a fingerprint belonged to a lineage, a classifier that learned Llama-2-chat in 2023 should recognise Llama 3.2 in 2026. It does not (@fig:crossgen). Applied to the 5,590 modern responses, the UltraFeedback TextCNN labels only 1.9% of Llama 3.2's responses as Llama-2; it calls them GPT (39%), Bard (32%) or MPT (23%) instead. The other modern families are sorted almost entirely into the two 2023 families that used markdown most, GPT and Bard: Nemotron-3 is "GPT" 74% of the time, DiffusionGemma "Bard" 66%. The classifier is confident while doing this (mean max-probability 0.80), which is the usual failure mode of a discriminative model under distribution shift.

Together with the fine-grained result (Llama-2 sizes separable, GPT-3.5 and GPT-4 not) this says what a fingerprint is: not the base model's weights or lineage, but the post-training recipe of a particular release. Llama-2-chat's "Hello! I'm here to help" is gone from Llama 3.2; what remains of Llama 3.2's identity in 2026 is that it formats less than its contemporaries. Attribution systems therefore need to be re-enrolled for every model release, which the learning curve in @tab:curve suggests is cheap: a few hundred examples per model.

#fig("/figures/rq3_crossgen.png", [Where the 2023 classifier puts 2026 responses. Rows are actual modern families, columns the predicted UltraFeedback family (row-normalised).]) <fig:crossgen>
