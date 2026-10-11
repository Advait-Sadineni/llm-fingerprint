#import "../template.typ": fig, wide

*Paraphrase attack.* If the fingerprint is style, rewriting the style should remove it while leaving the content. I sampled 300 UltraFeedback test responses (60 per family, under 2,500 characters) and had Llama 3.2 rewrite each one with the instruction "keep every fact, instruction and the overall meaning, but change the wording, sentence structure and formatting freely". The rewrites are genuine paraphrases: median character-level similarity to the original is 0.15 and they run 19% longer. Every saved classifier was then scored on the originals and on the rewrites (@tab:paraphrase, @fig:paraphrase). Accuracy falls from 0.60–0.72 to 0.28–0.37, within ten points of chance; the BiLSTM, which relies most on surface form, loses the most (−0.43), the TF-IDF and character models, which keep some topical signal, the least (−0.30). Where does the rewritten text go? Overwhelmingly to GPT (45–62% of predictions for the neural models) and never to Llama-2 (6–11%), which is the cross-generation result of Section 5.3 seen from the other side: a 2026 model's prose reads to a 2023 classifier like GPT. Two conclusions follow. The fingerprint is almost entirely a property of surface realisation, so attribution of paraphrased or edited text is close to hopeless for models of this kind; and the small residue that survives (0.28–0.37 against 0.20) is the content-level signal that the function-word ablation showed the models barely use when style is available.

#wide[#figure(
  include "../tables/paraphrase.typ",
  caption: [Accuracy on 300 UltraFeedback test responses before and after paraphrasing by Llama 3.2 (chance 0.20). The last column is the family most often predicted for the rewritten text.],
) <tab:paraphrase>]

#fig("/figures/rq4_paraphrase.png", [Accuracy before and after the paraphrase attack. Hatched bars are the rewritten responses.]) <fig:paraphrase>
