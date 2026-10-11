#table(columns: 3, align: (left,right,right,),
    [*Text seen by the classifier*], [*TextCNN F1*], [*BiLSTM F1*],
    [Full text], [0.736 ± 0.007], [0.725 ± 0.009],
    [Markdown and line breaks stripped], [0.689 ± 0.002 (-0.046)], [0.679 ± 0.007 (-0.046)],
    [First 100 tokens], [0.685 ± 0.009 (-0.051)], [0.675 ± 0.009 (-0.051)],
    [First 50 tokens], [0.624 ± 0.017 (-0.112)], [0.623 ± 0.006 (-0.102)],
    [Function words, punctuation and line breaks only], [0.705 ± 0.007 (-0.031)], [0.739 ± 0.008 (+0.013)],
    [Word order shuffled], [0.639 ± 0.003 (-0.097)], [0.644 ± 0.022 (-0.082)],
)
