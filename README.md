# SPEAKER-EMBED-AUGM

Optimizing speaker embeddings via Population Based Augmentation.  

Underlying DNN is ECAPA-TDNN (extracted from SpeechBrain) with additive angular margin, softmax and cross-entropy loss.  

## Loss change

Testing loss change on validation - using 20 000 utterances from Voxceleb2

10 epochs

0 - 18.881734848022460
1 - 21.204872131347656
2 - 21.497272491455078
3 - 21.376222610473633
4 - 21.291334152221680
5 - 21.156986236572266
6 - 21.088050842285156
7 - 20.899959564208984
8 - 20.833074569702150
9 - 20.754510879516600

