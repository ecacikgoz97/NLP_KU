# Comp442 Assignment3: Neural Machine Translation
This repo contains the implementation of a Seq2Seq model by using RNNs in julia/knet.

## Overview
I have achieved **8.23 BLUE Score** in 10 epochs with a basic embedding-encoder-decoder-output layer network. Dropouts are applied in each input of the layer. Training from scratch took ~38 minutes with Nvidia 3060RTX. You can download pre-trained weights from [here](https://drive.google.com/drive/folders/1fukG7Vd_q2w87gZmw8hAOsf1cYCRldN3).

## Data
Our dataset is Turkish-English pair from [TED Talks Dataset](https://github.com/neulab/word-embeddings-for-nmt).

## Results
'BLEU = 8.23, 37.0/11.9/4.9/2.1 (BP=1.000, ratio=1.113, hyp_len=91854, ref_len=82502)'


## Issues
As our Course Assistants mentioned, our loss is slightly different due to different ordering of words in the vocabulary.This cause some small errors in test; however, since they are really small, they can be ignored.


### Reference
<a id="1">[1]</a>
Sutskever, Ilya, Oriol Vinyals, and Quoc V. Le. "Sequence to sequence learning with neural networks." In Advances in neural information processing systems, pp. 3104-3112. 2014. ([PDF](https://papers.nips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html))
