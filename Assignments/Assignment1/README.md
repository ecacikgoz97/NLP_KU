# Comp442 Assignment1: Large Movie Review Dataset
This repo contains the implementation of Naive Bayes algorithm for Sentiment Analysis by using [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) with Julia.

## Overview
I have achieved 88.164% train set accuracy and 87.884% test set accuracy by using a Naive Bayes approach which basically predicts the class labels by doing a bag-of-words assumption. For pre-processing; I lowercased it, ignore the unncesessary punctuations, split our sentences to words, and convert our words to IDs. Other techniques couldn't help me to achieve higher results.

## Data
Train set contains 25,000 reviews with 12,500 postive and 12,500 negative classes. <br />
Test set contains 25,000 reviews with 12,500 postive and 12,500 negative classes. <br />
Total 50,000 reviews with 25,000 of them in postive class and 25,000 of them in negative class. <br />

## Results
|Dataset Type|  Accuracy | Time|
| ---| --- | --- |
|Training Set|  88.164% | 0.916919 seconds|
|Test Set| 87.884%| 0.568173 seconds|


### Reference
<a id="1">[1]</a>
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
