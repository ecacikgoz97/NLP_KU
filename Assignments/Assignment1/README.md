# Comp442 Assignment1: Large Movie Review Dataset
## Prepared by Emre Can Acikgoz
This repo contatins the implementation of Naive Bayes algorithm for Sentiment analysis with Julia.

# Overview
I have achieved 88.164% train set accuracy and 87.884% test set accuracy by using a Naive Bayes approach which basically predicts the class labels by doing a bag-of-words assumption. For pre-processing; I lowercased it, ignore the unncesessary punctuations, split our sentences to words, and convert our words to IDs. Other techniques couldn't help me to achieve higher results.

## Data
Train set contains 25,000 reviews with 12,500 postive and 12,500 negative classes. 
Test set contains 25,000 reviews with 12,500 postive and 12,500 negative classes.
Total 50,000 reviews with 25,000 are in postive class and 25,000 are in negative class.

## Results
|Train Set Accuracy|  88.164% | 0.916919 seconds|
|Test Set Accuracy| 87.884%| 0.568173 seconds|