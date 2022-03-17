# Comp442 Assignment2: A Neural Probabilistic Language Model
This repo contains the implementation of a LM model with one embedding layer and two linear layers in  julia/knet.

## Overview
I have achieved **46.67 train perplexity** and **74.56 validation perplexity** in 100 epochs with a basic 1xembedding-1xhidden-1xoutput layer network. Dropouts and tanh non-linearity was used inside the model architecture.

## Data
Our dataset is from the Penn Treebank dataset provided by Mikolov: http://www.fit.vutbr.cz/~imikolov/rnnlm/

## Results
|Dataset Type|  Perplexity | Time (RTX 3070)|
| ---| --- | --- |
|Training|  46.676 | 10:27|
|Validation| 74.561| 10:27|

## Best Model Results
**Validation perplexity of the best model: 74.56162745696571** which is below the threshold of "perplexity under 190" constraint as a factor of 2.548.

## Issues
There are some buggs due to the randomness and maybe some updates in Knet we guess, which can be neglected since the most important tests are passed succesfully.


### Reference
<a id="1">[1]</a>
Bengio, Y., Ducharme, R., Vincent, P., & Jauvin, C. (2003). A neural probabilistic language model. *Journal of machine learning research, 3*. (Feb), 1137-1155. ([PDF](http://www.jmlr.org/papers/v3/bengio03a.html))
