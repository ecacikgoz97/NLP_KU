#jl Use `Literate.notebook(juliafile, ".", execute=false)` to convert to notebook.

# # Attention-based Neural Machine Translation
#
# **Reference:** Luong, Thang, Hieu Pham and Christopher D. Manning. "Effective Approaches to Attention-based Neural Machine Translation." In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 1412-1421. 2015.
#
# * https://www.aclweb.org/anthology/D15-1166/ (main paper reference)
# * https://arxiv.org/abs/1508.04025 (alternative paper url)
# * https://github.com/tensorflow/nmt (main code reference)
# * https://www.tensorflow.org/beta/tutorials/text/nmt_with_attention (alternative code reference)
# * https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/seq2seq/python/ops/attention_wrapper.py:2449,2103 (attention implementation)

using Knet, Test, Base.Iterators, Printf, LinearAlgebra, Random, CUDA, IterTools

# ## Code and data from previous projects
#
# Please copy or include the following types and related functions from previous projects:
# `Vocab`, `TextReader`, `MTData`, `Embed`, `Linear`, `mask!`, `loss`, `int2str`,
# `bleu`.

## Your code here


# ## S2S: Sequence to sequence model with attention
#
# In this project we will define, train and evaluate a sequence to sequence encoder-decoder
# model with attention for Turkish-English machine translation. The model has two extra
# fields compared to `S2S_v1`: the `memory` layer computes keys and values from the encoder,
# the `attention` layer computes the attention vector for the decoder.

struct Memory; w; end

struct Attention; wquery; wattn; scale; end

struct S2S
    srcembed::Embed       # encinput(B,Tx) -> srcembed(Ex,B,Tx)
    encoder::RNN          # srcembed(Ex,B,Tx) -> enccell(Dx*H,B,Tx)
    memory::Memory        # enccell(Dx*H,B,Tx) -> keys(H,Tx,B), vals(Dx*H,Tx,B)
    tgtembed::Embed       # decinput(B,Ty) -> tgtembed(Ey,B,Ty)
    decoder::RNN          # tgtembed(Ey,B,Ty) . attnvec(H,B,Ty)[t-1] = (Ey+H,B,Ty) -> deccell(H,B,Ty)
    attention::Attention  # deccell(H,B,Ty), keys(H,Tx,B), vals(Dx*H,Tx,B) -> attnvec(H,B,Ty)
    projection::Linear    # attnvec(H,B,Ty) -> proj(Vy,B,Ty)
    dropout::Real         # dropout probability
    srcvocab::Vocab       # source language vocabulary
    tgtvocab::Vocab       # target language vocabulary
end


# ## Load pretrained model and data
#
# We will load a pretrained model (16.20 bleu) for code testing.  The data should be loaded
# with the vocabulary from the pretrained model for word id consistency.

if !isdefined(Main, :pretrained) || pretrained === nothing
    @info "Loading reference model"
    if !isfile("comp542attn.jld2")
        download("https://github.com/denizyuret/Knet.jl/releases/download/v1.4.5/comp542attn.tar.gz", "comp542attn.tar.gz")
        run(`tar xzf comp542attn.tar.gz`)
    end
    pretrained = Knet.load("comp542attn.jld2", "model")
end
datadir = "datasets/tr_to_en"
if !isdir(datadir)
    @info "Downloading data"
    download("http://www.phontron.com/data/qi18naacl-dataset.tar.gz", "qi18naacl-dataset.tar.gz")
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end
if !isdefined(Main, :tr_vocab)
    BATCHSIZE, MAXLENGTH = 64, 50
    @info "Reading data"
    tr_vocab = pretrained.srcvocab # Vocab("$datadir/tr.train", mincount=5)
    en_vocab = pretrained.tgtvocab # Vocab("$datadir/en.train", mincount=5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    dtrn = MTData(tr_train, en_train, batchsize=BATCHSIZE, maxlength=MAXLENGTH)
    ddev = MTData(tr_dev, en_dev, batchsize=BATCHSIZE)
    dtst = MTData(tr_test, en_test, batchsize=BATCHSIZE)
end

# ## Part 1. Model constructor
#
# The `S2S` constructor takes the following arguments:
# * `hidden`: size of the hidden vectors for both the encoder and the decoder
# * `srcembsz`, `tgtembsz`: size of the source/target language embedding vectors
# * `srcvocab`, `tgtvocab`: the source/target language vocabulary
# * `layers=1`: number of layers
# * `bidirectional=false`: whether the encoder is bidirectional
# * `dropout=0`: dropout probability
#
# Hints:
# * You can find the vocabulary size with `length(vocab.i2w)`.
# * If the encoder is bidirectional `layers` must be even and the encoder should have `layers÷2` layers.
# * The decoder will use "input feeding", i.e. it will concatenate its previous output to its input. Therefore the input size for the decoder should be `tgtembsz+hidden`.
# * Only `numLayers`, `dropout`, and `bidirectional` keyword arguments should be used for RNNs, leave everything else default.
# * The memory parameter `w` is used to convert encoder states to keys. If the encoder is bidirectional initialize it to a `(hidden,2*hidden)` parameter, otherwise set it to the constant 1.
# * The attention parameter `wquery` is used to transform the query, set it to the constant 1 for this project.
# * The attention parameter `scale` is used to scale the attention scores before softmax, set it to a parameter of size 1.
# * The attention parameter `wattn` is used to transform the concatenation of the decoder output and the context vector to the attention vector. It should be a parameter of size `(hidden,2*hidden)` if unidirectional, `(hidden,3*hidden)` if bidirectional.

function S2S(hidden::Int, srcembsz::Int, tgtembsz::Int, srcvocab::Vocab, tgtvocab::Vocab;
             layers=1, bidirectional=false, dropout=0)
    ## Your code here
end

#-
@testset "Testing S2S constructor" begin
    H,Ex,Ey,Vx,Vy,L,Dx,Pdrop = 8,9,10,length(dtrn.src.vocab.i2w),length(dtrn.tgt.vocab.i2w),2,2,0.2
    m = S2S(H,Ex,Ey,dtrn.src.vocab,dtrn.tgt.vocab;layers=L,bidirectional=(Dx==2),dropout=Pdrop)
    @test size(m.srcembed.w) == (Ex,Vx)
    @test size(m.tgtembed.w) == (Ey,Vy)
    @test m.encoder.inputSize == Ex
    @test m.decoder.inputSize == Ey + H
    @test m.encoder.hiddenSize == m.decoder.hiddenSize == H
    @test m.encoder.direction == Dx-1
    @test m.encoder.numLayers == (Dx == 2 ? L÷2 : L)
    @test m.decoder.numLayers == L
    @test m.encoder.dropout == m.decoder.dropout == Pdrop
    @test size(m.projection.w) == (Vy,H)
    @test size(m.memory.w) == (Dx == 2 ? (H,2H) : ())
    @test m.attention.wquery == 1
    @test size(m.attention.wattn) == (Dx == 2 ? (H,3H) : (H,2H))
    @test size(m.attention.scale) == (1,)
    @test m.srcvocab === dtrn.src.vocab
    @test m.tgtvocab === dtrn.tgt.vocab
end


# ## Part 2. Memory
#
# The memory layer turns the output of the encoder to a pair of tensors that will be used as
# keys and values for the attention mechanism. Remember that the encoder RNN output has size
# `(H*D,B,Tx)` where `H` is the hidden size, `D` is 1 for unidirectional, 2 for
# bidirectional, `B` is the batchsize, and `Tx` is the sequence length. It will be
# convenient to store these values in batch major form for the attention mechanism, so
# *values* in memory will be a permuted copy of the encoder output with size `(H*D,Tx,B)`
# (see `@doc permutedims`). The *keys* in the memory need to have the same first dimension
# as the *queries* (i.e. the decoder hidden states). So *values* will be transformed into
# *keys* of size `(H,B,Tx)` with `keys = m.w * values` where `m::Memory` is the memory
# layer. Note that you will have to do some reshaping to 2-D and back to 3-D for matrix
# multiplications. Also note that `m.w` may be a scalar such as `1` e.g. when `D=1` and we
# want keys and values to be identical.


function (m::Memory)(x)
    ## Your code here
end

# You can use the following helper function for scaling and linear transformations of 3-D tensors:
mmul(w,x) = (w == 1 ? x : w == 0 ? 0 : reshape(w * reshape(x,size(x,1),:), (:, size(x)[2:end]...)))

#-
@testset "Testing memory" begin
    H,D,B,Tx = pretrained.encoder.hiddenSize, pretrained.encoder.direction+1, 4, 5
    x = KnetArray(randn(Float32,H*D,B,Tx))
    k,v = pretrained.memory(x)
    @test v == permutedims(x,(1,3,2))
    @test k == mmul(pretrained.memory.w, v)
end


# ## Part 3. Encoder
#
# `encode()` takes a model `s` and a source language minibatch `src`. It passes the input
# through `s.srcembed` and `s.encoder` layers with the `s.encoder` RNN hidden states
# initialized to `0` in the beginning, and copied to the `s.decoder` RNN at the end. The
# steps so far are identical to `S2S_v1` but there is an extra step: The encoder output is
# passed to the `s.memory` layer which returns a `(keys,values)` pair. `encode()` returns
# this pair to be used later by the attention mechanism.

function encode(s::S2S, src)        
    ## Your code here
end

#-
@testset "Testing encoder" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,D,B,Tx = pretrained.encoder.hiddenSize, pretrained.encoder.direction+1, size(src1,1), size(src1,2)
    @test size(key1) == (H,Tx,B)
    @test size(val1) == (H*D,Tx,B)
    @test (pretrained.decoder.h,pretrained.decoder.c) === (pretrained.encoder.h,pretrained.encoder.c)
    @test norm(key1) ≈ 1214.4755f0
    @test norm(val1) ≈ 191.10411f0
    @test norm(pretrained.decoder.h) ≈ 48.536964f0
    @test norm(pretrained.decoder.c) ≈ 391.69028f0
end


# ## Part 4. Attention
#
# The attention layer takes `cell`: the decoder output, and `mem`: a pair of (keys,vals)
# from the encoder, and computes and returns the attention vector. First `a.wquery` is used
# to linearly transform the cell to the query tensor. The query tensor is reshaped and/or
# permuted as appropriate and multiplied with the keys tensor to compute the attention
# scores. Please see `@doc bmm` for the batched matrix multiply operation used for this
# step. The attention scores are scaled using `a.scale` and normalized along the time
# dimension using `softmax`. After the appropriate reshape and/or permutation, the scores
# are multiplied with the `vals` tensor (using `bmm` again) to compute the context
# tensor. After the appropriate reshape and/or permutation the context vector is
# concatenated with the cell and linearly transformed to the attention vector using
# `a.wattn`. Please see the paper and code examples for details.
#
# Note: the paper mentions a final `tanh` transform, however the final version of the
# reference code does not use `tanh` and gets better results. Therefore we will skip `tanh`.

function (a::Attention)(cell, mem)
    ## Your code here
end

#-
@testset "Testing attention" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,B = pretrained.encoder.hiddenSize, size(src1,1)
    x = KnetArray(randn(Float32,H,B,5))
    y = pretrained.attention(x, (key1, val1))
    @test size(y) == size(x)
    @test isapprox(norm(y), 810.0; rtol=0.05)
end


# ## Part 5. Decoder
#
# `decode()` takes a model `s`, a target language minibatch `tgt`, the memory from the
# encoder `mem` and the decoder output from the previous time step `prev`. After the input
# is passed through the embedding layer, it is concatenated with `prev` (this is called
# input feeding). The resulting tensor is passed through `s.decoder`. Finally the
# `s.attention` layer takes the decoder output and the encoder memory to compute the
# "attention vector" which is returned by `decode()`.

function decode(s::S2S, tgt, mem, prev)
    ## Your code here
end

#-
@testset "Testing decoder" begin
    src1,tgt1 = first(dtrn)
    key1,val1 = encode(pretrained, src1)
    H,B = pretrained.encoder.hiddenSize, size(src1,1)
    cell = randn!(similar(key1, size(key1,1), size(key1,3), 1))
    cell = decode(pretrained, tgt1[:,1:1], (key1,val1), cell)
    @test size(cell) == (H,B,1)
    @test isapprox(norm(cell), 132.0; rtol=0.05)
end


# ## Part 6. Loss
#
# The loss function takes source language minibatch `src`, and a target language minibatch
# `tgt` and returns `sumloss/numwords` if `average=true` or `(sumloss,numwords)` if
# `average=false` where `sumloss` is the total negative log likelihood loss and `numwords` is
# the number of words predicted (including a final eos for each sentence). The source is first
# encoded using `encode` yielding a `(keys,vals)` pair (memory). Then the decoder is called to
# predict each word of `tgt` given the previous word, `(keys,vals)` pair, and the previous
# decoder output. The previous decoder output is initialized with zeros for the first
# step. The output of the decoder at each step is passed through the projection layer giving
# word scores. Losses can be computed from word scores and masked/shifted `tgt`.

function (s::S2S)(src, tgt; average=true)
    ## Your code here
end

#-
@testset "Testing loss" begin
    src1,tgt1 = first(dtrn)
    @test pretrained(src1,tgt1) ≈ 1.4666592f0
    sumloss,cntloss = pretrained(src1,tgt1,average=false)
    @test sumloss ≈ 1949.1901f0 && cntloss == 1329
end

# ## Part 7. Greedy translator
#
# An `S2S` object can be called with a single argument (source language minibatch `src`, with
# size `B,Tx`) to generate translations (target language minibatch with size `B,Ty`). The
# keyword argument `stopfactor` determines how much longer the output can be compared to the
# input. Similar to the loss function, the source minibatch is encoded yield a `(keys,vals)`
# pair (memory). We generate the output one time step at a time by calling the decoder with
# the last output, the memory, and the last decoder state. The last output is initialized to
# an array of `eos` tokens and the last decoder state is initialized to an array of
# zeros. After computing the scores for the next word using the projection layer, the highest
# scoring words are selected and appended to the output. The generation stops when all outputs
# in the batch have generated `eos` or when the length of the output is `stopfactor` times the
# input.

function (s::S2S)(src; stopfactor = 3) 
    ## Your code here
end

#-
@testset "Testing translator" begin
    src1,tgt1 = first(dtrn)
    tgt2 = pretrained(src1)
    @test size(tgt2) == (64, 41)
    @test tgt2[1:3,1:3] == [14 25 10647; 37 25 1426; 27 5 349]
end


# ## Part 8. Training
#
# `trainmodel` creates, trains and returns an `S2S` model. The arguments are described in
# comments.

function trainmodel(trn,                  # Training data
                    dev,                  # Validation data, used to determine the best model
                    tst...;               # Zero or more test datasets, their loss will be periodically reported
                    bidirectional = true, # Whether to use a bidirectional encoder
                    layers = 2,           # Number of layers (use `layers÷2` for a bidirectional encoder)
                    hidden = 512,         # Size of the hidden vectors
                    srcembed = 512,       # Size of the source language embedding vectors
                    tgtembed = 512,       # Size of the target language embedding vectors
                    dropout = 0.2,        # Dropout probability
                    epochs = 0,           # Number of epochs (one of epochs or iters should be nonzero for training)
                    iters = 0,            # Number of iterations (one of epochs or iters should be nonzero for training)
                    bleu = false,         # Whether to calculate the BLEU score for the final model
                    save = false,         # Whether to save the final model
                    seconds = 60,         # Frequency of progress reporting
                    )
    @show bidirectional, layers, hidden, srcembed, tgtembed, dropout, epochs, iters, bleu, save; flush(stdout)
    model = S2S(hidden, srcembed, tgtembed, trn.src.vocab, trn.tgt.vocab; 
                layers=layers, dropout=dropout, bidirectional=bidirectional)

    epochs == iters == 0 && return model

    (ctrn,cdev,ctst) = collect(trn),collect(dev),collect.(tst)
    traindata = (epochs > 0 
                 ? collect(flatten(shuffle!(ctrn) for i in 1:epochs))
                 : shuffle!(collect(take(cycle(ctrn), iters))))

    bestloss, bestmodel = loss(model, cdev), deepcopy(model)
    progress!(adam(model, traindata), seconds=seconds) do y
        devloss = loss(model, cdev)
        tstloss = map(d->loss(model,d), ctst)
        if devloss < bestloss
            bestloss, bestmodel = devloss, deepcopy(model)
        end
        println(stderr)
        (dev=devloss, tst=tstloss, mem=Float32(CUDA.usage[]))
    end
    save && Knet.save("attn-$(Int(time_ns())).jld2", "model", bestmodel)
    bleu && Main.bleu(bestmodel,dev)
    return bestmodel
end

# Train a model: If your implementation is correct, the first epoch should take about 24
# minutes on a v100 and bring the loss from 9.83 to under 4.0. 10 epochs would take about 4
# hours on a v100. With other GPUs you may have to use a smaller batch size (if memory is
# lower) and longer time (if gpu speed is lower).

## Uncomment the appropriate option for training:
model = pretrained  # Use reference model
## model = Knet.load("attn-1538395466294882.jld2", "model")  # Load pretrained model
## model = trainmodel(dtrn,ddev,take(dtrn,20); epochs=10, save=true, bleu=true)  # Train model

# Code to sample translations from a dataset
data1 = MTData(tr_dev, en_dev, batchsize=1) |> collect;
function translate_sample(model, data)
    (src,tgt) = rand(data)
    out = model(src)
    println("SRC: ", int2str(src,model.srcvocab))
    println("REF: ", int2str(tgt,model.tgtvocab))
    println("OUT: ", int2str(out,model.tgtvocab))
end

# Generate translations for random instances from the dev set
translate_sample(model, data1)

# Code to generate translations from user input
function translate_input(model)
    v = model.srcvocab
    src = [ get(v.w2i, w, v.unk) for w in v.tokenizer(readline()) ]'
    out = model(src)
    println("SRC: ", int2str(src,model.srcvocab))
    println("OUT: ", int2str(out,model.tgtvocab))
end

# Generate translations for user input
## translate_input(model)

# ## Competition
#
# The reference model `pretrained` has 16.2 bleu. By playing with the optimization algorithm
# and hyperparameters, using per-sentence loss, and (most importantly) splitting the Turkish
# words I was able to push the performance to 21.0 bleu. I will give extra credit to groups
# that can exceed 21.0 bleu in this dataset.
