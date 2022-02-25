using Languages, Statistics

function dictFrequency(dictionary, scale)
    new_dict = Dict()
    for (key, values) in dictionary
        new_dict[key] = log(values / scale)
    end
    return new_dict
end

function classPriors(data)
    class1 = 0
    class2 = 0
    for review in data
        if review[2] == 1
            class1 += 1
        else
            class2 += 1
        end
    end
    priors=[class1/(class1+class2), class2/(class1+class2)]
end

function prediction(review, WordPos_freq, WordNeg_freq)
    ProbPos, ProbNeg = [], []
    for word in review
        if haskey(WordPos_freq, word)
            push!(ProbPos, WordPos_freq[word])
        else
            push!(ProbPos, log(1/length(WordPos_freq)))
        end
        if haskey(WordNeg_freq, word)
            push!(ProbNeg, WordNeg_freq[word])
        else
            push!(ProbNeg, log(1/length(WordNeg_freq)))
        end
    end
    if sum(ProbPos) > sum(ProbNeg)
        label = 1
    else
        label = 2
    end
    return label
end

function accuracy(data, WordPos_freq, WordNeg_freq)
    res = []
    for (review, label) in data
        flag = label==prediction(review, WordPos_freq, WordNeg_freq)
        push!(res, flag)
    end
    return mean(res)
end

