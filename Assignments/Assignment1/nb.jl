using Statistics


function dictFrequency(dictionary, scale)
    """
    Calculate the fraction of each word over among all words in that document
    for corresponding class and take the loglikelihood.
    
    Arguments:
        dictionary(dict): class specific dictionary that contains number of 
                          words in each document.
        scale(Float): normalizing factor.
                        
    Return:
        new_dict(dict): dictionary that stores the log-likelihoods for given
                        document.
    """
    
    new_dict = Dict()
    for (key, values) in dictionary
        new_dict[key] = log(values / scale)
    end
    return new_dict
end


function classPriors(data)
    """
    Calcualte class priors.
    *Since we have equal number of positive and negative classes,
    it may be not used.
    
    Arguments:
        data(list): nested list of the reviews as [[words], class].
    
    Return:
        pririors(array): 2-dimensional array of scores.
    """
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
    """
    This function calculetes the prediction of a review,
    according to the log-likelihood of the corresponding words.
    It makes the prediction for each word in all the documents,
    irrespective of classes, calcuate the likelihood and store it.
    It assigns the label for grater sum probability.
    
    Arguments:
        review(list): list of words in all classes.
        WordPos_freq(dict): Word of frequencies for positive class.
        WordNeg_freq(dict): Word of frequencies for negative class.
    
    returns:
        label(scalar): Prediction of given review as scalar. 1 for
                       positive reviews 2 for negative reviews.
    """
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
    """
    Calculate the accuracy for given data according to
    the outputs of our predictions.
    
    Arguments:
        data(list): Nested list of reviews and labels.
        WordPos_freq(dict): Word of frequencies for positive class.
        WordNeg_freq(dict): Word of frequencies for negative class.
    
    Return:
        accuracy(float): accuracy of our naive approach.
    
    """
    res = []
    for (review, label) in data
        flag = label==prediction(review, WordPos_freq, WordNeg_freq)
        push!(res, flag)
    end
    return mean(res)
end

