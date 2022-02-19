function DataLoader(path::String, class::String)
    
    if lowercase(class) == "pos"
        tag = 1
    elseif lowercase(class) == "neg"
        tag = 2
    else
        error("class must be either 'pos' or 'neg'")
    end
    
    data = []
    for file in readdir(path)
        full_path = joinpath(path, file)
        f = open(full_path, "r")
        review = read(f, String)
        review = lowercase(review)
        review = replace(review, r"<br>" => " ", r"[^a-zA-Z\s-]" => " ", r"--" => " ")
        #review = split(review, " ")
        wordids = w2i.(split(review))
        push!(data, (wordids, tag))
        close(f)
    end
    return data
end

function freqWords(data)
    """
    Calculate word frequencies in a dictionary.
    """
    words = Dict()
    for review in data
        for word in review[1]
            words[word] = get(words, word, 0) + 1
        end
    end
    return words
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
    priors=[class1, class2]
end

function word_counter(data)
    num_words = Dict()
    cntr = 0
    for review in data
        for word in review[1]
            cntr+=1
        end
    end
    return cntr
end