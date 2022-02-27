using Languages, Statistics

function DataLoader(path::String, class::String)
    """
    This function labels the corresponding reviews either 1 or 2
    to positive and negative classes. Then it pre-process the data as:
    lowercase each word,remove specified punctuations, split the 
    sentences, and finally convert words to IDs.
    
    Arguments:
        path(String): corresponding data path that to be download
        class(String): specifiy whether the path is class of positives
                       or negatives.
    Return:
        data(list): returns the pre-processed data as nested list.
    
    """
    
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
        review = replace.(review, "<br>" => " ", r"[^a-zA-Z\s-]" => " ", "--" => " ", "\u85" => " ", "-" => " ", "\t" => " ")
        words = split(review, " ")
        words = setdiff(words, " ")
        words = w2i.(words)
        push!(data, (words, tag))
        close(f)
    end
    return data
end

function build_wordcount_dict(arr)
    """
    This functions counts the words for specific review in given array.
    
    Arguments:
        arr(list): List of reviews.
    
    Return:
        word_dict(dict): number of specific words as dictionary. Keys
                         are the words and Values are the number of
                         of that word in document.
    
    """
    word_dict = Dict()
    for review in arr
        for word in review[1] 
            if !haskey(word_dict, word)
                get!(word_dict, word, 0)
            end
            word_dict[word] += 1
        end
    end
    word_dict
end

