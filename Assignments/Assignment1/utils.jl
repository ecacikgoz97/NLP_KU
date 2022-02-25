using Languages, Statistics

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
        #review = replace(review, stop_words => " ")
        review = replace.(review, "<br>" => " ", r"[^a-zA-Z\s-]" => " ", "--" => " ", "\u85" => " ", "-" => " ", "\t" => " ")
        #review = split(review, " ")
        #wordids = w2i.(split(review))
        words = split(review, " ")
        #words = setdiff(words, stop_words)
        words = setdiff(words, " ")
        words = w2i.(words)
        push!(data, (words, tag))
        close(f)
    end
    return data
end

function build_wordcount_dict(arr)
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

