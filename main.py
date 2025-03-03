from pygments.lexers.jvm import JavaLexer
from collections import Counter
from nltk import ngrams
from math import log2
import numpy as np

FOLDER_PATH = "/home/jmp/CSCI-420-GenAI/CSCI420-Assignment-1/data"
lexer = JavaLexer()
final_model = Counter()
training_corpus = []
eval_corpus = []
test_corpus = []
tri_eval_set = []
penta_eval_set = []
nona_eval_set = []
tokenized_training_corpus = []
words = []
test_set = []
best_n = -1


with open(f"{FOLDER_PATH}/Training_Corpus.txt") as file:
    for line in file:
        training_corpus.append(line.strip())

with open(f"{FOLDER_PATH}/Eval_Set.txt", "r") as file:
    for line in file:
        eval_corpus.append(line.strip())

with open(f"{FOLDER_PATH}/Test_Set.txt", "r") as file:
    for line in file:
        test_corpus.append(line.strip())

for method in training_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method
    tokenized_training_corpus += [tokenized_method]

for method in eval_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    tri_eval_set += ngrams(tokenized_method, 3)
    penta_eval_set += ngrams(tokenized_method, 5)
    nona_eval_set += ngrams(tokenized_method, 9)

for method in test_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    test_set += [tokenized_method]

def create_ngrams(tokenized_corpus, N):
    ngram = []
    for method in tokenized_corpus:
        ngram += list(ngrams(method, N))

    ngram_counts = Counter(ngram)
    return ngram_counts


def calc_perp(n_minus_one_gram_Counts, ngram_Counts, N, eval_set):   
    n_minus_one_vocab_size = sum(n_minus_one_gram_Counts.values())
    prob_array = []

    for ngram in eval_set:
        numerator = ngram_Counts[ngram] + 1
        denominator = n_minus_one_gram_Counts[ngram[0:N-1]] + n_minus_one_vocab_size
        token_probability = log2(float(numerator) / float(denominator))
        prob_array.append(token_probability)
        
    total_ngram_perplexity = np.exp(-np.mean(prob_array))
        
    return total_ngram_perplexity

def calc_perp2(n_minus_one_model, model, eval_set, n):  
    prob_array = []

    for ngram in eval_set:
        numerator = 1
        denominator = sum(n_minus_one_model.values()) 
        try:
            numerator += model[ngram[:n-1]][ngram[n-1]]
            
            for words in model[ngram[:n-1]]:
                if words != ngram[n-1]:
                    denominator += model[ngram[:n-1]][words]
        except KeyError:
            continue
        
        token_probability = log2(float(numerator) / float(denominator))
        prob_array.append(token_probability)
        
    total_ngram_perplexity = np.exp(-np.mean(prob_array))
        
    return total_ngram_perplexity

def make_model(n):
    ngram_list = []
    model = {}
    for method in tokenized_training_corpus:
        ngram_list += list(ngrams(method, n))
    
    for gram in ngram_list:
        word = gram[n-1]
        context = gram[:n-1]

        if context not in model:
            model[context] = {word: 1}
        else:
            if word not in model[context]:
                model[context][word] = 1
            else:
                model[context][word] += 1
    
    return model

bigrams = create_ngrams(tokenized_training_corpus, 2)
trigram_model = make_model(3)

quadgrams = create_ngrams(tokenized_training_corpus, 4)
pentagram_model = make_model(5)

octagrams = create_ngrams(tokenized_training_corpus, 8)
nonagram_model = make_model(9)


print(calc_perp2(bigrams, trigram_model, tri_eval_set, 3))
print(calc_perp2(quadgrams, pentagram_model, penta_eval_set, 5))
print(calc_perp2(octagrams, nonagram_model, nona_eval_set, 9))
quit()
trigrams = create_ngrams(tokenized_training_corpus, 3)


pentagrams = create_ngrams(tokenized_training_corpus, 5)


nonagrams = create_ngrams(tokenized_training_corpus, 9)

trigram_perplexity = calc_perp(bigrams, trigrams, 3, tri_eval_set)
pentagram_perplexity = calc_perp(quadgrams, pentagrams, 5, penta_eval_set)
nonagram_perplexity = calc_perp(octagrams, nonagrams, 9, nona_eval_set)

perplexity_list = [trigram_perplexity, pentagram_perplexity, nonagram_perplexity]
min_perplexity = min(perplexity_list)
min_index = perplexity_list.index(min_perplexity)

match min_index:
    case 0:
        final_model = trigrams
        best_n = 3
        print(f"The best performing model is trigrams, with {trigram_perplexity} perplexity")
    case 1:
        final_model = pentagrams
        best_n = 5
        print(f"The best performing model is pentagrams, with {pentagram_perplexity} perplexity")
    case 2:
        final_model = nonagrams
        best_n = 9
        print(f"The best performing model is nonagrams, with {nonagram_perplexity} perplexity")

for method in test_set:
    start_of_method = method[:best_n-1]
    for gram in final_model:
        print(gram)
        quit()












