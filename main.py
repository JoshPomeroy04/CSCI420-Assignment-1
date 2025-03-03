from pygments.lexers.jvm import JavaLexer
from collections import Counter
from nltk.lm import MLE
from nltk import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary
from math import log2
from math import inf
import numpy as np

FOLDER_PATH = "/home/jmp/CSCI-420-GenAI/CSCI420-Assignment-1/data"
lexer = JavaLexer()
training_corpus = []
eval_corpus = []
tri_eval_set = []
penta_eval_set = []
nona_eval_set = []
tokenized_training_corpus = []
words = []

with open(f"{FOLDER_PATH}/Training_Corpus.txt") as file:
    for line in file:
        training_corpus.append(line.strip())

with open(f"{FOLDER_PATH}/Eval_Set.txt", "r") as file:
    for line in file:
        eval_corpus.append(line.strip())

for method in training_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method
    tokenized_training_corpus += [tokenized_method]

for method in eval_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    tri_eval_set += ngrams(tokenized_method, 3)
    penta_eval_set += ngrams(tokenized_method, 5)
    nona_eval_set += ngrams(tokenized_method, 9)

def create_ngrams(tokenized_corpus, N):
    ngram = []
    for method in tokenized_corpus:
        ngram += list(ngrams(method, N))

    ngram_counts = Counter(ngram)
    return ngram_counts


def calc_perp(n_minus_one_gram_Counts, ngram_Counts, N, eval_set):   
    n_minus_one_vocab_size = sum(n_minus_one_gram_Counts.values())
    no_of_ngrams=0
    prob_array = []

    for ngram in eval_set:
        no_of_ngrams += 1
        numerator = ngram_Counts[ngram] + 1
        denominator = n_minus_one_gram_Counts[ngram[0:N-1]] + n_minus_one_vocab_size
        token_probability = log2(float(numerator) / float(denominator))
        prob_array.append(token_probability)
        
    total_ngram_perplexity = np.exp(-np.mean(prob_array))
        
    return total_ngram_perplexity

bigrams = create_ngrams(tokenized_training_corpus, 2)
trigrams = create_ngrams(tokenized_training_corpus, 3)

quadgrams = create_ngrams(tokenized_training_corpus, 4)
pentagrams = create_ngrams(tokenized_training_corpus, 5)

octagrams = create_ngrams(tokenized_training_corpus, 8)
nonagrams = create_ngrams(tokenized_training_corpus, 9)

trigram_perplexity = calc_perp(bigrams, trigrams, 3, tri_eval_set)
pentagram_perplexity = calc_perp(quadgrams, pentagrams, 5, penta_eval_set)
nonagram_perplexity = calc_perp(octagrams, nonagrams, 9, nona_eval_set)

print(trigram_perplexity)
print(pentagram_perplexity)
print(nonagram_perplexity)









