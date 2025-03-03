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
pent_eval_set = []
non_eval_set = []
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
    pent_eval_set += ngrams(tokenized_method, 5)
    non_eval_set += ngrams(tokenized_method, 9)

def create_ngrams(tokenized_corpus, N):
    ngram = []
    for method in tokenized_corpus:
        ngram += list(ngrams(method, N))

    ngram_counts = Counter(ngram)
    return ngram_counts


def calcTrigramPerp(bigram_Counts, trigram_Counts, tokens):   
    bi_vocab_size = sum(bigram_Counts.values())
    no_of_trigrams=0
    prob_array = []

    for trigram in tokens:
        no_of_trigrams += 1
        numerator = trigram_Counts[trigram] + 1
        denominator = bigram_Counts[trigram[0:2]] + bi_vocab_size
        token_probability = log2(float(numerator) / float(denominator))
        prob_array.append(token_probability)
        
    total_trigram_perplexity = np.exp(-np.mean(prob_array))
        
    return total_trigram_perplexity

bigrams = create_ngrams(tokenized_training_corpus, 2)
trigrams = create_ngrams(tokenized_training_corpus, 3)

pentagrams = create_ngrams(tokenized_training_corpus, 5)
nongrams = create_ngrams(tokenized_training_corpus, 9)

trigram_perplexity = calcTrigramPerp(bigrams, trigrams, tri_eval_set)

print(trigram_perplexity)

"""
print(trigram_model.vocab.lookup("public"))
print(trigram_model.unmasked_score("public"))
print(trigram_model.logscore("PagesByTestSystem"))
print(trigram_model.logscore("partition"))
print(trigram_model.logscore("("))
print(trigram_model.logscore("Function"))
print(trigram_model.logscore("<"))
print(trigram_model.logscore("List"))
print(trigram_model.logscore("WikiPage"))
print(trigram_model.logscore("partition", ['public', 'PagesByTestSystem']))
print(trigram_model.vocab.lookup(['>']))
print(trigram_model.vocab.lookup(['public', 'PagesByTestSystem', 'partition', '(', 'Function', '<', 'List', '<', 'WikiPage']))
"""
# [self.logscore(ngram[-1], ngram[:-1]) for ngram in text_ngrams]

#print(trigram_model.unmasked_score("public")+trigram_model.logscore("PagesByTestSystem")+trigram_model.logscore("partition")+trigram_model.logscore("(") + trigram_model.logscore("Function") + trigram_model.logscore("<") + trigram_model.logscore("List") + trigram_model.logscore("WikiPage") + trigram_model.logscore(">"))
#print(pentgram_model.perplexity(test))

#print(nongram_model.perplexity(test))









