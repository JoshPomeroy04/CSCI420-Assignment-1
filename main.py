from pygments.lexers.jvm import JavaLexer
from nltk.lm import MLE
from nltk import ngrams


FOLDER_PATH = "/home/jmp/CSCI-420-GenAI/CSCI420-Assignment-1/data"
N = 2
training_corpus = []
eval_test_corpus = []
tokenized_methods = []
lexer = JavaLexer()
training = []
words = []

with open(f"{FOLDER_PATH}/Training_Corpus.txt") as file:
    for line in file:
        training_corpus.append(line.strip())

with open(f"{FOLDER_PATH}/Test_Set.txt", "r") as file:
    for line in file:
        eval_test_corpus.append(line.strip())

with open(f"{FOLDER_PATH}/Eval_Set.txt", "r") as file:
    for line in file:
        eval_test_corpus.append(line.strip())

for method in training_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method
    tokenized_methods += [tokenized_method]
    
for method in eval_test_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method

for method in tokenized_methods:
    training.append(ngrams(method, N))

lm = MLE(N)
lm.fit(training, words)
test = [('public', 'void')]
print(lm.perplexity(test))




