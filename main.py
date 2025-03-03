from pygments.lexers.jvm import JavaLexer
from nltk.lm import MLE
from nltk import ngrams
from nltk.lm.preprocessing import padded_everygram_pipeline


FOLDER_PATH = "/home/jmp/CSCI-420-GenAI/CSCI420-Assignment-1/data"
N = 2
training_corpus = []
eval_test_corpus = []
tokenized_training_corpus = []
lexer = JavaLexer()
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
    tokenized_training_corpus += [tokenized_method]
    
for method in eval_test_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method


def create_ngrams_model(tokenized_corpus, vocab, N):
    training = []
    train, voc = padded_everygram_pipeline(N, tokenized_corpus)
    for method in tokenized_corpus:
        training.append(ngrams(method, N))
    
    lm = MLE(N)
    lm.fit(train, vocab)
    return lm

trigram_model = create_ngrams_model(tokenized_training_corpus, words, 3)
pentgram_model = create_ngrams_model(tokenized_training_corpus, words, 5)
nongram_model = create_ngrams_model(tokenized_training_corpus, words, 9)

# public void testSystemStopped
# public void defaultIsPerTestSystemTestRun() { TestRun run = registry.createRun(Collections.emptyList()); assertEquals(PerTestSystemTestRun.class, run.getClass()); addFactory(pages -> Optional.empty()); run = registry.createRun(Collections.emptyList()); assertEquals(PerTestSystemTestRun.class, run.getClass()); }
test = [('public', 'void', 'testInvalidPath',)]

print(trigram_model.perplexity(test))
print(trigram_model.vocab.lookup(['public', 'void', 'testInvalidPath']))
print(pentgram_model.perplexity(test))
print(pentgram_model.vocab.lookup(['public', 'void', 'testInvalidPath']))
print(nongram_model.perplexity(test))
print(nongram_model.vocab.lookup(['public', 'void', 'testInvalidPath']))








