from pygments.lexers.jvm import JavaLexer


FOLDER_PATH = "/home/jmp/CSCI-420-GenAI/CSCI420-Assignment-1/data"
corpus = []
tokens = []
lexer = JavaLexer()


with open(f"{FOLDER_PATH}/Training_Corpus.txt") as file:
    for line in file:
        corpus.append(line.strip())

for method in corpus:
    tokens += [t[1] for t in lexer.get_tokens(method)]

print(tokens)

