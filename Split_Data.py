import random

FOLDER_PATH = "CSCI-420-GenAI/CSCI420-Assignment-1/data"
TRAINING_SIZE = 20000
TEST_SIZE = 2500
EVAL_SIZE = 2500
raw_corpus = []

with open(f"{FOLDER_PATH}/Cleaned_Corpus.txt", "r") as file:
    for line in file:
        raw_corpus.append(line.strip())

random.shuffle(raw_corpus)

with open(f"{FOLDER_PATH}/Training_Corpus.txt", "w") as file:
    for x in range(TRAINING_SIZE):
        file.write(f"{raw_corpus.pop(0)}\n")

with open(f"{FOLDER_PATH}/Test_Set.txt", "w") as file:
    for x in range(TEST_SIZE):
        file.write(f"{raw_corpus.pop(0)}\n")

with open(f"{FOLDER_PATH}/Eval_Set.txt", "w") as file:
    for x in range(EVAL_SIZE):
        file.write(f"{raw_corpus.pop(0)}\n")


