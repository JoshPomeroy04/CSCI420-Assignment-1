from pygments.lexers.jvm import JavaLexer
from collections import Counter
from nltk import ngrams
from math import log2
from math import floor
import numpy as np
import csv
import sys
import random

FOLDER_PATH = "/home/jmpomeroy/CSCI_420/CSCI420-Assignment-1/data"
FILE_NAME = ""
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
test_set_grams = []
best_n = -1
output_file = ""

if len(sys.argv) > 1:
    FILE_NAME = sys.argv[1]
    file_lines = []
    try:
        with open(FILE_NAME) as file:
            for line in file: 
                file_lines.append(line.strip())
        
        training_size = floor(len(file_lines) * .80)

        eval_size = floor(len(file_lines) * .10)
        test_size = floor(len(file_lines) * .10)

        training_corpus = file_lines[:training_size]
        eval_corpus = file_lines[training_size:training_size + eval_size]
        test_corpus = file_lines[training_size + eval_size:training_size+eval_size+test_size]
        output_file = "results_teacher_model.csv"
    except FileNotFoundError:
        print("File Not Found")
else:
    # Extract Training Corpus File
    with open(f"{FOLDER_PATH}/Training_Corpus.txt") as file:
        for line in file:
            training_corpus.append(line.strip())

    # Extract Eval Set File
    with open(f"{FOLDER_PATH}/Eval_Set.txt", "r") as file:
        for line in file:
            eval_corpus.append(line.strip())

    # Extract Test Set File
    with open(f"{FOLDER_PATH}/Test_Set.txt", "r") as file:
        for line in file:
            test_corpus.append(line.strip())
    
    output_file = "results_student_model.csv"

# Tokenize every method in the Training Set
for method in training_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    words += tokenized_method
    tokenized_training_corpus += [tokenized_method]

# Tokenize every method in the Eval Set and create specific Eval Sets for each ngram
for method in eval_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    tri_eval_set += ngrams(tokenized_method, 3)
    penta_eval_set += ngrams(tokenized_method, 5)
    nona_eval_set += ngrams(tokenized_method, 9)

# Tokenize every method in the Test Set
for method in test_corpus:
    tokenized_method = [t[1].strip() for t in lexer.get_tokens(method) if t[1].strip() != '']
    test_set += [tokenized_method]


def create_ngrams(tokenized_corpus, n):
    """Creates a Counter() containing ngrams.

    Args:
       tokenized_corpus: Tokenized list of values
       n: Integer representing order of gram to create

    Returns:
       Counter() Object containing the counts of each ngram
    """
    ngram = []
    for method in tokenized_corpus:
        ngram += list(ngrams(method, n))

    ngram_counts = Counter(ngram)
    return ngram_counts

def calc_perp(n_minus_one_model, model, eval_set, n):  
    """Calculates perplexity of a given model on a given set.

    Args:
       n_minus_one_model: ngram model of 1 lower order than the one being calculated
       model: Model to calculate perplexity of
       eval_set: List of ngrams matching the order of the model to calculate perplexity on
       n: Order of the model

    Returns:
       Float representing the Perplexity
    """
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
    """Creates a ngram model of n order.

    Args:
       n: Order of the model

    Returns:
       Dictionary representing the model
    """
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

def convert_model_to_prob(model):
    """Converts the number next to word occurances to their probability instead of their count.

    Args:
       model: Model to convert

    Returns:
       Converted model with words in descending order of likelyness given x context
    """
    for context in model:
        total = 0
        for word in model[context]:
            total += model[context][word]
        
        for word in model[context]:
            model[context][word] /= total
        
        model[context] = sorted(model[context].items(), key=lambda x: x[1], reverse=True)
        
    return model

def weighted_random_selection(items, weights):
  """
  Selects a random item from a list based on specified weights.

  Args:
    items: A list of items to choose from.
    weights: A list of weights corresponding to each item, representing probabilities.

  Returns:
    A randomly selected item from the list.
  """
  if len(items) != len(weights):
    raise ValueError("The number of items and weights must be equal.")

  return random.choices(items, weights=weights, k=1)[0]


bigrams = create_ngrams(tokenized_training_corpus, 2)
trigram_model = make_model(3)

quadgrams = create_ngrams(tokenized_training_corpus, 4)
pentagram_model = make_model(5)

octagrams = create_ngrams(tokenized_training_corpus, 8)
nonagram_model = make_model(9)


print("Calculating Trigram Perplexity...")
trigram_perplexity = calc_perp(bigrams, trigram_model, tri_eval_set, 3)
print(f"Trigram Perplexity: {trigram_perplexity}\n")
print("Calculating Pentagram Perplexity...")
pentagram_perplexity = calc_perp(quadgrams, pentagram_model, penta_eval_set, 5)
print(f"Pentagram Perplexity: {pentagram_perplexity}\n")
print("Calculating Nonagram Perplexity...")
nonagram_perplexity = calc_perp(octagrams, nonagram_model, nona_eval_set, 9)
print(f"Nonagram Perplexity: {nonagram_perplexity}\n")

"""
Extracted Corpus Perplexities
"""
#trigram_perplexity = 952908.4984631359
#pentagram_perplexity = 7704743.087282966
#nonagram_perplexity = 19059891.011915024

"""
Teacher Corpus Perplexities
"""
#trigram_perplexity = 156285.80460023868
#pentagram_perplexity = 3207097.3825185383
#nonagram_perplexity = 35417361.621163145

perplexity_list = [trigram_perplexity, pentagram_perplexity, nonagram_perplexity]
min_perplexity = min(perplexity_list)
min_index = perplexity_list.index(min_perplexity)

match min_index:
    case 0:
        final_model = convert_model_to_prob(trigram_model)
        best_n = 3
        print(f"The best performing model is trigrams, with {trigram_perplexity} perplexity")
    case 1:
        final_model = convert_model_to_prob(pentagram_model)
        best_n = 5
        print(f"The best performing model is pentagrams, with {pentagram_perplexity} perplexity")
    case 2:
        final_model = convert_model_to_prob(nonagram_model)
        best_n = 9
        print(f"The best performing model is nonagrams, with {nonagram_perplexity} perplexity")

with open(f"{FOLDER_PATH}/{output_file}", "w", newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["ID", "Given String", "Predicted Continuation", "Full Method"])

    counter = 1
    for method in test_set[:100]:
        method_length = len(method) - best_n - 1
        start_of_method = method[:best_n-1]
        generated_method = method[:best_n-1]
        predictions = []
        current_pos = 0
        paren_count = 0
        bracket_count = 0

        while(method_length):
            try:
                possible_words = []
                possible_word_weights = []

                for word in list(final_model[tuple(generated_method[current_pos:current_pos + best_n-1])]):
                    possible_words.append(word[0])
                    possible_word_weights.append(word[1])

                prediction = weighted_random_selection(possible_words, possible_word_weights)

                # Ensure balanced parens
                if prediction == ')':
                    if paren_count > 0:
                        predictions.append((prediction, possible_word_weights[possible_words.index(prediction)]))
                        generated_method.append(prediction)
                        paren_count -= 1
                    else:
                        if len(possible_words) == 1:
                            break
                        
                        possible_word_weights.pop(possible_words.index(')'))
                        possible_words.remove(')')
                        prediction = weighted_random_selection(possible_words, possible_word_weights)

                        if prediction == '(':
                            paren_count += 1
                        elif prediction == '{':
                            bracket_count += 1

                        predictions.append((prediction, possible_word_weights[possible_words.index(prediction)]))
                        generated_method.append(prediction)

                # Ensure balanced brackets
                elif prediction == '}':
                    if bracket_count > 0:
                        predictions.append((prediction, possible_word_weights[possible_words.index(prediction)]))
                        generated_method.append(prediction)
                        bracket_count -= 1
                    else:
                        if len(possible_words) == 1:
                            break
                        
                        possible_word_weights.pop(possible_words.index('}'))
                        possible_words.remove('}')
                        prediction = weighted_random_selection(possible_words, possible_word_weights)

                        if prediction == '(':
                            paren_count += 1
                        elif prediction == '{':
                            bracket_count += 1

                        predictions.append((prediction, possible_word_weights[possible_words.index(prediction)]))
                        generated_method.append(prediction)
                else:
                    if prediction == '{':
                        bracket_count += 1
                    elif prediction == '(':
                        paren_count += 1
                    
                    predictions.append((prediction, possible_word_weights[possible_words.index(prediction)]))
                    generated_method.append(prediction)
            except KeyError:
                break
            method_length -= 1
            current_pos += 1

        while(paren_count):
            predictions.append((')', "Paren balancing"))
            generated_method.append(')')
            paren_count -= 1
        while(bracket_count):
            predictions.append(('}', "Bracket balancing"))
            generated_method.append(')')
            bracket_count -= 1

        csv_writer.writerow([counter, start_of_method, predictions, ' '.join(generated_method)])
        counter += 1
        if counter == 101:
            break

# Calculate Perplexity on test set
lower_gram = create_ngrams(tokenized_training_corpus, best_n-1)
for method in test_set:
    test_set_grams += ngrams(method, best_n)
test_model = make_model(best_n)
test_perp = calc_perp(lower_gram, test_model, test_set_grams, best_n)
print(f"Perplexity on Test set is: {test_perp}")

"""
Student Test Set Perplexity: 969314.0181455543
Teacher Test Set Perplexity: 156094.86393119284
"""











