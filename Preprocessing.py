import re
from pygments.lexers.jvm import JavaLexer
from pygments.lexers import get_lexer_by_name
from pygments.token import Token
import pandas as pd

FOLDER_PATH = "CSCI-420-GenAI/CSCI420-Assignment-1/data"
methods_array = []
cleaned_methods = []

def remove_duplicates(data):
    """Remove duplicate methods based on method content.
      Almost Type-1 with the exception of comments
    """
    return data.drop_duplicates(subset="Method Java", keep="first")

def filter_ascii_methods(data):
    """Filter methods to include only those with ASCII characters."""
    data = data[data["Method Java"].apply(lambda x: all(ord(char) < 128 for char in x))]
    return data

def remove_outliers(data, lower_percentile=5, upper_percentile=95):
    """Remove outliers based on method length."""
    method_lengths = data["Method Java"].apply(len)
    lower_bound = method_lengths.quantile(lower_percentile / 100)
    upper_bound = method_lengths.quantile(upper_percentile / 100)
    return data[(method_lengths >= lower_bound) & (method_lengths <= upper_bound)]

def remove_boilerplate_methods(data):
    """Remove boilerplate methods like setters and getters."""
    boilerplate_patterns = [
        r"\bset[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Setter methods
        r"\bget[A-Z][a-zA-Z0-9_]*\(.*\)\s*{",  # Getter methods
    ]
    boilerplate_regex = re.compile("|".join(boilerplate_patterns))
    data = data[~data["Method Java"].apply(lambda x: bool(boilerplate_regex.search(x)))]
    return data

def remove_comments_from_dataframe(df: pd.DataFrame, method_column: str, language: str) -> pd.DataFrame:
    """
    Removes comments from Java methods in a DataFrame and adds a new column with cleaned methods.

    Args:
        df (pd.DataFrame): DataFrame containing the methods.
        method_column (str): Column name containing the raw Java methods.
        language (str): Programming language for the lexer (e.g., 'java').

    Returns:
        pd.DataFrame: Updated DataFrame with a new column 'Java Method No Comments'.
    """
    # Define a function to remove comments from a single method
    def remove_comments(code):
        lexer = get_lexer_by_name(language)
        tokens = lexer.get_tokens(code)
        # Filter out comments using a lambda function
        clean_code = ''.join(token[1] for token in tokens if not (lambda t: t[0] in Token.Comment)(token))

        return clean_code

    # Apply the function to the specified column and add a new column with the results
    df["Method Java No Comments"] = df[method_column].apply(remove_comments)
    return df


with open(f"{FOLDER_PATH}/Corpus.txt", "r") as file:
    for line in file:
        methods_array.append(line.strip())

data = pd.DataFrame({"Method Java" : methods_array})

print("Initial dataset size:", len(data))
data = remove_duplicates(data)
print("After removing duplicates:", len(data))

data = filter_ascii_methods(data)
print("After filtering ASCII methods:", len(data))

data = remove_outliers(data)
print("After removing outliers:", len(data))

data = remove_boilerplate_methods(data)
print("After removing boilerplate methods:", len(data))

data = remove_comments_from_dataframe(data, "Method Java", "Java")
print("After cleaning comments:", len(data))


cleaned_methods = data['Method Java No Comments'].tolist()

with open(f"{FOLDER_PATH}/Cleaned_Corpus.txt", "w") as file:
    for method in cleaned_methods:
        method = method.strip()
        file.write(f"{method}\n")




