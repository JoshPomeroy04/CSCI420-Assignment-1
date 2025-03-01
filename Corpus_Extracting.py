import pandas as pd
from pydriller import Repository
import csv
import javalang
import os
import re

FOLDER_PATH = "CSCI-420-GenAI/CSCI420-Assignment-1/data"

def extract_methods_from_java(code):
    """
    Extract methods from Java source code using javalang parser.

    Args:
        code (str): The Java source code.

    Returns:
        list: A list of tuples containing method names and their full source code.
    """
    methods = []
    try:
        # Parse the code into an Abstract Syntax Tree (AST)
        tree = javalang.parse.parse(code)
        lines = code.splitlines()

        # Traverse the tree to find method declarations
        for _, node in tree.filter(javalang.tree.MethodDeclaration):
            method_name = node.name

            # Determine the start and end lines of the method
            start_line = node.position.line - 1
            end_line = None

            # Use the body of the method to determine its end position
            if node.body:
                last_statement = node.body[-1]
                if hasattr(last_statement, 'position') and last_statement.position:
                    end_line = last_statement.position.line

            # Extract method code
            if end_line:
                method_code = "\n".join(lines[start_line:end_line+1])
            else:
                # If end_line couldn't be determined, extract up to the end of the file
                method_code = "\n".join(lines[start_line:])

            methods.append((method_name, method_code))

    except Exception as e:
        print(f"Error parsing Java code: {e}")
    return methods


def extract_methods_to_csv_from_master(repo_path, output_csv):
    """
    Extract methods from Java files in the master branch and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        for commit in Repository(repo_path, only_in_branch="master").traverse_commits():
            print(f"Processing commit: {commit.hash}")

            #We only look into the modified files. In other words, we are looking into the history of the software system by traversing each commit.
            #Various Generative AI methods for SD have been trained on data collected in this way; for example bug fixing.
            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([commit.hash, modified_file.filename, method_name, method_code, commit_link])

                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")


def extract_methods_to_csv(repo_path, output_csv):
    """
    Extract methods from Java files in a repository and save them in a CSV file.

    Args:
        repo_path (str): Path to the Git repository.
        output_csv (str): Path to the output CSV file.
    """
    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Branch Name", "Commit Hash", "File Name", "Method Name", "Method Code", "Commit Link"])

        branch_name = "master"
        for commit in Repository(repo_path, only_in_branch=branch_name).traverse_commits():
            print(f"Processing commit: {commit.hash}")

            for modified_file in commit.modified_files:
                if modified_file.filename.endswith(".java") and modified_file.source_code:
                    methods = extract_methods_from_java(modified_file.source_code)

                    for method_name, method_code in methods:
                        commit_link = f"{repo_path}/commit/{commit.hash}"
                        csv_writer.writerow([branch_name, commit.hash, modified_file.filename, method_name, method_code, commit_link])

                    print(f"Extracted methods from {modified_file.filename} in commit {commit.hash}")

def replace_multiple_spaces(text):
  """Replaces multiple spaces with a single space in a string.

  Args:
    text: The input string.

  Returns:
    The string with multiple spaces replaced by single spaces.
  """
  return re.sub(r"\s+", " ", text)

csv_corpus = pd.read_csv(f'{FOLDER_PATH}/Raw_Data/results.csv')

repoList = []
for idx,row in csv_corpus.iterrows():
  repoList.append("https://www.github.com/{}".format(row['name']))

for repo in repoList[:30]:

    fileNameToSave = ''.join(repo.split('github.com')[1:])
    fileNameToSave = fileNameToSave.replace('/','_')

    # Specify the path to the output CSV file
    output_csv_file = f"{FOLDER_PATH}/Extracted_Data/extracted_methods_{fileNameToSave}.csv"
    # Run the extraction
    try:
        extract_methods_to_csv_from_master(repo, output_csv_file)
    except:
        continue

with open(f"{FOLDER_PATH}/Corpus.txt", "w") as file:
    try:
        for filename in os.listdir(f"{FOLDER_PATH}/Extracted_Data"):
            file_path = os.path.join(f"{FOLDER_PATH}/Extracted_Data", filename)
            csv_corpus = pd.read_csv(file_path)
            for idx,row in csv_corpus.iterrows():
                method = replace_multiple_spaces(row['Method Code'].replace("\n", " ").replace("\t", " "))
                print(f"Writing: {method}")
                file.write(f"{method}\n")

    except FileNotFoundError:
        print(f"Error: Folder not found at {f"{FOLDER_PATH}/Extracted_Data"}")
    except Exception as e:
        print(f"An error occurred: {e}")

