# CSCI420-Assignment-1

# **1. Introduction**  
This project explores **code completion in Java**, leveraging **N-gram language modeling**. The N-gram model predicts the next token in a sequence by learning the probability distributions of token occurrences in training data. The model selects the most probable token based on learned patterns, making it a fundamental technique in natural language processing and software engineering automation.  

# **2. Getting Started**  

This project is implemented in **Python 3.10.12** 64-bit. It was developed and tested on **Ubuntu 22.04.3 LTS**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
~ $ git clone https://github.com/your-repository/your-project.git

(2) Navigate into the repository:

~ $ cd your-project
~/your-project $
```

Ensure you set the FOLDER_PATH variable to be the correct folder path of the data folder on your machine. This variable is easily found at the top of main.py.

## **2.2 Install Packages**

Install the required dependencies:

pip install nltk
pip install pygments
pip install numpy
pip install collections
pip install csv

## **2.3 Run N-gram**

(1) Run N-gram Demo

The script can take a corpus of Java methods as input, if no input is provided, it defaults to the precollected data in the data folder. The script identifies the best N value for the model based on perplexity. It then tests the best model on a section of the corpus split off for testing. If a file is provided as input then the output file is results_teacher_model.csv where if no file is provided then the output file is results_student_model.csv. Both files are created in the data folder. 

python main.py corpus.txt or python main.py