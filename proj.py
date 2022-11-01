import sys
import os
import re
import string
from tkinter import N
import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import scipy.spatial as sp
from sklearn.metrics import f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

translate = str.maketrans("", "", string.punctuation)
stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any','are',"aren't", 'as', 'at', 'be', 'because',
            'been','before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do',
            'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't",
            'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his',
            'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me',
            'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
            'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such',
            'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd",
            "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we',
            "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who',
            "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours',
            'yourself', 'yourselves']

stop_words_ignored = ['all', 'and', 'again', 'any', "aren't", "can't", 'cannot', "couldn't", "didn't", "doesn't", "don't", "few", "hadn't", "hasn't", "haven't", 'more', 'most',
                    "mustn't", 'nor', 'not', 'no', 'off', 'or' 'same', "shan't", "shouldn't", 'some', 'such', 'too', 'very', "wasn't", "weren't", "won't", "wouldn't"]

vectorizer = None
classifier = None
review = None
labels = None

def stem_lem(line):

    p_line = None

    stemmer = nltk.stem.snowball.EnglishStemmer()
    p_line = [stemmer.stem(word) for word in line]

    return p_line

def evaluate(solution, output):
    sol_lines = solution
    out_lines = output.splitlines()

    correct_labels = 0
    n_labels = 0       

    for j in range(len(sol_lines)):
        if out_lines[j] == '\n':
            continue

        if re.sub('[\s\x00]','',out_lines[j]) in str(sol_lines[j]):
            correct_labels += 1
        n_labels +=1 

    accuracy = (correct_labels/n_labels)*100

    print('Accuracy {:.3f}'.format(accuracy))

    return 

def evaluate_labels(solution, output):
    sol_lines = solution
    out_lines = output.splitlines()

    correct_labels = {}
    n_labels = {}       

    for i in range(len(sol_lines)):
        if sol_lines[i] not in correct_labels:
            correct_labels[sol_lines[i]] = 0
        if sol_lines[i] not in n_labels:
            n_labels[sol_lines[i]] = 0
        if sol_lines[i] == out_lines[i]:
            correct_labels[sol_lines[i]] += 1 
        n_labels[sol_lines[i]] += 1

    accuracy = {}
    for label in correct_labels:
        if label not in accuracy:
            accuracy[label] = 0
        accuracy[label] = (correct_labels[label]/n_labels[label])*100
        #print('Accuracy for label {}: {:.3f}'.format(label, accuracy[label]))

    return 

def preprocess_review(review):

    # Lowercasing the entire line
    p_review = review.lower()
    
    for word in nltk.word_tokenize(p_review):
        if word in stop_words and word not in stop_words_ignored:
            p_review = re.sub('{}\s'.format(word),'', p_review)

    # Remove punctuation using regex
    p_review = re.sub('[?!/\.,;:`\']','',  p_review)

    # Tokenization 
    p_review = nltk.word_tokenize(p_review)

    # Stemmerization and Lemmatization
    p_review = stem_lem(p_review)

    str_review = ''
    for word in p_review:
        str_review += ('{} '.format(word))
    return str_review.strip()


def preprocess_data(train_file_name):
    train_file = open(train_file_name, 'r')
    train_lines = train_file.readlines()
    train_file.close()

    train_set = []
    labels = [] 
    #train set
    for line in train_lines:
        split_line = line.split('\t')
        label = split_line[0]
        
        review = split_line[1].strip()

        proc_review = preprocess_review(review)
        
        train_set.append(proc_review)
        
        labels.append(label)

    return train_set, labels

def train_data(train_set, labels):

    global vectorizer
    global classifier

    #new_train, test, new_labels, test_labels = train_test_split(train_set, labels, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    trainVectorizerArray = vectorizer.fit_transform(train_set)
    
    #classifier = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0)
    #classifier.fit(trainVectorizerArray, labels)

    #classifier = svm.SVC(kernel='rbf', C=1, gamma=1.0)
    #classifier.fit(trainVectorizerArray, labels)

    #classifier = MultinomialNB(alpha=3.5, fit_prior=True)
    #classifier.fit(trainVectorizerArray, labels)

    #classifier = KNeighborsClassifier(n_neighbors=5)  
    #classifier.fit(trainVectorizerArray, labels)

    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)

    #test_parameters(trainVectorizerArray, labels, cv)

    scores = cross_val_score(classifier, trainVectorizerArray, labels, cv=cv)
    print(scores)
    print(scores.mean())

    #vec = vectorizer.transform([phrase]).toarray()
    #result_file += get_cosine_sim(cosine_set, vec) + "\n"

def test_parameters(X,y,cv_p):
    grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid = {'n_estimators': [200],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'random_state': [0]
        },  
    scoring='accuracy',
    verbose=10,
    cv=cv_p)

    grid.fit(X,y)

    #print the best parameters from all possible combinations
    print("best parameters are: ", grid.best_params_)

def classify(test):
    test_file = open(test, 'r') 
    test_lines = test_file.readlines()
    test_file.close()

    global vectorizer
    global classifier

    test_set = []
    result_file = ""

    # test set
    for line in test_lines:
        p_review = preprocess_review(line)
  
        test_set.append(p_review)
        
        #vec = vectorizer.transform([phrase]).toarray()
        #result_file += get_cosine_sim(cosine_set, vec) + "\n"

        testVectorizerArray = vectorizer.transform([p_review])
        aux_predict = classifier.predict(testVectorizerArray)[0]
        result_file += aux_predict + "\n"

    print(result_file)

    return result_file

def classify_dummy(test_set):
    global vectorizer
    global classifier

    result_file = ""

    # test set
    for review in test_set:
        #vec = vectorizer.transform([phrase]).toarray()
        #result_file += get_cosine_sim(cosine_set, vec) + "\n"

        testVectorizerArray = vectorizer.transform([review])
        aux_predict = classifier.predict(testVectorizerArray)[0]
        result_file += aux_predict + "\n"

    #print(result_file)

    return result_file
    
def main():
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]

    p_reviews, labels = preprocess_data(train_file_name)
    new_train, test, train_labels, test_labels = train_test_split(p_reviews, labels, test_size=0.30, random_state=0)
    train_data(new_train, train_labels)
    result_file = classify_dummy(test)
    evaluate(test_labels, result_file)
    #evaluate_labels(test_labels, result_file)

    #train_data(p_reviews, labels)
    #result_file = classify(test_file_name)

    #evaluate(labels, result_file)
    #evaluate_labels(test_set_labels, result_file)

    #f_score = f1_score(test_set_labels, result_file.splitlines(), average='weighted')
    #print("f_score:" + f_score)
    return 0

main()
 

