#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:18:53 2020

@author: alexandrinio
"""

import re
from string import punctuation
import pandas as pd
import sys
sys.path.append('.../Text_mining_CW1')


#######################################################

"""
Read input file
@ param string path
@ returns data
"""
def read_file(path):
    file = open(path,'r+')
    data = file.read()
    data = data.split('\n')
    file.close()
    return data

#######################################################

"""
Converts the question in lowercase.
@ param string question
@ returns question in lowercase
"""
def lowercase(question):
    
    return str(question).lower()


"""
Removes the numbers from the question.
@ param string question
@ return the question without numbers
"""
def remove_numbers(question):
    return re.sub(r'\d+', "", question)

def clean_numbers(question):

    question = re.sub('[0-9]{5,}', '#####', question)
    question = re.sub('[0-9]{4}', '####', question)
    question = re.sub('[0-9]{3}', '###', question)
    question = re.sub('[0-9]{2}', '##', question)
    return question


"""
Decontracts the question.
@ param string question
@ return the question cleaned
"""
def decontracted(question):
    question = re.sub(r"n\'t", " not", question)
    question = re.sub(r"\'re", "are", question)
    question = re.sub(r"\'s", "is", question)
    question = re.sub(r"\'d", "would", question)
    question = re.sub(r"\'ll", "will", question)
    question= re.sub(r"\'ve", "have", question)
    question = re.sub(r"\'m", "am", question)
    return question

"""
Removes the punctations from a question.
@ param string question
@ return the question without punctuation
"""
def remove_punctuation(question):
    return ''.join(c for c in question if c not in punctuation)

"""
All pre_processing together.
@ param string question
@ return the question cleaned
"""

def pre_process(question,lcase):
    if lcase == 'True':
        question = lowercase(question)
    question = decontracted(question)
    question = remove_punctuation(question)
    question = remove_numbers(question)
    question = remove_stopwords(question)
    return question

#######################################################

"""
@ param my_dictionary
@ return my_dictionary without stop_words and words with less than k appereance
"""
def remove_stopwords(question):
    stopwords = read_file('../data/stopwords.txt')
    question = question.split()
    return ' '.join([word for word in question if word not in stopwords])

def stop_words_removal(my_dictionary):
    stopwords = read_file('../data/stopwords.txt')
    return {key:value for (key,value) in my_dictionary.items() if key not in stopwords}

def remove_less_than_k(my_dictionary):
    k = 3
    return {key:value for (key,value) in my_dictionary.items() if value > k}

def build_dictionary(my_dictionary):
    #my_dictionary = stop_words_removal(my_dictionary)
    my_dictionary = remove_less_than_k(my_dictionary)
    # Add 'unknown' word to represent all stopwords and removed words
    my_dictionary['#UNK#'] = 6000
    # Provide indexes for all words of dict
    vocab = set(my_dictionary.keys())
    my_dictionary = {word: i for i, word in enumerate(vocab)}
    
    return my_dictionary

#######################################################

    

