# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:30:25 2020

@author: Konstantinos Stefanopoulos
"""

import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###############################################################################
###############################################################################
###############################################################################
def create_questions_tag_dict(questions_and_tags):
    tags = []
    questions = []
    suck_my_dict = {}
    for q in questions_and_tags:
        
        # Split tag and question
        tag,question = q.split(' ',1)
        
        # Remove punction from question and turn it to lowercase
        question = question.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        
        # Add tag and question to corresponding list
        tags.append(tag)
        questions.append(question)
        
        # Enrich dictionary with words from questions
        bag_of_words = question.split(' ')
        for word in bag_of_words: 
            word = word.strip()
            if word:
                if word in suck_my_dict:
                    suck_my_dict[word] += 1
                else:
                    suck_my_dict[word] = 1
                    
    return questions, tags, suck_my_dict

###############################################################################
###############################################################################
###############################################################################

def bag_of_words(question,suck_my_dict,embedding_maxtrix):
    
    bow_vec = 0
    words = question.split(' ')
    if ' ' in words:
        words.remove(' ')
    if '' in words:
        words.remove('')
    words = [w for w in words if w not in stopwords]
    words = ['unknown' if w not in suck_my_dict else w for w in words ]
    indexes = [suck_my_dict[w] for w in words]
    
    for index in indexes:
        
        bow_vec += embedding_maxtrix(torch.LongTensor([index]))
    
    bow_vec = (1/len(indexes))*bow_vec
    
#    context_idxs = torch.tensor(bow_vec, dtype=torch.long)
    return bow_vec

###############################################################################
###############################################################################
###############################################################################
    
def read_file(path):
    the_file = open(path,'r+') 
    file_data = the_file.read()
    file_data = file_data.split('\n')
    the_file.close()
    return file_data

###############################################################################
###############################################################################
###############################################################################

torch.manual_seed(1)
tags = []
questions = []
k = 2
D = 100

stopwords = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/stopwords.txt')
temp_questions = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/questions.txt')
temp_questions.pop()

questions,tags ,suck_my_dict = create_questions_tag_dict(temp_questions)

# Remove words with less than k appearences
suck_my_dict = {key:value for (key,value) in suck_my_dict.items() if value > k}

# Remove stopwords
suck_my_dict = {key:value for (key,value) in suck_my_dict.items() if key not in stopwords}

# Add 'unknown' word to represent all stopwords and removed words
suck_my_dict['unknown'] = 6000 

# Get number of words in dict
N = len(suck_my_dict.keys())

# Provide indexes for all words of dict
vocab = set(suck_my_dict.keys())
suck_my_dict = {word: i for i, word in enumerate(vocab)}

embedding_maxtrix = nn.Embedding(N, D)  #  N words in vocab, D dimensional embeddings

bow_vec = bag_of_words(questions[0],suck_my_dict,embedding_maxtrix)
    

