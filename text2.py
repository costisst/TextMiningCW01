# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 19:02:22 2020

@author: cstef
"""
import numpy as np
import pandas as pd
import re
import string
from numpy import array
import torch
import torch.nn as nn
import math
import time
import torch
import random
import torch.optim as optim

###############################################################################
###############################################################################
###############################################################################


def bag_of_words(question,glove_dict_words,embedding_maxtrix):
    
    bow_vec = 0
    words = question.split(' ')
    if ' ' in words:
        words = [w for w in words if w != ' ']
    if '' in words:
        words = [w for w in words if w != '']
    words = [w for w in words if w not in stopwords]
    words = ['#UNK#' if w not in glove_dict_words else w for w in words ]
    indexes = [glove_dict_words[w] for w in words]
    
    for index in indexes:
        
        bow_vec += embedding_maxtrix(torch.LongTensor([index])).cuda()
    
    if len(indexes) == 0:
        bow_vec =  torch.zeros([1, 300], dtype=torch.int32).cuda()
    else:
        bow_vec = (1/len(indexes))*bow_vec
        
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
    
def create_questions_tags(questions_and_tags):
    tags = []
    questions = []
    for q in questions_and_tags:
        
        # Split tag and question
        tag,question = q.split(' ',1)
        tag,_ = tag.split(':',1)
        
        # Remove punction from question and turn it to lowercase
        question = question.translate(str.maketrans('', '', string.punctuation)).strip().lower()
        
        # Add tag and question to corresponding list
        tags.append(tag)
        questions.append(question)    
                    
    return questions, tags

###############################################################################
###############################################################################
###############################################################################
    
# Load data 
stopwords = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/stopwords.txt')
temp_questions = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/questions.txt')
temp_questions.pop()
random.shuffle(temp_questions)
temp_questions = set(temp_questions)


# Create question and tag lists
questions,tags  = create_questions_tags(temp_questions)

# Create dictionary with pretrained vectors
glove = open('D:/Msc_AI_UoM/Semester 2/Text mining/glove.small.txt')
glove_dict = {}
for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    glove_dict[word] = coefs
glove.close()

# Tag dict
unique_tags = set(tags)
tag_dict = {word: i for i, word in enumerate(unique_tags)}

# Add indexes to words
glove_dict_words = glove_dict
glove_dict_words = {word: i for i, word in enumerate(glove_dict)}

# Indexes for pretrained vectors
pretrained_vectors = list(glove_dict.values())


weight = torch.FloatTensor(pretrained_vectors)
embedding_maxtrix = nn.Embedding.from_pretrained(weight) #Input for NN

# Create Training and Testing sets
training_questions = questions[:math.floor(0.9*len(questions))]
training_tags = tags[:math.floor(0.9*len(tags))]

testing_questions = questions[math.floor(0.9*len(questions)):]
testing_tags = tags[math.floor(0.9*len(tags)):]

# Create Neural Network
model = nn.Sequential(nn.Linear(300, 100),
                      nn.ReLU(), 
#                      nn.Linear(100, 100),
#                      nn.ReLU(),
#                      nn.Linear(500, 250),
#                      nn.Sigmoid(),
#                      nn.Linear(100, 50),
#                      nn.ReLU(),           
                      nn.Linear(100, 50),
                      nn.LogSoftmax(dim=1))

# Use cuda
model.cuda()

# Define the loss
criterion = nn.NLLLoss()

# Optimizers require the parameters to optimize and a learning rate
learning_rate = [0.0001,0.001,0.01,0.1,1,5,10]
optimizer = optim.Adam(model.parameters(), lr=0.00005)


test_score = []
train_score = []
epochs = 50
for e in range(epochs):
    running_loss = 0
    count_samples = 0
    count_correct = 0
    count_samples_test = 0
    count_correct_test = 0
    for question, tag in zip(training_questions,training_tags):
        
        count_samples += 1
        
        # Produce the input vector
        bow_vec = bag_of_words(question,glove_dict_words,embedding_maxtrix)
        
        # Produce target
        tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long).cuda()
          
        # Training pass
        optimizer.zero_grad()
        output = model(bow_vec.float())
        loss = criterion(output, tag_tens)
        
        if torch.eq(tag_tens, torch.exp(output).argmax()).cuda():
            count_correct += 1
#        print(loss)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Testing
    for question, tag in zip(testing_questions,testing_tags):
            
        count_samples_test += 1
        
        # Produce the input vector
        bow_vec = bag_of_words(question,glove_dict_words,embedding_maxtrix)
        
        # Produce target
        tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long).cuda()
          
        output = model(bow_vec.float())
    #        print(output)
        loss = criterion(output, tag_tens)
        
        if torch.eq(tag_tens, torch.exp(output).argmax()).cuda():
            count_correct_test += 1
            
        running_loss += loss.item()
    
    test_score.append([count_correct_test/count_samples_test])
    train_score.append([count_correct/count_samples])
    print("Epochs: {0}".format(e))    
    print("Training loss: {0}".format(count_correct/count_samples))
    print("Testing loss: {0}".format(count_correct_test/count_samples_test))   

