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

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden = nn.Linear(100, 75)
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(75, 50)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x

###############################################################################
###############################################################################
###############################################################################
def create_questions_tag_dict(questions_and_tags):
    tags = []
    questions = []
    my_dictionary = {}
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
                if word in my_dictionary:
                    my_dictionary[word] += 1
                else:
                    my_dictionary[word] = 1
                    
    return questions, tags, my_dictionary

###############################################################################
###############################################################################
###############################################################################

def bag_of_words(question,my_dictionary,embedding_maxtrix):
    
    bow_vec = 0
    words = question.split(' ')
    if ' ' in words:
        words.remove(' ')
    if '' in words:
        words.remove('')
    words = [w for w in words if w not in stopwords]
    words = ['unknown' if w not in my_dictionary else w for w in words ]
    indexes = [my_dictionary[w] for w in words]
    
    for index in indexes:
        
        bow_vec += embedding_maxtrix(torch.LongTensor([index]))
    
    if len(indexes) == 0:
        bow_vec =  torch.zeros([1, D], dtype=torch.int32)
    else:
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
D = 300

stopwords = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/stopwords.txt')
temp_questions = read_file('D:/Msc_AI_UoM/Semester 2/Text mining/questions.txt')
temp_questions.pop()

questions,tags ,my_dictionary = create_questions_tag_dict(temp_questions)

# Remove words with less than k appearences
my_dictionary = {key:value for (key,value) in my_dictionary.items() if value > k}

# Remove stopwords
my_dictionary = {key:value for (key,value) in my_dictionary.items() if key not in stopwords}

# Add 'unknown' word to represent all stopwords and removed words
my_dictionary['unknown'] = 6000 

# Get number of words in dict
N = len(my_dictionary.keys())

# Provide indexes for all words of dict
vocab = set(my_dictionary.keys())
my_dictionary = {word: i for i, word in enumerate(vocab)}

#  N words in vocab, D dimensional embeddings
embedding_maxtrix = nn.Embedding(N, D)  


# Produce the input vector
bow_vec = bag_of_words(questions[0],my_dictionary,embedding_maxtrix)

# Tag dict
unique_tags = set(tags)
tag_dict = {word: i for i, word in enumerate(unique_tags)}


model = nn.Sequential(nn.Linear(300, 100),
                      nn.ReLU(),
                      nn.Linear(100, 50),
                      nn.LogSoftmax(dim=1))
# Define the loss
criterion = nn.NLLLoss()
# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.003)

epochs = 5
for e in range(epochs):
    running_loss = 0
    for question, tag in zip(questions,tags):
        # Flatten MNIST images into a 784 long vector
        bow_vec = bag_of_words(question,my_dictionary,embedding_maxtrix)
        
        tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long)
          
        # Training pass
        optimizer.zero_grad()
        
        output = model(bow_vec.float())
        loss = criterion(output, tag_tens)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss/len(questions)}")