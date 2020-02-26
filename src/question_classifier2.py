#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:17:05 2020

@author: alexandrinio
"""
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from preprocesing import read_file
from models import create_questions_tag_dict,bag_of_words,pre_trained_dictionary



if __name__ == '__main__':
    # Load data 
    data = read_file("data/data.txt")
    random.shuffle(data)
    data = set(data)
    D = 300
    
    # Create question and tag lists
    questions,tags,_  = create_questions_tag_dict(data)

   # Create dictionary with pretrained vectors
    glove_dict,glove_dict_words = pre_trained_dictionary()
    
    # Tag dict
    unique_tags = set(tags)
    tag_dict = {word: i for i, word in enumerate(unique_tags)}
    
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
    model = nn.Sequential(nn.Linear(D, 600),
                          nn.ReLU(),  
                          nn.Linear(600, 200),
                          nn.ReLU(),
                          nn.Linear(200,50),
                          nn.LogSoftmax(dim=1))

    
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
            tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long)
              
            # Training pass
            optimizer.zero_grad()
            output = model(bow_vec.float())
            loss = criterion(output, tag_tens)
            
            if torch.eq(tag_tens, torch.exp(output).argmax()):
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
            tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long)
              
            output = model(bow_vec.float())
        #        print(output)
            loss = criterion(output, tag_tens)
            
            if torch.eq(tag_tens, torch.exp(output).argmax()):
                count_correct_test += 1
                
            running_loss += loss.item()
        
        test_score.append([count_correct_test/count_samples_test])
        train_score.append([count_correct/count_samples])
        print("Epochs: {0}".format(e))    
        print("Training loss: {0}".format(count_correct/count_samples))
        print("Testing loss: {0}".format(count_correct_test/count_samples_test))   