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
from preprocesing import read_file,build_dictionary
from models import create_questions_tag_dict,bag_of_words


if __name__ == '__main__':
    data = read_file("data/data.txt")
    data = set(data)
    D = 300
    
    # Create dictionary based on data
    questions,tags ,my_dictionary = create_questions_tag_dict(data)

    # Preprocess the dictionary
    my_dictionary = build_dictionary(my_dictionary)
    
    # Get number of words in dict
    N = len(my_dictionary.keys())
    
    #  N words in vocab, D dimensional embeddings
    embedding_matrix = nn.Embedding(N, D)
    
    '''
    lookup_tensor = torch.tensor([my_dictionary["woman"]], dtype=torch.long)
    hello_embed = embedding_matrix(lookup_tensor)
    print(hello_embed)
    '''
    
    # Create Training and Testing sets
    training_questions = questions[:math.floor(0.9*len(questions))]
    training_tags = tags[:math.floor(0.9*len(tags))]

    testing_questions = questions[math.floor(0.9*len(questions)):]
    testing_tags = tags[math.floor(0.9*len(tags)):]
    
    # Tag dict
    unique_tags = set(tags)
    tag_dict = {word: i for i, word in enumerate(unique_tags)}
    
    training_set = dict( zip(training_questions,training_tags ))
    
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
    optimizer = optim.Adam(model.parameters(), lr=0.00005  )
    
    
    test_score = []
    train_score = []
    epochs = 25
    for e in range(epochs):
        running_loss = 0
        count_samples = 0
        count_correct = 0
        count_samples_test = 0
        count_correct_test = 0
        for question, tag in zip(training_questions,training_tags):
            
            count_samples += 1
            
            # Produce the input vector
            bow_vec = bag_of_words(question,my_dictionary,embedding_matrix)
            
            # Produce target
            tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long)
              
            # Training pass
            optimizer.zero_grad()
            output = model(bow_vec.float())
            loss = criterion(output, tag_tens)
            
            if torch.eq(tag_tens, torch.exp(output).argmax()):
                count_correct += 1

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Testing
        for question, tag in zip(testing_questions,testing_tags):
                
            count_samples_test += 1
            
            # Produce the input vector
            bow_vec = bag_of_words(question,my_dictionary,embedding_matrix)
            
            # Produce target
            tag_tens = torch.tensor([tag_dict[tag]], dtype=torch.long)
              
            output = model(bow_vec.float())
            
            loss = criterion(output, tag_tens)
            
            if torch.eq(tag_tens, torch.exp(output).argmax()):
                count_correct_test += 1
                
            running_loss += loss.item()
        
        test_score.append([count_correct_test/count_samples_test])
        train_score.append([count_correct/count_samples])
        print("Epochs: {0}".format(e))    
        print("Training loss: {0}".format(count_correct/count_samples))
        print("Testing loss: {0}".format(count_correct_test/count_samples_test))
    
