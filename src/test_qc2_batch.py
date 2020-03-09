#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 11:17:05 2020

@author: alexandrinio
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from preprocesing import read_file
from models import create_questions_tag_dict,bag_of_words,pre_trained_dictionary
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable



if __name__ == '__main__':
    # Load data 
    data = read_file("data/data.txt")
    random.Random(5).shuffle(data)
    data = set(data)
    D = 300
    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    device = torch.device("cpu")
    
    # Create question and tag lists
    questions,tags,_  = create_questions_tag_dict(data,True)

   # Create dictionary with pretrained vectors
    glove_dict,glove_dict_words = pre_trained_dictionary()
    
    # Get number of words in dict
    N = len(glove_dict_words.keys())
    
    # Tag dict
    unique_tags = set(tags)
    tag_dict = {word: i for i, word in enumerate(unique_tags)}
    
    # Indexes for pretrained vectors
    pretrained_vectors = list(glove_dict.values())
    
 
    # Create Training and Testing sets
    training_questions = questions[:math.floor(0.9*len(questions))]
    training_tags = tags[:math.floor(0.9*len(tags))]
    
    testing_questions = questions[math.floor(0.9*len(questions)):]
    testing_tags = tags[math.floor(0.9*len(tags)):]



    bow_vec = []
    tag_tens = []
    for question, tag in zip(training_questions,training_tags):
        
        vector = bag_of_words(question,glove_dict_words)
        bow_vec.append(vector)
        # Produce target
        tag_tens.append(tag_dict[tag])
     
    bow_vec_test = []
    tag_tens_test = []
    # TA idia me panw mono gia to testing set
    for question, tag in zip(testing_questions,testing_tags):
        vector = bag_of_words(question,glove_dict_words)
        bow_vec_test.append(vector)
        # Produce target
        tag_tens_test.append(tag_dict[tag])
    
    # Create Neural Network
    class FeedForwardNN(nn.Module):

        def __init__(self, pretrained_vectors):
            super(FeedForwardNN, self).__init__()
            self.weight = torch.FloatTensor(pretrained_vectors)
            self.embeddings = nn.Embedding.from_pretrained(self.weight, freeze = True)
            self.linear1 = nn.Linear(300, 100)
            self.linear2 = nn.Linear(100, 50)
    
        def forward(self, inputs , phase_type):
            bow_v = 0
            bow_vecs = []
#            bow_vecs1 = 0
            bow_vecs12 = 0
            if phase_type == 'train':
                
                for indexes in inputs:
                    
                    for index in indexes:
                
                        bow_v += self.embeddings(torch.LongTensor([index]).to(device))
                        
                    if len(inputs) == 0:
                        bow_v =  torch.zeros([1, D], dtype=torch.int32).to(device)
                    else:
                        bow_v = (1/len(indexes))*bow_v
                    bow_vecs.append(bow_v)
                bow_vecs12 = torch.cat(bow_vecs, 0)
                out = F.relu(self.linear1(bow_vecs12))
                
            else:
                for indexes in inputs:
                        
                    bow_v += self.embeddings(torch.LongTensor([indexes]).to(device))
                        
                if len(inputs) == 0:
                    bow_v =  torch.zeros([1, D], dtype=torch.int32).to(device)
                else:
                    bow_v = (1/len(inputs))*bow_v
                out = F.relu(self.linear1(bow_v))

            log_probs = F.log_softmax(self.linear2(out), dim=1)
            return log_probs
        
     
        
    

    # the model
    model = FeedForwardNN(pretrained_vectors).to(device)
    
    # Define the loss
    criterion = nn.NLLLoss()
    
    # Optimizers require the parameters to optimize and a learning rate
    learning_rate = [0.0001,0.001,0.01,0.1,1,5,10]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    
    test_loss = 0 
    test_score = []
    train_score = []
    epochs = 500
    batch_size = 1
    number_of_batches = math.ceil(len(bow_vec)/batch_size)
    for e in range(epochs):
        correct_train = 0
        correct_test = 0
        test_loss1 = 0 
        test_loss2 = 0
        model.train()
        for i in range(0, number_of_batches -1):
            
            question = bow_vec[ (i*batch_size) : ((i+1)*batch_size) ]
            target = tag_tens[ (i*batch_size) : ((i+1)*batch_size) ]
            target = torch.tensor(target, dtype=torch.long).to(device)
            
            model.zero_grad()
            output = model(question,'train')
            loss = criterion(output, target)
            test_loss1 += criterion(output, target)
            
            # get the index of the max log-probability
            pred1 = output.data.max(1)[1]  
            correct_train += pred1.eq(target.data).sum()
            loss.backward()
            optimizer.step()
        print(test_loss1)
#        print(model.embeddings(torch.LongTensor([29]).to(device)).data[0][0:5])
#        print(model.embeddings(torch.LongTensor([604]).to(device)).data[0][0:5])
#        print(model.embeddings(torch.LongTensor([3000]).to(device)).data[0][0:5])
#        print(model.embeddings(torch.LongTensor([6987]).to(device)).data[0][0:5])
        test_loss = test_loss1 / len(bow_vec)
        print(e)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct_train, len(bow_vec),100. * correct_train / len(bow_vec)))
         # run a test loop
        model.eval()
        with torch.no_grad():
            for test_question, test_target in zip(bow_vec_test,tag_tens_test):
                
                test_target = torch.tensor([test_target], dtype=torch.long).to(device)
                output = model(test_question, 'test')
                test_loss2 += criterion(output, test_target)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                if torch.eq(test_target, torch.exp(output).argmax()):
                     correct_test += 1
            test_loss = test_loss2 / len(bow_vec_test)
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss2, correct_test, len(bow_vec_test),100. * correct_test / len(bow_vec_test)))