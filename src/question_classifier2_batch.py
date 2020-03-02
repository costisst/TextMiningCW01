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
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


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

    class CustomDataset(Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
        
        def __getitem__(self, index):
            return (self.x[index], self.y[index])

        def __len__(self):
            return len(self.x)

    bow_vec = []
    tag_tens = []
    for question, tag in zip(training_questions,training_tags):
        bow_vec.append(bag_of_words(question,glove_dict_words,embedding_maxtrix)) 
        # Produce target
        tag_tens.append(torch.tensor([tag_dict[tag]], dtype=torch.long))
     
    bow_vec1 = []
    tag_tens1 = []
    for question, tag in zip(testing_questions,testing_tags):
        bow_vec1.append(bag_of_words(question,glove_dict_words,embedding_maxtrix)) 
        # Produce target
        tag_tens1.append(torch.tensor([tag_dict[tag]], dtype=torch.long))

    train_dataset = CustomDataset(bow_vec,tag_tens)
    valid_dataset = CustomDataset(bow_vec1,tag_tens1)
    
    
#    device = "cuda" if torch.cuda.is_available() else "cpu"
#


    
    # Create Neural Network
    model = nn.Sequential(nn.Linear(D, 1000),
                          nn.ReLU(),  
                          nn.Linear(1000, 200),
                          nn.ReLU(),
                          nn.Linear(200,50),
                          nn.LogSoftmax(dim=1))


    
    # Define the loss
#    criterion = nn.NLLLoss()
    
    # Optimizers require the parameters to optimize and a learning rate
    learning_rate = [0.0001,0.001,0.01,0.1,1,5,10]
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    
    train_loader = DataLoader(train_dataset, batch_size=50)
    test_loader = DataLoader(valid_dataset, batch_size=50)
    
    test_loss = 0
    correct = 0
    test_score = []
    train_score = []
    epochs = 60
    for e in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(target)
            data = data.squeeze(1)
            target = target.squeeze(1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            test_loss += criterion(output, target).data
            pred1 = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred1.eq(target.data).sum()
            loss.backward()
            optimizer.step()
            
        test_loss /= len(train_loader.dataset)
        print(e)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(train_loader.dataset),100. * correct / len(train_loader.dataset)))
         # run a test loop
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.squeeze(1)
            target = target.squeeze(1)
            output = model(data)
             # sum up batch loss
            test_loss += criterion(output, target).data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))