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
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable


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
    
    class CustomDataset(Dataset):
        def __init__(self, x_tensor, y_tensor):
            self.x = x_tensor
            self.y = y_tensor
        
        def __getitem__(self, index):
            return (self.x[index], self.y[index])

        def __len__(self):
            return len(self.x)    
    
    # Tag dict
    unique_tags = set(tags)
    tag_dict = {word: i for i, word in enumerate(unique_tags)}
    
    training_set = dict( zip(training_questions,training_tags ))
    
    
    bow_vec = []
    tag_tens = []
    for question, tag in zip(training_questions,training_tags):
        bow_vec.append(bag_of_words(question,my_dictionary,embedding_matrix)) 
        # Produce target
        tag_tens.append(torch.tensor([tag_dict[tag]], dtype=torch.long))
     
    bow_vec1 = []
    tag_tens1 = []
    for question, tag in zip(testing_questions,testing_tags):
        bow_vec1.append(bag_of_words(question,my_dictionary,embedding_matrix)) 
        # Produce target
        tag_tens1.append(torch.tensor([tag_dict[tag]], dtype=torch.long))

    train_dataset = CustomDataset(bow_vec,tag_tens)
    valid_dataset = CustomDataset(bow_vec1,tag_tens1)
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=50)
    test_loader = DataLoader(valid_dataset, batch_size=50)   
    
    test_loss = 0
    correct1 = 0
    correct = 0
    test_score = []
    train_score = []
    epochs = 200
    for e in range(epochs):
        correct1 = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            data = Variable(data)
            target = Variable(target)
            data = data.squeeze(1)
            target = target.squeeze(1)
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, target)
            test_loss += criterion(output, target).data
            pred1 = output.data.max(1)[1]  # get the index of the max log-probability
            correct1 += pred1.eq(target.data).sum()
            loss.backward()
            optimizer.step()
            
        test_loss /= len(train_loader.dataset)
        print(e)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct1, len(train_loader.dataset),100. * correct1 / len(train_loader.dataset)))
         # run a test loop
        test_loss = 0
        
        for data, target in test_loader:
            data, target = Variable(data), Variable(target)
            data = data.squeeze(1)
            target = target.squeeze(1)
            output = model(data.float())
             # sum up batch loss
            test_loss += criterion(output, target).data
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
                

            

    
