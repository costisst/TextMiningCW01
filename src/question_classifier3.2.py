# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F
from preprocesing import read_file,build_dictionary
from models import create_questions_tag_dict,pre_trained_dictionary

def prepare_sequence(seq, to_ix):
    seq = ['#UNK#' if w not in to_ix else w for w in seq.split() ]
    idxs = [to_ix[w] for w in seq]
    return idxs

def prepare_tags(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return np.array(idxs)

def prepare_data(questions,tags):
    data = []
    for q,t in zip(questions,tags):
        ques = q.split()
        tues = [t]
        d =  (ques, tues)
        data.append(d)
    return data

def prepare_tags_to_ix(unique_tags):
    tag_to_ix = {word: i for i, word in enumerate(unique_tags)}
    return tag_to_ix

def pad_features(reviews_int, seq_length):
    ''' Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(reviews_int), seq_length), dtype = int)

    
    for i, review in enumerate(reviews_int):
        review_len = len(review)
        
        if review_len <= seq_length:
            zeroes = list(np.zeros(seq_length-review_len))
            new = zeroes+review
        elif review_len > seq_length:
            new = review[0:seq_length]
        
        features[i,:] = np.array(new)
    
    return features
    

data = read_file("data/data.txt")
data = set(data)
# Create dictionary based on data
questions,tags ,my_dictionary = create_questions_tag_dict(data,'True')

EMBEDDING_DIM = 64
HIDDEN_DIM = 32
pre_train = 'False'

if pre_train == 'False':    
    # Preprocess the dictionary
    my_dictionary = build_dictionary(my_dictionary)
else:
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 300
    # Create dictionary with pretrained vectors
    glove_dict,glove_dict_words = pre_trained_dictionary()
    # Indexes for pretrained vectors
    pretrained_vectors = list(glove_dict.values())
    
    weights = torch.FloatTensor(pretrained_vectors)
    

# Get number of words in dict
N = len(my_dictionary.keys())

# Removing questions that are more than 15 length
questions_len = [len(x.split(' ')) for x in questions]
questions_int = [ i for i, l in enumerate(questions_len) if l<15 ]

# Keep the question/tags of questions_int indexes
questions = [questions[i] for i in questions_int]
tags = [tags[i] for i in questions_int]
tags = np.array(tags)

# Tranform questions to bow
questions = [prepare_sequence(q,my_dictionary) for q in questions]

questions = pad_features(questions,5)
# Tag dict
tag_to_ix = set(tags)
tag_to_ix = prepare_tags_to_ix(tag_to_ix)

tags = prepare_tags(tags,tag_to_ix)

train_X = questions[:math.floor(0.9*len(questions))]
train_X = train_X[:-1]
train_y = tags[:math.floor(0.9*len(tags))]
train_y = train_y[:-1]
valid_X = questions[math.floor(0.9*len(questions)):]
valid_X = valid_X[:-2]
valid_y = tags[math.floor(0.9*len(tags)):]
valid_y = valid_y[:-2]


# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(valid_X), torch.from_numpy(valid_y))

# dataloaders
batch_size = 10

# make sure to SHUFFLE your data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

class LSTMModel (nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers, output_dim):
        super(LSTMModel,self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim
        if pre_train == 'False':
            self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(vocab_size)
            
        # Number of hidden layers
        self.n_layers = n_layers
        
        # Building LSTM
        # batch_first= True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(embedding_dim,hidden_dim, n_layers, batch_first=True,bidirectional=True)
        
        # Linear layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.sftmx = nn.Softmax()
        
    def forward(self, questions):
        embeds = self.word_embeddings(questions)
        
        # Initialize hidden state with zeros
        h0 = Variable(torch.zeros(self.n_layers*2, questions.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.n_layers*2, questions.size(0), self.hidden_dim))
        
        out, (hn, cn) = self.lstm(embeds, (h0,c0))

        out = self.fc(out[:, -1, :])
        tag_scores = F.log_softmax(out, dim=1)
        return tag_scores
    
layer_dim = 1
output_dim = len(tag_to_ix)
vocab_size = len(my_dictionary)

if pre_train == 'False':
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, vocab_size, layer_dim, output_dim)
else:
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, weights, layer_dim, output_dim)

criterion = nn.NLLLoss()
learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

seq_dim = 5
num_epochs = 1000

for epoch in range(num_epochs):
    running_loss = 0
    # Calculate Accuracy         
    correct_training = 0
    total_training = 0
    count_samples_valid = 0
    count_correct_valid = 0
    for i, (questions, tags) in enumerate(train_loader):
        
        # Total number of labels
        total_training += tags.size(0)
        
        #questions = Variable(questions.view(-1, seq_dim, EMBEDDING_DIM))
        tags = Variable(tags)
            
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
        
        # Forward pass to get output/logits
        # outputs.size() --> 100, 10
        outputs = model(questions)
        
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        
#        print("Actual: {0} | Predicted: {1}".format(tags,predicted))
        
        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, tags)
        
        correct_training += (predicted == tags).sum()
        
        # Getting gradients w.r.t. parameters
        loss.backward()
        
        # Updating parameters
        optimizer.step()
        
        running_loss += loss.item()
        
        accuracy_training = 100 * correct_training/ total_training
        
    # Calculate Accuracy         
    correct_validation = 0
    total_validation = 0
    # Validation
    for i, (questions, tags) in enumerate(valid_loader):
    
        # Forward pass only to get logits/output
        outputs = model(questions)
        
        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)
        
        # Total number of labels
        total_validation += tags.size(0)
            
        running_loss += loss.item()
        
        correct_validation += (predicted == tags).sum()
        
        accuracy_validation = 100 * correct_validation/ total_validation
    
    print("Epochs: {0}".format(epoch))    
    print("Training loss: {0}".format(running_loss))
    print("Training Accuracy: {0}".format(accuracy_training))
    print("Validation Accuracy: {0}".format(accuracy_validation))
#        
#        if iter % 100 == 0:
#            # Calculate Accuracy         
#            correct = 0
#            total = 0
#            # Iterate through test dataset
#            for questions, tags in valid_loader:
#
#                #questions = Variable(questions.view(-1, seq_dim, EMBEDDING_DIM))
#                
#                # Forward pass only to get logits/output
#                outputs = model(questions)
#                
#                # Get predictions from the maximum value
#                _, predicted = torch.max(outputs.data, 1)
#                
#                # Total number of labels
#                total += tags.size(0)
#
#                correct += (predicted == tags).sum()
#            
#            accuracy = 100 * correct / total
#            
#
#            # Print Loss
#            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data, accuracy))
#        