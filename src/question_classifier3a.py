# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from preprocesing import read_file,build_dictionary
from models import create_questions_tag_dict,pre_trained_dictionary

def prepare_sequence(seq, to_ix):
    seq = ['#UNK#' if w not in to_ix else w for w in seq ]
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

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
    

data = read_file("data/data.txt")
data = set(data)
# Create dictionary based on data
questions,tags ,my_dictionary = create_questions_tag_dict(data,True)

EMBEDDING_DIM = 100
HIDDEN_DIM = 100
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

data = prepare_data(questions,tags)

# Tag dict
tag_to_ix = set(tags)
tag_to_ix = prepare_tags_to_ix(tag_to_ix)

training_data = data[:math.floor(0.9*len(data))]
testing_data = data[math.floor(0.9*len(data)):]


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        if pre_train == 'False':
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.word_embeddings = nn.Embedding.from_pretrained(vocab_size)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
    
if pre_train == 'False':
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(my_dictionary), len(tag_to_ix))
else:
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, weights, len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)


test_score = []
train_score = []
epochs = 20
for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    running_loss = 0
    count_samples = 0
    count_correct = 0
    count_samples_test = 0
    count_correct_test = 0
    for sentence, tags in training_data:
        count_samples += 1
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, my_dictionary)
        print(tags)
        target = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, target)
        
        if torch.eq(target, torch.exp(tag_scores[-1]).argmax()):
            count_correct += 1
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Testing
    for sentence, tags in testing_data:
            
        count_samples_test += 1
        
        sentence_in = prepare_sequence(sentence, my_dictionary)
        target = prepare_sequence(tags, tag_to_ix)
          
        tag_scores = model(sentence_in)
        
        loss = loss_function(tag_scores, target)
        
        if torch.eq(target, torch.exp(tag_scores[-1]).argmax()):
            count_correct_test += 1
            
        running_loss += loss.item()
    
    test_score.append([count_correct_test/count_samples_test])
    train_score.append([count_correct/count_samples])
    print("Epochs: {0}".format(epoch))    
    print("Training loss: {0}".format(count_correct/count_samples))
    print("Testing loss: {0}".format(count_correct_test/count_samples_test))