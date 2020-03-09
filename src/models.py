# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from preprocesing import read_file,build_dictionary,pre_process,clean_numbers


def create_questions_tag_dict(questions_and_tags,lowercase):
    tags = []
    questions = []
    my_dictionary = {}
    for q in questions_and_tags:
        
        # Split tag and question
        tag,question = q.split(' ',1)
        
        question = pre_process(question,lowercase)
        
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

def bag_of_words(question,my_dictionary):
    stopwords = read_file('../data/stopwords.txt')
    words = question.split(' ')
    if ' ' in words:
        words = [w for w in words if w != ' ']
    if '' in words:
        words = [w for w in words if w != '']
    words = [clean_numbers(w) for w in words]
    words = [w for w in words if w not in stopwords]
    words = ['#UNK#' if w not in my_dictionary else w for w in words ]
    indexes = [int(my_dictionary[w]) for w in words]
        
    return indexes

def pre_trained_dictionary():
    glove = open('../data/glove.small.txt')
    glove_dict = {}
    for line in glove:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float64')
        glove_dict[word] = coefs
    glove.close()
    # Add indexes to words
    glove_dict_words = glove_dict
    glove_dict_words = {word: i for i, word in enumerate(glove_dict)}
    return glove_dict,glove_dict_words

def create_train_dev_set(config):
    data = read_file("data/data.txt")
    # Remove the same examples
    data = set(data)
    data = list(data)

    # Create Training and Testing sets
    training_set = data[:math.floor(0.9*len(data))]

    dev_set = data[math.floor(0.9*len(data)):]
    
    with open('data/train.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in training_set)
    with open('data/dev.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in dev_set)
    
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_data(questions,tags):
    data = []
    for q,t in zip(questions,tags):
        ques = q.split()
        tues = [t for _ in range(len(ques))]
        d =  (ques, tues)
        data.append(d)
    return data

def prepare_tags_to_ix(unique_tags):
    tag_to_ix = {word: i for i, word in enumerate(unique_tags)}
    return tag_to_ix

def bilstm_training(config):
    # Only call this once to create your 
    #create_train_dev_set(config)

    # Red train and dev data from files
    training_data = read_file(config.path_train)
    training_data.remove('')
    dev_data = read_file(config.path_dev)
    dev_data.remove('') 
    
    # Create dictionary based on data
    questions,tags ,my_dictionary = create_questions_tag_dict(training_data,config.lowercase)

    EMBEDDING_DIM = config.word_embedding_dim
    HIDDEN_DIM = config.word_embedding_dim
    
    if config.pre_train == 'False':
        # Preprocess the dictionary
        my_dictionary = build_dictionary(my_dictionary)
        
        data = prepare_data(questions,tags)
        
        # Tag dict
        unique_tags = set(tags)
        
    else: 
        EMBEDDING_DIM = 300
        HIDDEN_DIM = 300
        
        # Create dictionary with pretrained vectors
        glove_dict,glove_dict_words = pre_trained_dictionary()
        
        # Tag dict
        unique_tags = set(tags)
        
        # Indexes for pretrained vectors
        pretrained_vectors = list(glove_dict.values())
        
        weights = torch.FloatTensor(pretrained_vectors)
        
        data = prepare_data(questions,tags)
        
        # Tag dict
        unique_tags = set(tags)
        

    word_to_ix = {}
    for sent, tags in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = prepare_tags_to_ix(unique_tags)
    training_data = data[:math.floor(0.9*len(data))]
    testing_data = data[math.floor(0.9*len(data)):]


    class LSTMTagger(nn.Module):
        # if pre_train = true --> vocab_size = weights
        def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
            '''Returns an biLSTM model.'''
            super(LSTMTagger, self).__init__()
            self.hidden_dim = hidden_dim
            if config.pre_train == 'False':
                self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            else:
                self.word_embeddings = nn.Embedding.from_pretrained(vocab_size)
    
            # The LSTM takes word embeddings as inputs, and outputs hidden states
            # with dimensionality hidden_dim.
            self.lstm = nn.LSTM(embedding_dim, hidden_dim,bidirectional=True)
    
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
    
        def forward(self, sentence):
            '''Passes the questions through the model'''
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
            return tag_scores

        def save_model(self):
            torch.save(self, "kostis.bow")
    
    if config.pre_train == 'False':
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    else:
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, weights, len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), config.lr_param)
    

    test_score = []
    train_score = []
    for epoch in range(config.epoch):  # again, normally you would NOT do 300 epochs, it is toy data
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
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
    
            # Step 3. Run our forward pass.
            tag_scores = model(sentence_in)
    
            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(tag_scores, targets)
            
            if torch.eq(targets[-1], torch.exp(tag_scores[-1]).argmax()):
                count_correct += 1
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation/Dev testing
        for sentence, tags in testing_data:
                
            count_samples_test += 1
            
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
              
            tag_scores = model(sentence_in)
            
            loss = loss_function(tag_scores, targets)
            
            if torch.eq(targets[-1], torch.exp(tag_scores[-1]).argmax()):
                count_correct_test += 1
                
            running_loss += loss.item()
        
        test_score.append([count_correct_test/count_samples_test])
        train_score.append([count_correct/count_samples])
        print("Epochs: {0}".format(epoch))    
        print("Training loss: {0}".format(count_correct/count_samples))
        print("Testing loss: {0}".format(count_correct_test/count_samples_test))
        
    model.save_model()
    
def bilstm_testing(path_test, path_model, path_eval_result, lowercase, pre_train, D):
    data = read_file(path_test)
    data = set(data)
    
    model = torch.load(path_model)
    model.eval()
    
    # Create dictionary based on data
    testing_questions,testing_tags ,my_dictionary = create_questions_tag_dict(data,lowercase)
    
    if pre_train == 'False':
        # Preprocess the dictionary
        my_dictionary = build_dictionary(my_dictionary)
        
    else:     
        # Create dictionary with pretrained vectors
        glove_dict,glove_dict_words = pre_trained_dictionary()
        
        # Indexes for pretrained vectors
        pretrained_vectors = list(glove_dict.values())
        
        weights = torch.FloatTensor(pretrained_vectors)
        
    data = prepare_data(testing_questions,testing_tags)
    
    # Tag dict
    unique_tags = set(tags)
    
        
    word_to_ix = {}
    for sent, tags in data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = prepare_tags_to_ix(unique_tags)


