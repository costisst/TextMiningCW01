# -*- coding: utf-8 -*-

from preprocesing import read_file,pre_process,clean_numbers
import torch
import numpy as np

def create_questions_tag_dict(questions_and_tags):
    tags = []
    questions = []
    my_dictionary = {}
    for q in questions_and_tags:
        
        # Split tag and question
        tag,question = q.split(' ',1)
        
        question = pre_process(question)
        
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

def bag_of_words(question,my_dictionary,embedding_maxtrix):
    stopwords = read_file('data/stopwords.txt')
    D = 300
    bow_vec = 0
    words = question.split(' ')
    if ' ' in words:
        words = [w for w in words if w != ' ']
    if '' in words:
        words = [w for w in words if w != '']
    words = [clean_numbers(w) for w in words]
    words = [w for w in words if w not in stopwords]
    words = ['#UNK#' if w not in my_dictionary else w for w in words ]
    indexes = [my_dictionary[w] for w in words]
    
    for index in indexes:
        
        bow_vec += embedding_maxtrix(torch.LongTensor([index]))
    
    if len(indexes) == 0:
        bow_vec =  torch.zeros([1, D], dtype=torch.int32)
    else:
        bow_vec = (1/len(indexes))*bow_vec
        
    return bow_vec

def pre_trained_dictionary():
    glove = open('data/glove.small.txt')
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


