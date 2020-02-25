"""
Imports
"""
import numpy as np
import pandas as pd
import re
import string
from numpy import array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from numpy import dot, diag
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
#from torch.utils import data
from torch.utils.data import Dataset
import codecs
import math

"""
Split of a Text
"""
def bow(text):
    text = clean_text(text).split()
    return text
    
"""
Bag of words Vec
"""
def vec(text, model,dictY):
    t=0
    temp = np.zeros((300))
    temp.fill(0)
    stopwords_file = open('C:/Users/gioek/Programming/NLP/stopwords.small.txt','r+') 
    stopwords_data = stopwords_file.read()
    stopwords = stopwords_data.split('\n')
    text= bow(text)
    text = [elem for elem in text if elem not in stopwords ]
    cal = 0
    if len(text)==0:
        return temp
    for i in text:
        p=0
        k=0
        for key, value in dictY.items():
            if key == i:
                t = p
            p+=1
        k = weights_matrix[t]
        temp = temp + k
    cal = temp/len(text)
    return cal
"""
Preprocessing
"""
def clean_text(text):
    """
    Applies some pre-processing on the given text.
    Steps :
    - Removing HTML tags
    - Removing punctuation
    - Lowering text
    """
    
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)    

    return text
"""
Read File
"""
def read_file(path):
    with open(path, 'rb') as f:
        txt = f.read()
    lines = txt.decode('latin').splitlines()
    df = []
    for idx, line in enumerate(lines):
        match = re.match('([A-Z]+)\:([a-z]+)[ ]+(.+)',line)
        df.append(match.groups())
    df = pd.DataFrame(df, columns = ['label', 'sublabel','question'])
    return df
"""
Create my dictionary
"""
def dic(df):
    questions = []
    my_dictionary = {}
    question = ''
    for q in range(0,len(df.index)):
        # Remove punction from question and turn it to lowercase
        question = clean_text(df['question'][q])
        question = question.translate(str.maketrans('', '', string.punctuation)).strip().lower()
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
    return my_dictionary

"""
Main
"""
file_name = "train_5500.label.txt"
df = read_file(file_name)
df["labels"] = df["label"] +':'+ df["sublabel"]
temp1 = df['question']
final = df['question']
Y = df['labels']
dicti = dic(df)
Y =Y.tolist()


i = 0
for q in temp1:
    final[i] = clean_text(temp1[i]) # mono ta questions gia to training
    i +=1

hey = []
the_file = open('glove.small.txt','r+') 
file_data = the_file.read()
file_data = file_data.split('\n')
the_file.close()
for i in file_data:
    yo = i.split()
    hey.append(yo[0]) # TIS LEKSEIS APO TO GLOVE


     
glove = codecs.open('C:/Users/gioek/Programming/NLP/glove.small.txt', encoding = 'utf-8')
glove_dict = {}
for line in glove:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float64')
    glove_dict[word] = coefs
glove.close()


matrix_len = len(dicti)
weights_matrix = np.zeros((matrix_len, 300))
words_found = 0

for i, word in enumerate(dicti):
    try: 
        weights_matrix[i] = glove_dict[word] #iNITIALIZE TA WEIGHTS GIA KATHE LEKSH 
        words_found += 1
    except KeyError:
        weights_matrix[i] = glove_dict['#UNK#']


q = 0
pipi =0
preTrainedVectors = []
for i in final:   
    if q!=3025 and q!=4749:     
        preTrainedVectors.append(vec(final[q],weights_matrix,dicti)) # GIA TRAIN
#    if vec(final[q],weights_matrix,dictY).all()==0: # 3025 4749
#        print(q)
    q+=1
    print(q)



from sklearn.preprocessing import LabelEncoder
Y.pop(3025)
Y.pop(4748)
values = array(Y)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)


weight = torch.FloatTensor(preTrainedVectors)
embedding = nn.Embedding.from_pretrained(weight) #Input for NN


tag_tens = torch.tensor(integer_encoded, dtype=torch.long)

# Create Training and Testing sets
training_questions = preTrainedVectors[:math.floor(0.9*len(preTrainedVectors))]
training_tags = integer_encoded[:math.floor(0.9*len(integer_encoded))]

testing_questions = preTrainedVectors[math.floor(0.9*len(preTrainedVectors)):]
testing_tags = integer_encoded[math.floor(0.9*len(integer_encoded)):]