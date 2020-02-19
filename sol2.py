import numpy as np
import pandas as pd
import re
import gensim
from numpy import dot, diag
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.externals import joblib 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import spacy
"""
-Ti kanw me to model?
-Allazw model?
-Freeze, fine-tune?
-Ti kanw me tis lekseis pou den uparxoun?
"""
#################################################################################################
#################################################################################################
def bow(text):
    text = clean_text(text).split()
    return text

#################################################################################################
#################################################################################################
def vec(text, model):
    temp = np.zeros((50))
    temp.fill(0)
    stopwords_file = open('C:/Users/gioek/Programming/NLP/stopwords.txt','r+') 
    stopwords_data = stopwords_file.read()
    stopwords = stopwords_data.split('\n')
    text= bow(text)
    text = [elem for elem in text if elem not in stopwords ]
    for i in text:
        k = model.get_vector(i)
#        k = model(i)
        temp += k
    cal = 1/len(text)*temp
    return cal  
#################################################################################################
#################################################################################################
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
#################################################################################################
#################################################################################################
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
#################################################################################################
#################################################################################################
"""
Prepare the data
"""
file_name = "train_5500.label.txt"
df = read_file(file_name)
temp = clean_text(df['question'][0])
model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/gioek/Programming/NLP/glove.6B.50d.txt", binary=False)
#temp2 = vec(temp,model)
