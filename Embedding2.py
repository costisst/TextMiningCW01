"""
Imports
"""
import numpy as np
import pandas as pd
import re
import gensim
import string
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy import array
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from numpy import dot, diag
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import KeyedVectors
#from torch.utils import data
from torch.utils.data import Dataset
"""
Staff to Do
-Freeze/ fine-tune
-Out of Vocabulary Words
"""

"""
Split of a Text
"""
def bow(text):
    text = clean_text(text).split()
    return text

"""
Bag of words Vec
"""
def vec(text, model,oov):
    temp = np.zeros((50))
    temp.fill(0)
    stopwords_file = open('C:/Users/gioek/Programming/NLP/stopwords.small.txt','r+') 
    stopwords_data = stopwords_file.read()
    stopwords = stopwords_data.split('\n')
    text= bow(text)
    text = [elem for elem in text if elem not in stopwords ]
    geia = np.random.rand(50)
    for i in text:
        if i not in oov:
            k = model.get_vector(i)
        else:
            k = geia
        temp = temp + k
    cal = temp/len(text)
    return cal.astype(np.float32)
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
Unknown words
"""
#def unknown(hey,text):
#    temp = np.zeros((50))
#    temp.fill(0)
#    stopwords_file = open('C:/Users/gioek/Programming/NLP/stopwords.small.txt','r+') 
#    stopwords_data = stopwords_file.read()
#    stopwords = stopwords_data.split('\n')
#    text= bow(text)
#    text = [elem for elem in text if elem not in stopwords ]
#    skata = []
#    for i in text:
#        if i not in hey:
#            skata.append(i)
#    return skata

"""
DataLoader
"""
class DatasetQ(Dataset):
    def __init__(self,final,Y):
        self.final = final
        self.Y = Y

    def __len__(self):
        return len(self.final)

    def __getitem__(self, idx):
        # Load data and get label
        X = self.final[idx]
        y = self.Y[idx]
#        print(X.shape)
#        print(X)
#        print(y.shape)
        return X, y



"""
Main
"""
file_name = "train_5500.label.txt"
file_nameY = "TREC_10.label.txt"
df = read_file(file_name)
dfY = read_file(file_nameY)
df["labels"] = df["label"] +':'+ df["sublabel"]
dfY["labels"] = dfY["label"] +':'+ dfY["sublabel"]
#text = clean_text(df['question'][0])
#textY = clean_text(dfY['question'][0])
lis = df['question']
lisY = dfY['question']
final = df['question']
finalY = dfY['question']
Y = df['labels']
YY = dfY['labels']
dictX = dic(df)
dictY = dic(dfY)
dictY.update(dictX) #oles tis lekseis tou training kai tou test
Y =Y.tolist()
YY = YY.tolist()

i = 0
for q in lis:
    final[i] = clean_text(lis[i]) # mono ta questions gia to training
    i +=1
i = 0
for q in lisY:
    finalY[i] = clean_text(lisY[i]) # mono ta questions gia to testing
    i +=1
hey = [] # EXEI TO VOCAB TOU GLOVE
with open('glove.6B.50d.new2.txt') as f:
    for i in range(0,8235):
        line = f.readline()
        yo = line.split()
        hey.append(yo[0])
model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/gioek/Programming/NLP/glove.6B.50d.new2.txt",binary=False)
##model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/gioek/Programming/NLP/glove.6B.50d.txt",binary=False)
#model = gensim.models.KeyedVectors.load_word2vec_format("C:/Users/gioek/Programming/NLP/glove.small.new.txt",binary=True)
#

#
#skata = []
#for i in final:  
#    temp = unknown(hey,i)
#    for j in range(0,len(temp)):
#        skata.append(temp[j])
#skata2 = []
#for i in finalY:  
#    temp = unknown(hey,i)
#    for j in range(0,len(temp)):
#        skata2.append(temp[j])
fskata = []
for i in dictY:
    if i not in hey:
        fskata.append(i)
        
#fskata = skata+skata2


kapa = []
q = 0
for i in final:
    if q!=3025 and q!=4749:
        kapa.append(vec(final[q],model,fskata)) # GIA TRAIN
    q +=1
    
kapaY = []
q = 0
for i in finalY:
    kapaY.append(vec(finalY[q],model,fskata)) # GIA TEST
    q +=1
    


from sklearn.preprocessing import LabelEncoder
Y.pop(3025)
Y.pop(4748)
values = array(Y)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
dataset = DatasetQ(kapa,integer_encoded) #GIA TRAIN
valuesY = array(YY)
label_encoderY = LabelEncoder()
integer_encodedY = label_encoder.fit_transform(valuesY)
datasetY = DatasetQ(kapaY,integer_encodedY) # GIA TEST
print(datasetY)

"""
Neural Network
"""
def create_nn(batch_size=10, learning_rate=1E-4, epochs=50,log_interval=10):

    train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size)
    
    test_loader = torch.utils.data.DataLoader(datasetY,batch_size=batch_size)
    
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(50, 5000)
            self.fc3 = nn.Linear(5000, 50)
        
        def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.fc3(x)
                return F.log_softmax(x,dim = 1)
    

    net = Net()
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    # create a loss function
    criterion = nn.NLLLoss()
    # run the main training loop
    test_loss = 0
    correct = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data = Variable(data)
            target = Variable(target)
#            print(data.data[0])
            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
#            
#            
#            pred = net_out.data.max(1)[1]  # get the index of the max log-probability
#            correct += pred.eq(target.data).sum()
            test_loss /= len(train_loader.dataset)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data))
#    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(train_loader.dataset),
#        100. * correct / len(train_loader.dataset)))
    
    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

create_nn()
