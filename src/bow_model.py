import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from preprocesing import read_file,build_dictionary,pre_process,clean_numbers
from models import create_questions_tag_dict,bag_of_words,pre_trained_dictionary

def create_batch(bow_list, target_list, i, number_of_batches, dictionary_size, config):
    questions_full_rank = [] 
    batch_size = config.batch_size
    # if last batch
    if i == (number_of_batches - 1):
        if batch_size == 1:
            target = [target_list[i]]
            question = [bow_list[i]]
        else:
            target = target_list[ (i*batch_size) : -1 ]
            question = bow_list[ (i*batch_size) : -1 ]  
    # Create batch according to batch size
    else:
        target = target_list[ (i*batch_size) : ((i+1)*batch_size) ]
        question = bow_list[ (i*batch_size) : ((i+1)*batch_size) ]  
    # Add padding if required
    if len(question) > 1:
        max_length = max(len(row) for row in question)
        for vector in question: 
            if config.pre_train == 'False':
                vector += [dictionary_size-1] * (max_length - len(vector))
            else:
                vector += [dictionary_size] * (max_length - len(vector))
            questions_full_rank.append(vector)
    else:
        questions_full_rank = question[0]
        
    return questions_full_rank,target

def train_model(number_of_batches, bow_list, target_list, bow_list_validation, target_list_validation, model, device , dictionary_size, config):       
    # Define the loss
    criterion = nn.NLLLoss()
    
    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.Adam(model.parameters(), config.lr_param)
    for e in range(config.epoch):      
        correct_train = 0
        correct_validation = 0
        train_loss = 0 
        validation_loss = 0
        # Training Phase
        model.train()
        for i in range(0, number_of_batches):
            # Create question and label batches for training
            questions_batch, target_batch = create_batch(bow_list, target_list, i, number_of_batches, dictionary_size, config)
            # Turn indices of targets into tensor
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)
            model.zero_grad()
            output = model(questions_batch,'train',config, device)
            loss = criterion(output, target_batch)
            # Count total training phase loss
            train_loss += criterion(output, target_batch)
            # get the index of the max log-probability
            prediction = output.data.max(1)[1] 
            # Count correct guesses of NN
            correct_train += prediction.eq(target_batch.data).sum()
            loss.backward()
            optimizer.step()
        # print(model.embeddings(torch.LongTensor([6987]).to(device)).data[0][0:5])
        train_loss = train_loss / len(bow_list)
        print('\nEpoch: ',e)
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, correct_train, len(bow_list),100. * correct_train / len(bow_list)))
         # Testing Phase
        model.eval()
        with torch.no_grad():
            for validation_question, validation_target in zip(bow_list_validation,target_list_validation):
                validation_target = torch.tensor([validation_target], dtype=torch.long).to(device)
                output = model(validation_question, 'validation',config,device)
                validation_loss += criterion(output, validation_target)
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                if torch.eq(validation_target, pred):
                     correct_validation += 1
            validation_loss = validation_loss / len(bow_list_validation)
            print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(validation_loss, correct_validation, len(bow_list_validation),100. * correct_validation / len(bow_list_validation)))

    torch.save(model, config.path_model)

def test_model(bow_list_test, target_list_test, device, tag_dict, config):
    # Define the loss
    criterion = nn.NLLLoss()

    model = torch.load(config.path_model)
    model.eval()

    test_loss = 0
    correct_test = 0
    true_label = []
    predicted_label = []
    model.eval()
    with torch.no_grad():
        for test_question, test_target in zip(bow_list_test,target_list_test):
            
            true_label.append(test_target)
            test_target = torch.tensor([test_target], dtype=torch.long).to(device)
            output = model(test_question, 'test',config, device)
            test_loss += criterion(output, test_target)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            predicted_label.append(pred.item())
            if torch.eq(test_target, pred):
                 correct_test += 1
        test_loss = test_loss / len(bow_list_test)
        print('\ntest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct_test, len(bow_list_test),100. * correct_test / len(bow_list_test)))
    metrics = confusion_matrix(tag_dict, true_label, predicted_label)
    return metrics

def bow(config, train_or_test):
    # Load data 
    # Only call this once to create your train,dev dataset
    #create_train_dev_set(config)

    # Red train and dev data from files
    train_data = read_file(config.path_train)
    train_data.remove('')
    dev_data = read_file(config.path_dev)
    dev_data.remove('') 
    test_data = read_file(config.path_test)
    test_data.remove('')
    
    #random.Random(5).shuffle(train_data)
    #random.Random(5).shuffle(dev_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    # Create questions and tags list and the vocabulary 
    questions_train, tags_train, vocabulary  = create_questions_tag_dict(train_data,config.lowercase)
    questions_dev, tags_dev, _  = create_questions_tag_dict(train_data,config.lowercase)
    questions_test, tags_test, _  = create_questions_tag_dict(test_data,config.lowercase)

    EMBEDDING_DIM = config.word_embedding_dim
    pretrained_vectors = []
    torch.manual_seed(5)
    
    if config.pre_train == 'False':
        # Preprocess the dictionary
        vocabulary = build_dictionary(vocabulary)
        vocabulary['Padding'] = len(vocabulary.keys())
    elif config.pre_train == 'True':
        EMBEDDING_DIM = 300
        # Create dictionary with pretrained vectors
        glove_dict,vocabulary = pre_trained_dictionary()
        # Indices for pretrained vectors
        pretrained_vectors = list(glove_dict.values())
        # Add vector of zeroes to represent padding in vocabulary
        vector_of_zeroes = np.zeros([EMBEDDING_DIM])
        pretrained_vectors.append(vector_of_zeroes)
        
        
    # Get number of words in dict
    N = len(vocabulary.keys())
    
    # Target dictionary, each target is assigned to a different number
    unique_tags = set(tags_train)
    tag_dict = {word: i for i, word in enumerate(unique_tags)}
 
    # Prepare data for Neural Network
    bow_list = []
    target_list = []
    for question, tag in zip(questions_train, tags_train):     
        # Bag of words on the dataset
        bow_vec = bag_of_words(question,vocabulary)
        bow_list.append(bow_vec)
        # Get target index value from target dictionary
        target_list.append(tag_dict[tag])
     
    bow_list_validation = []
    target_list_validation = []
    # Same for validation sets
    for question, tag in zip(questions_dev, tags_dev):
        bow_vec = bag_of_words(question,vocabulary)
        bow_list_validation.append(bow_vec)
        # Produce target
        target_list_validation.append(tag_dict[tag])

    bow_list_test = []
    target_list_test = []
    # Same for validation sets
    for question, tag in zip(questions_test, tags_test):
        bow_vec = bag_of_words(question,vocabulary)
        bow_list_test.append(bow_vec)
        # Produce target
        target_list_test.append(tag_dict[tag])  
        

    # Create model
    model = FeedForwardNN(pretrained_vectors, config, N).to(device)
    
    batch_size = config.batch_size
    number_of_batches = math.ceil(len(bow_list)/batch_size)

    if train_or_test == 'train':
        # Training
        train_model(number_of_batches, bow_list, target_list, bow_list_validation, target_list_validation, model, device, N, config)
    elif train_or_test == 'test':
        # Testing 
        metrics = test_model(bow_list_test, target_list_test, device,tag_dict, config)
        print(metrics)

def confusion_matrix(tag_dict, true_label, predicted_label):
    
    # reversed_dict = dict((v,k) for k,v in tag_dict.items())
    label_list  = zip(true_label,predicted_label)
    label_list = list(label_list)
    label_list.sort()
    my_list = [(0,0)]*50
    previous_index = label_list[0][0]
    first_tup = 0
    second_tup = 0
    for true_index,predicted_index in label_list:
        if true_index == previous_index:
            first_tup = my_list[true_index][0] + 1
            if predicted_index == true_index:
                second_tup = my_list[true_index][1] + 1
        else:
            first_tup = 0
            second_tup = 0
            first_tup = my_list[true_index][0] + 1
            if predicted_index == true_index:
                second_tup = my_list[true_index][1] + 1
        my_list[true_index] = (first_tup,second_tup)        
        previous_index = true_index    
            
            
    confusion_matrix = np.zeros([50,50], dtype=int)
    for a,b in label_list:
        if a==b:
            confusion_matrix[a][a] += 1
        else:
            confusion_matrix[a][b] += 1        
            
    metrics = np.zeros([50,3])
    for i in range(0,50):
        
        if confusion_matrix[i][i] !=0 :
            recall = confusion_matrix[i][i] / (confusion_matrix[i][i] + (confusion_matrix[i,:].sum()-confusion_matrix[i][i]))
            precision = confusion_matrix[i][i] / (confusion_matrix[i][i] + (confusion_matrix[:,i].sum()-confusion_matrix[i][i]))
            f1_score = 2*((recall*precision)/(recall+precision))
        else:
            recall = 0
            precision = 0
            f1_score = 0
        
        metrics[i][0] = recall
        metrics[i][1] = precision
        metrics[i][2] = f1_score
        
    return metrics

# Neural Network
class FeedForwardNN(nn.Module):

    def __init__(self, pretrained_vectors,config,N):
        super(FeedForwardNN, self).__init__()
        if config.pre_train == 'False':
            self.embeddings = nn.Embedding(N,config.word_embedding_dim)
        elif config.pre_train == 'True':
            self.weight = torch.FloatTensor(pretrained_vectors)
            self.embeddings = nn.Embedding.from_pretrained(self.weight, freeze = False)
        self.linear1 = nn.Linear(config.word_embedding_dim, 100)
        self.linear2 = nn.Linear(100, 50)
        self.D = config.word_embedding_dim

    def forward(self, index_list , phase_type, config, device):
        bow_v = 0
        if phase_type == 'train':
            if config.batch_size == 1:  
                bow_v = self.create_nn_input_vector(index_list,config,device)                                  
            else:
                bow_v = self.embeddings(torch.LongTensor(index_list).to(device))
                bow_v = torch.mean(bow_v,dim=1)
        else:
            bow_v = self.create_nn_input_vector(index_list,config,device)
        out = F.relu(self.linear1(bow_v)) 
        log_probs = F.log_softmax(self.linear2(out), dim=1)
        return log_probs
    
    def create_nn_input_vector(self, index_list,config,device):
        
        bow_v = 0
        if len(index_list) == 0:
            bow_v =  torch.zeros([1, config.word_embedding_dim], dtype=torch.int32).to(device)
        else:
            for indices in index_list:                          
                bow_v += self.embeddings(torch.LongTensor([indices]).to(device))
            bow_v = (1/len(index_list))*bow_v
        return bow_v
