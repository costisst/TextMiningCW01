# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:15:41 2020

@author: Erebus
"""
import matplotlib.pyplot as plt

#correct_train.cpu().numpy() / len(bow_list)

def graph(train,test):
    plt.plot(train)
    plt.plot(test)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    
    
    
    
# GIA TA BATCH
    kap = []
    kap1 = []
    pepega = [5,10,15,20,25,30,35,40,45,50]
    for batch_size in pepega:
        number_of_batches = math.ceil(len(bow_list)/batch_size)
        model = FeedForwardNN(pretrained_vectors).to(device)
        criterion = nn.NLLLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#        for e in range(epochs):      
#            correct_train = 0
#            correct_test = 0
#            train_loss = 0 
#            test_loss = 0
#            # Training Phase
#            model.train()
#            for i in range(0, number_of_batches):
#                # Create question and label batches for training
#                questions_batch, target_batch = create_batch(bow_list, target_list, i, batch_size,number_of_batches)
#                # Turn indices of targets into tensor
#                target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)
#                model.zero_grad()
#                output = model(questions_batch,'train',batch_size)
#                loss = criterion(output, target_batch)
#                # Count total training phase loss
#                train_loss += criterion(output, target_batch)
#                # get the index of the max log-probability
#                prediction = output.data.max(1)[1] 
#                # Count correct guesses of NN
#                correct_train += prediction.eq(target_batch.data).sum()
#                loss.backward()
#                optimizer.step()
            yoyo = correct_train.cpu().numpy() / len(bow_list)
#            train_loss = train_loss / len(bow_list)
            
#            print('\nEpoch: ',e)
#            print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(train_loss, correct_train, len(bow_list),100. * correct_train / len(bow_list)))
#             # Testing Phase
#            model.eval()
#            with torch.no_grad():
#                for test_question, test_target in zip(bow_list_test,target_list_test):
#                    test_target = torch.tensor([test_target], dtype=torch.long).to(device)
#                    output = model(test_question, 'test',batch_size)
#                    test_loss += criterion(output, test_target)
#                    pred = output.data.max(1)[1]  # get the index of the max log-probability
#                    if torch.eq(test_target, pred):
#                         correct_test += 1
                test_loss = test_loss / len(bow_list_test)
                yoyo2 = correct_test / len(bow_list_test)
#                print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct_test, len(bow_list_test),100. * correct_test / len(bow_list_test)))
        kap.append(yoyo)
        kap1.append(yoyo2)