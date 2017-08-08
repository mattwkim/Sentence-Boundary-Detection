#Name: Matt Kim
#uniquename: mattwkim

import numpy as np
from sklearn import tree
import sys
#Send all print statements to a SBD.test.out file
sys.stdout = open('SBD.test.out', 'w')

#Prepare vocab list and the container of 'NEOS'/'EOS' labels
myfile = open('SBD.train', 'r')
vocab = []
labelvectors = []
for line in myfile:
    if 'EOS' in line:
        labelvectors.append(line.split()[-1])
    word = line.split()[-2]
    if word not in vocab:
        vocab.append(word)
myfile.close()
myfile = open('SBD.test', 'r')
for line in myfile:
    word = line.split()[-2]
    if word not in vocab:
        vocab.append(word)
vocab = sorted(set(vocab))

featurevectors = []
#Initialize a container of 0's that match up index wise to the vocab list
vocabint = [0 for word in vocab]

#Open and parse train file
myfile = open('SBD.train', 'r').read().split('\n')
myfile = filter(lambda x: x != "", [line.strip() for line in myfile])

#Create the train feature vectors
for i in range(0, len(myfile)):
    #Because I want vocabint to stay as a container of 0's, create a copy
    copy = vocabint[:]
    featurevector = []
    if 'EOS' in myfile[i]:
        #Find if L word exists in vocab
        index = vocab.index(myfile[i].split()[-2])
        #Change the index in the container of 0 that matches the index
        #of the word in the vocab to a 1
        copy[index] = 1
        #Copy container represents onehotencoding(L). Put into feature vector
        featurevector.extend(copy)
        
        #Check if a R word exists
        if (i + 1) <= (len(myfile) - 1):
            #Follow same procedure for one hot encoding and encode R
            copy = vocabint[:]
            rightindex = vocab.index(myfile[i + 1].split()[-2])
            copy[index] = 1
            featurevector.extend(copy)
        #If R doesn't exist, put a one hot encoding of a blank string
        #into the feature vector. It represents: onehotencoding(" ").
        else:
            featurevector.extend(vocabint)
            
        #Check if left is less than 3 chars   
        if len(myfile[i].split()[-2]) < 3:
            featurevector.append(1)
        else:
            featurevector.append(0)
            
        #Check if left first letter is uppercase
        if myfile[i].split()[-2][0].isupper():
            featurevector.append(1)
        else:
            featurevector.append(0)
        
        #Check if there is a R word
        if (i + 1) <= (len(myfile) - 1):
            #If R exists, check if first letter is uppercase
            if myfile[i + 1].split()[-2][0].isupper():
                featurevector.append(1)
            else:
                featurevector.append(0)
        #If R doesn't exist, append a 0
        else:
            featurevector.append(0)
            
        #Check if L has multiple periods
        if myfile[i].split()[-2][0].count('.') > 1:
            featurevector.append(1)
        else:
            featurevector.append(0)
        #Check if L has numbers
        if any(char.isdigit() for char in myfile[i].split()[-2][0]):
            featurevector.append(1)
        else:
            featurevector.append(0)
            
        #Check if L has quotations
        if '"' in myfile[i].split()[-2]:
            featurevector.append(1)
        else:
            featurevector.append(0)
        featurevectors.append(featurevector)
        
#Delete container of lines from the train file to save memory
del myfile

#Create and train/fit decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(featurevectors, labelvectors)

#Will create new featurevectors for test, so delete the one for train
del featurevectors
featurevectors = []

#Open and parse test file
myfile = open('SBD.test', 'r').read().split('\n')
myfile = filter(lambda x: x != "", [line.strip() for line in myfile])

#Create test feature vectors to use for predictions against the tree
for i in range(0, len(myfile)):
    #Because I want vocabint to stay as a container of 0's, create a copy
    copy = vocabint[:]
    featurevector = []
    if 'EOS' in myfile[i]:
        #Find if L word exists in vocab
        index = vocab.index(myfile[i].split()[-2])
        #Change the index in the container of 0 that matches the index
        #of the word in the vocab to a 1
        copy[index] = 1
        #Copy container represents onehotencoding(L). Put into feature vector
        featurevector.extend(copy)
        
        #Check if a R word exists
        if (i + 1) <= (len(myfile) - 1):
            #Follow same procedure for one hot encoding and encode R
            copy = vocabint[:]
            rightindex = vocab.index(myfile[i + 1].split()[-2])
            copy[index] = 1
            featurevector.extend(copy)
        #If R doesn't exist, put a one hot encoding of a blank string
        #into the feature vector. It represents: onehotencoding(" ").
        else:
            featurevector.extend(vocabint)
            
        #Check if left is less than 3 chars   
        if len(myfile[i].split()[-2]) < 3:
            featurevector.append(1)
        else:
            featurevector.append(0)
            
        #Check if left first letter is uppercase
        if myfile[i].split()[-2][0].isupper():
            featurevector.append(1)
        else:
            featurevector.append(0)
        
        #Check if there is a R word
        if (i + 1) <= (len(myfile) - 1):
            #If R exists, check if first letter is uppercase
            if myfile[i + 1].split()[-2][0].isupper():
                featurevector.append(1)
            else:
                featurevector.append(0)
        #If R doesn't exist, append a 0
        else:
            featurevector.append(0)
            
        #Check if L has multiple periods
        if myfile[i].split()[-2][0].count('.') > 1:
            featurevector.append(1)
        else:
            featurevector.append(0)
        #Check if L has numbers
        if any(char.isdigit() for char in myfile[i].split()[-2][0]):
            featurevector.append(1)
        else:
            featurevector.append(0)
            
        #Check if L has quotations
        if '"' in myfile[i].split()[-2]:
            featurevector.append(1)
        else:
            featurevector.append(0)
        featurevectors.append(featurevector)

#Since feature vectors have been created for both, these arent necessary
del vocab
del vocabint


predictions = clf.predict(featurevectors)

#Keep track of how many EOS/NEOS encountered
eosneosindex = 0

#Divide at the end to get the accuracy percentage
totalnumofeosneos = 0

#Total number of matched predictions
score = 0

#Parse all of the predictions into a list
predictions = [prediction for prediction in predictions]

#Create the SBD.test.out file
#All prints are redirected to SBD.test.out
for line in open('SBD.test', 'r'):
    if 'EOS' in line:
        #Check if matched prediction and update num of matches
        if line.split()[2] == predictions[eosneosindex]:
            score += 1
        line = line.split()
        
        #Replace neos/eos label from test with prediction
        line[2] = predictions[eosneosindex]
        line = line[0] + " " + line[1] + " " + line[2]
        
        #Update # of eos/neos encountered
        eosneosindex += 1

        #Update total number of eos/neos in the test file
        totalnumofeosneos += 1

    #Fancy way of printing and removes the '\n' from the lines with TOK
    sys.stdout.write(line)

#Flush all output and send it to SBD.test.out
sys.stdout.flush()

#Officially done with creating feature vectors, so delete it for memory
del featurevectors

#Divide total number of matches by the number of EOS/NEOS labels in train
score /= float(totalnumofeosneos)


print "Accuracy for all 8 features: " + str(score)

