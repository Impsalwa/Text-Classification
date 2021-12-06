# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 12:30:50 2021

@author: Salwa
"""
import numpy as np 
import matplotlib.pyplot as plt
import string 
from sklearn.model_selection import train_test_split

input_files= [
    'edgar_allan_poe.txt',
    'robert_frost.txt',
    ]
#run in jupyter notebook to show the top head of the file 
#!head edgar_allan_poe.txt
#!head robert_frost.txt

#collect data into lists 
#start by creating two lists , one for texts or poem and other for labels 
input_texts =[]
labels =[]

#loop into each file, we use also enumerate to gives us index 
for label, f in enumerate(input_files):
    print (f"{f} corresponds to label {label}")
    for line in open(f):
        line = line.rstrip().lower()
        if line :
            #remove punctuation 
            line = line.translate(str.maketrans('','',string.punctuation))
            
            #append lines in text list and labels in labels list 
            
            input_texts.append(line)
            labels.append(line)
#do split train and test
train_text, test_text, ytrain, ytest = train_test_split(input_texts, labels)
#see how many lines we have 
print(len(ytrain))
print(len(ytest))
# print row from train text a randon ones 
print(train_text[:5])
#didn't get it 
print(ytrain[:5])

#convert our text into intgers 
idx = 1
word2idx = {'<unk>': 0}
#populate word2idx 
for text in train_text:
    tokens = text.split()
    for token in tokens:
        if token not in word2idx:
            word2idx[token] = idx
            idx += 1
print(word2idx)
print(len(word2idx))

#convert our data into intgers format
train_text_int =[]
test_text_int =[]
#loop in text 
for text in train_text:
    tokens = text.split()
    #map each token to its conrespond text and the result wil be a list of intger 
    line_as_int = [word2idx[token] for token in tokens]
    train_text_int.append(line_as_int)
#do the same with test set
for text in test_text:
    tokens =text.split()
    line_as_int = [word2idx.get(token, 0) for token in tokens]
    test_text_int.append(line_as_int)
print(train_text_int[100:105])

#build A and pi matricis 
#initialize  them for both classes 
#V is the vocabulary 
V =len(word2idx)

A0 =np.ones((V,V))
pi0=np.ones(V)

A1= np.ones((V,V))
pi1 = np.ones(V)

print(V)
print (A0, pi0)
print(A1, pi1)

#compute count for A and pi
def compute_count(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                #means the first word in the sentence 
                pi[idx] += 1
            else:
                #the last word existes , so count a transition
                A[last_idx, idx] += 1
            #update last idx 
            last_idx = idx 
            
#use the function 
print(compute_count([t for t, y in zip(train_text_int, ytrain) if  y==0], A0, pi0))
print(compute_count([t for t, y in zip(train_text_int, ytrain) if  y==1], A1, pi1))

#normlize A na pi so they are valid probability matrics 
#this is equivalent to the formulas shown before 

A0 /= A0.sum(axis=1, keepdims =True)
pi0 /= pi0.sum()
print(A0 ,pi0)
A1 /= A1.sum(axis=1, keepdims =True)
pi1 /= pi1.sum()
print(A1, pi1)

#log A nad pi since we don't need the actual probs 

logA0 = np.log(A0)
logpi0 = np.log(pi0)

logA1 = np.log(A1)
logpi1 = np.log(pi1)

#compute priors 

count0 = sum(y==0 for y in ytrain)
count1 = sum(y==1 for y in ytrain)
total = len(ytrain)
#compute prior probabilities 
p0 = count0 / total
p1 = count1 / total

#calculate logs 
logp0 = np.log(p0)
logp1 = np.log(p1)

print (p0, p1)

#build a classifier 
class classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.k = len(logpriors) #number of classes
    def _compute_log_likelihood(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis [class_]
         
        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                #its the first token 
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]
            
            #update last_idx 
            last_idx = idx
            
        return logprob

    #define the predict function 
    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            posteriors = [self._compute_log_likelihood(input_, c) + self.logpriors[c] \
                          for c in range(self.k)]
            pred = np.argmax(posteriors)
            predictions[i] = pred
        return predictions
#the hard work is over 
# each array must be in order since classes are assumed to index these lists
#initiate a classifier object 
clf = classifier([logA0, logA1], [logpi0, logpi1], [logp0, logp1])

#the training 
Ptrain = clf.predict(train_text_int)
print (f"train accuracy: {np.mean(Ptrain == ytrain)}")

#test set same thing 
Ptest= clf.predict(test_text_int)
print(f"test accuracy: {np.mean(Ptest == ytest)}")

#for the imbalance classes we call 
from sklearn.metrics import confusion_matrix, f1_score

#for the train
cm = confusion_matrix(ytrain, Ptrain)
print(cm)

f1_score(ytrain, Ptrain)
print(f1_score)


#for the test 
cm_test = confusion_matrix(ytest, Ptest)
print(cm_test)

f1_score(ytest, Ptest)
print(f1_score)

        




















        
    






























