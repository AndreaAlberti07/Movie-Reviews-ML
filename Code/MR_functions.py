import numpy as np
import matplotlib.pyplot as plt
import collections
import os
import pandas as pd
from porter import stem



'''The functions defined to BUILD A VOCABULARY are \textit{'remove punctuation'}, \textit{'read file'}, \textit{'dir words count'} and \textit{'create vocabulary'}. 
Since a version of the algorithm using the Porter Stemmer algorithm is proposed, the functions support the optional argument 
\textit{'stemming'} which can be set to \textit{'True'} to perform stemming.'''

'''The functions defined to EXTRACT THE FEATURES are \textit{'bag of words dir'} and
\textit{'bag of words dir'} that expands the previous one to the entire directory.'''

'''The function defined to train the classifier is \textit{'binary NBC training'} while the function \textit{'binary NBC inference'} is used to compute prediction.'''


###################### FUNCTIONS DEFINITION ######################


#Replace punctuation symbols with spaces.
def remove_punctuation(text):
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text

#Takes a file and return a list of the words inside it, removing punctuation and settings all words to lower().
def read_file(filename, stemming = False):
    with open(filename) as f:
        text=f.read()
    text = remove_punctuation(text.lower())
    words = [w for w in text.split() if len(w) > 2]
    if stemming == True:
        words = [stem(w) for w in words]
    return words

#takes a directory in input and provides a dictionary with the number of occurrences for each word inside the directory files.
def dir_words_count(dir_path, stemming = False):   
    cnt = collections.Counter()
    for filename in os.listdir(dir_path):
        words = read_file(dir_path + filename, stemming)
        cnt.update(words)
    return cnt     

#takes a counter and writes the number_words most frequently used in dest_filename. The words are written from the most used to the least used.
def create_vocabulary(cnt, n, dest_filename):
    with open(dest_filename, "w") as f:
        for word, count in sorted(cnt.most_common(n)):
            print(word, file = f)
    
            
#takes a vocabulary, takes a file and return the bag of words for that file with the label in last position
def bag_of_words(filename, vocname, label, stemming = False):
    vocabulary = read_file(vocname)
    bow_tmp = [0]*len(vocabulary)
    words = read_file(filename, stemming)
    for word in words:
        if word in vocabulary:
            index = vocabulary.index(word)
            bow_tmp[index] += 1
    bow_tmp.append(label)
    return bow_tmp

def bag_of_words_dir(pos_dirname, neg_dirname, vocname, stemming = False):
    #Negative reviews
    bow = []
    rev_list = os.listdir(neg_dirname)
    for rev in rev_list:
        bow_rev = bag_of_words(neg_dirname + rev, vocname, 0, stemming)
        bow.append(bow_rev) #list of lists. each list is referred to a single review

    #Positive reviews
    rev_list = os.listdir(pos_dirname)
    for rev in rev_list:
        bow_rev = bag_of_words(pos_dirname + rev, vocname, 1, stemming)
        bow.append(bow_rev) #list of lists. each list is referred to a single review
    
    bow = np.stack(bow) #creates array m x n (m = number of lists in the nested list bow_test; n = number of elements in each list)
    return bow

#takes a bow in input and outputs the probability distribution for words of the two classes separately
def binary_NBC_training(bow):
    X = bow[:,:-1]
    Y = bow[:,-1]
    
    parameters = {}
    
    #P(X|Y)
    pos_p= X[Y == 1, :].sum(0) + 1 #Laplacian smoothing
    pos_p = pos_p / pos_p.sum()
    parameters['pos_p'] = np.log(pos_p)
    neg_p = X[Y == 0, :].sum(0) + 1
    neg_p = neg_p / neg_p.sum()
    parameters['neg_p'] = np.log(neg_p)
    
    #P(Y)
    pos_prior = Y.mean()
    parameters['pos_prior'] = np.log(pos_prior)
    neg_prior = 1 - pos_prior
    parameters['neg_prior'] = np.log(neg_prior)

    return parameters

#takes as input a dictionary containing the needed parameters and the bow to classify and outputs the prediction
def binary_NBC_inference(bow_to_classify, parameters, list_scores = False):
    
    #crossing out the class label
    bow_to_classify = bow_to_classify[:,:-1]
    
    #computing scores
    pos_score = bow_to_classify @ parameters['pos_p'] + parameters['pos_prior']
    neg_score = bow_to_classify @ parameters['neg_p'] + parameters['neg_prior']
    
    prediction = pos_score > neg_score
    
    if list_scores == True:
        return prediction.astype(int), pos_score, neg_score
    else:
        return prediction.astype(int)   

#stores the file to_store with the filename and other specifications useful to identify the stored data
def store_results(to_store, filename, description, vocsize):
    with open(filename, mode='a') as file:
        file.write(vocsize + ' ' + description + ' ' + to_store + '\n')
        

#LOGISTIC REGRESSION FUNCTIONS

#Model Building
def logreg_inference(X, w, b):
    logits = (X @ w) + b
    probability = 1/(1+np.exp(-logits))
    return probability

#Loss Function
def cross_entropy(P, Y):
    P=np.clip(P, 0.0001, 0.9999)
    ce = (-Y*np.log(P) - (1 - Y)*np.log(1-P)).mean()
    return ce
        
#Training algorithm      
def logreg_training(X, Y, steps, lr, tol):
    m, n = X.shape

    w_theta = np.zeros(n)
    b_theta = 0
    
    train_accuracies = []
    losses = []
    ITC = 0
    
    for i in range(steps):
        P = logreg_inference(X, w_theta, b_theta)
        
        if (i % (steps/1000)) == 0:
            loss = cross_entropy(P,Y)
            prediction = (P > 0.5)
            accuracy = (prediction == Y).mean()
            train_accuracies.append(accuracy)
            losses.append(loss)
        
        b_theta = b_theta - lr * (P-Y).mean()
        w_theta = w_theta - lr * (X.T @ (P-Y)) / m
        
        #if the loss is smaller than a threshold stop the training (early-stopping to reduce overfitting risk)
        if len(losses) > 1 :
            delta_loss = abs(losses[-1] - losses[-2])           
            if delta_loss < tol:
                break
        
        
        #compute the number of iteration until convergence
        ITC += 1
    return w_theta, b_theta, train_accuracies, losses, ITC

