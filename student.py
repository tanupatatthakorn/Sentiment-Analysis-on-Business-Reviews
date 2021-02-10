#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

""" STUDENT'S ANSWER: z5224734 z5281901
To begin with, the data was preprocessed by removing stop-words, capital letters, 
and removing non-alphanumeric characters. The processed review text was then passed 
down to 2 almost identical neural networks. The network starts with bidirectional 
LSTM architecture. This was chosen because LSTM can find the connection between 
strings of text which is helpful for determining the abstract concept of a sentence. 
The LSTM has 50 input nodes and 150 output nodes which were chosen from experimentation 
to be the best value and to make it learn more efficiently an additional Dropout function 
was used after LSTM with the probability of 0.15. The next layers for both neural networks 
then have to determine the category or rating which was done by fully connecting 150 nodes 
from LSTM to 64 nodes followed by ReLu activation. This was then connected to another fully 
connected layer. However, in this last layer, the rating neural network has only one output 
node with sigmoid activation which is different from the category NN with 5 output nodes. 
Additionally, since the output of category NN needed to answer which category the neural 
network thinks it reads, the Argmax function was used to select the output node with the 
highest activation value. The Argmax function was chosen because by keeping 5 output nodes 
expose to the loss function it was able to learn by back-propagating more easily in comparison 
to using only one output node and categorize by the value of activation, this is likely because 
each category not related to each other mathematically. Moreover, to be able to train these 2 NN 
together side by side the total lost function was calculated by addition of weighted rating output 
and category output with 30% and 70% weight respectively. By doing this the inaccuracy of category 
NN will have a larger effect on the loss function, this will encourage the network to mostly focus 
on minimizing the inaccuracy of category NN. The reason that this network should focus on category 
NN is that it was given a more difficult task than the rating NN since the category NN has to choose 
one category from 5 instead of polar yes-no questions of rating NN. 

Attempted Models
- CNN for category output: Failed to classify more than the standard lstm.
- Preprocessing by removing stop words and cleaning strings did not result in 
significant improvements
"""

import torch
import re
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
from sklearn import feature_extraction
import sys
from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def clean(s,lower=True):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    if lower:
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()
    else :
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE))

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    sample = clean(sample)

    return sample.split()

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # print('pre',sample)
    sample = remove_stopwords(sample)
    # print('remove_stopwords',sample)
    # sys.exit(1)

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=50)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    # Converts rating loss to value between 0 and 1
    ratingOutput = torch.round(torch.sigmoid(ratingOutput))
    # Returns the category with the highest value
    categoryOutput = categoryOutput.argmax(1)

    return ratingOutput.type(torch.LongTensor), categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        # For rating
        self.lstm = tnn.LSTM(50, 150, batch_first=True, num_layers=2, bidirectional=True, dropout=0.35)
        self.dropout = tnn.Dropout(p=0.15)
        self.l1 = tnn.Linear(150, 64)
        self.l2 = tnn.Linear(64, 1)
        # Similarly for category
        self.lstm_cat = tnn.LSTM(50, 150, batch_first=True, num_layers=2, bidirectional=True, dropout=0.35)
        self.dropout_cat = tnn.Dropout(p=0.15)
        self.l1_cat = tnn.Linear(150, 64)
        self.l2_cat = tnn.Linear(64, 5)

    def forward(self, input, length):
        # For rating
        rating = tnn.utils.rnn.pack_padded_sequence(input, length.cpu(), batch_first=True, enforce_sorted=False)
        # LSTM
        _, (rating, _) = self.lstm(rating)
        # Drop out to prevent overfitting
        rating = self.dropout(rating)
        # Fully connected layer 1
        rating = self.l1(rating[-2])
        rating = tnn.functional.relu(rating)
        # Output layer
        rating = self.l2(rating)
        # Squeeze output to one 
        rating = rating.squeeze(0).squeeze(-1)

        # Similarly for category but with 5 ouputs
        cat = tnn.utils.rnn.pack_padded_sequence(input, length.cpu(), batch_first=True, enforce_sorted=False)
        # LSTM
        _, (cat, _) = self.lstm_cat(cat)
        # Drop out to prevent overfitting
        cat = self.dropout_cat(cat)
        # Fully connected layer 1
        cat = self.l1_cat(cat[-2])
        cat = tnn.functional.relu(cat)
        # Output layer with multiple outputs (5)
        cat = self.l2_cat(cat)
        
        # Returing both rating and category outputs 
        return rating, cat

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # Cross Entropy for category classification
        self.cat_loss = tnn.CrossEntropyLoss()
        # Binary Cross Entropy for binary ratings
        self.rate_loss = tnn.BCEWithLogitsLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        c_loss = self.cat_loss(categoryOutput, categoryTarget)
        r_loss = self.rate_loss(ratingOutput.to(torch.float32), ratingTarget.to(torch.float32))
        # Weighted combination between categorical and rating losses
        return (0.3 * r_loss) + (0.7 * c_loss)

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
# Increased batch size to 64  
batchSize = 64
# 15 epochs, sufficient for training and does not overfit the model 
epochs = 15
# Increased the learning rate of SGD to 0.1 to avoid zero learning, very little overshoot
optimiser = toptim.SGD(net.parameters(), lr=0.1)