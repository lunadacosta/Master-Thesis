import pandas as pd

import time
import os
import psutil

from sklearn.feature_extraction.text import CountVectorizer

from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.optimizers import Adam

#import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

############# read data using pandas ####################################################
# sybls = pd.read_csv('dataset/SA/kiesel_symbols.csv')
# df = pd.read_csv('dataset/SA/kiesel_spam_ham.csv',names=['symbols', 'label'], sep=';')
#
# ## randomize dataset
# frame = df.sample(frac=1).reset_index(drop=True)

# ##### for separated validation and training sets #####
sybls = pd.read_csv('dataset/SA/kiesel_symbols.csv')
trainf = pd.read_csv('dataset/Kiesel_Corpus/K_Train.csv',names=['symbols', 'label'], sep=';')
valf = pd.read_csv('dataset/Kiesel_Corpus/K_Val.csv',names=['symbols', 'label'], sep=';')
# randomize dataset
#trainframe = trainf.sample(frac=1).reset_index(drop=True)
#valframe = valf.sample(frac=1).reset_index(drop=True)

#check if data has been read in correctly
#print(sybls.head())
#print(df.head())

#print(frame.iloc[0:10,:])

############## split the dataset in train and test dataset #################################
# from sklearn.model_selection import train_test_split
#
# symbols = sybls['Symbol'].values # returns a NumPy array instead of a Pandas Series object
# sentences = frame['symbols'].values
# y = frame['label'].values
# # split into train/test sets
# sentences_train, sentences_val, y_train, y_val = train_test_split(sentences, y, test_size=0.25, random_state=7)

#print(symbols[0:5])
#print(sentences)
#print(y)

########### If Dataset for training and testing are already separated ##########

symbols = sybls['Symbol'].values
sentences_train = trainf['symbols'].values
y_train = trainf['label'].values
sentences_val = valf['symbols'].values
y_val = valf['label'].values

# print(symbols[0:5])
# print(sentences_train[0:5])
# print(y_train[0:5])
# print(sentences_val[0:5])
# print(y_val[0:5])

########## Tokenize using CountVectorizer ##############

# create the transform
vectorizer = CountVectorizer()
# tokenize an create the vocabulary
vectorizer.fit(symbols)
#summarize
vocab = vectorizer.vocabulary_
#print("Vocabulary: ",vocab)
#print("Length of Vocab: ", len(vocab))
# encode document
X_train = vectorizer.transform(sentences_train)
X_val = vectorizer.transform(sentences_val)
# transform to binary numpy arrays
X_train = X_train.toarray()
X_val = X_val.toarray()
#summarize encoded training vector
#print(X_train)
#print("Train shape: ", X_train.shape)
#print("Val shape: ", X_val.shape)
#print("Train type: ",type(X_train))

###### Building a simple MLP with some dropout layers for regularization
#(to prevent overfitting to training samples)#######

# Function to create model
def create_model(learn_rate, activation, neurons):
    model = models.Sequential()
    model.add(Dropout(rate=0.2, input_shape=X_train.shape[1:]))  # rate: float, percentage of input to drop at Dropout layer.
                                                                 # input_shape: tuple, shape of input to the model: nr of features
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(rate=0.2))
    model.add(Dense(units=1, activation='sigmoid'))
    #### Compiling the model ####
    optimizer = Adam(lr=learn_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# define parameters:
batch_size = 25
epochs = 10
learn_rate = 0.1
activation = 'relu'
neurons = 50

model = create_model(learn_rate, activation, neurons)
#print(model.summary())  # gives an overview of the model

### start recording training running time ####
train_start_time = time.clock()
#### Training model ####

history = model.fit(X_train, y_train, epochs=epochs, verbose=False, validation_data=(X_val, y_val), batch_size=batch_size) #, class_weight={0: 1, 1: 0.2})

#### give final time for training ####
train_end_time = time.clock() - train_start_time
print("Training CPU Running Time: ",train_end_time, "seconds")

##### .evaluate() is used to measure the accuracy of the model ########

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Validation Accuracy:  {:.4f}".format(accuracy))

####### Visualization of the loss an the accuracy for the training
#and validation data set ############

# plt.style.use('ggplot')

# def plot_history(history):
#     acc = history.history['acc']
#     val_acc = history.history['val_acc']
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     x = range(1, len(acc) + 1)
#     # plot graphics
#     plt.figure(figsize=(12, 5))
#     # graphic for accuracy
#     plt.subplot(1, 2, 1)
#     plt.plot(x, acc, 'b', label='Training acc')
#     plt.plot(x, val_acc, 'r', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     # graphic for loss
#     plt.subplot(1, 2, 2)
#     plt.plot(x, loss, 'b', label='Training loss')
#     plt.plot(x, val_loss, 'r', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#
# plot_history(history)
# plt.show()

##### start recording prediction running time ####
pred_start_time = time.clock()

###### evaluation metrics #########
#roc curve and auc
#from sklearn.metrics import roc_curve

# predict probabilities
probs = model.predict_proba(X_val)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_val, probs)
print('AUC: %.3f' % auc)
# # calculate roc curve
# fpr, tpr, thresholds = roc_curve(y_val, probs)
# # plot no skill
# plt.plot([0, 1],[0, 1], linestyle='--')
# # plot the roc curve for the model
# plt.plot(fpr, tpr, marker='.')
# # show the plot
# plt.title('ROC Curve')
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.show()

####### confusion matrix #########

# predict probabilities for test set
yhat_probs = model.predict(X_val, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_val, verbose=0)
### Predictions are return in 2D array (one row for each example in the test dataset and one column for the prediction)
### scikit-learn metrics API expects a 1D array
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

### METRICS ###

# classification accuracy: (tp+tn)/Total
accuracy = accuracy_score(y_val, yhat_classes)
print('Classification Accuracy: %f' % accuracy)
# classification precision: tp / (tp + fp):
precision = precision_score(y_val, yhat_classes)
print('Precision: %f' % precision)
# Classification recall: tp / (tp + fn)
recall = recall_score(y_val, yhat_classes)
print('Recall: %f' % recall)
# F1 Score: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_val, yhat_classes)
print('F1 score: %f' % f1)
# confusion matrix
# matrix = confusion_matrix(y_val, yhat_classes)
# print(matrix)
tn, fp, fn, tp = confusion_matrix(y_val, yhat_classes).ravel()
print("TN: ", tn)
print("FP: ", fp)
print("FN: ", fn)
print("TP: ", tp)

##### give final prediction running time ####
pre_end_time = time.clock() - pred_start_time
print("Prediction CPU Running Time: ",pre_end_time, "seconds")

#### getting CPU consumption ####
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info().rss # in bytes
print('memory use: ', memoryUse)
print(psutil.virtual_memory())