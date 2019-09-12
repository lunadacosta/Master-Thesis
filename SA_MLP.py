import pandas as pd

############# read data using pandas ##############

filepath_dict = {'spam': 'dataset/SA/july_S.csv',
                 'ham': 'dataset/SA/july_H.csv'}

df_list = []

for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep=';')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
frame = df.sample(frac=1).reset_index(drop=True)
#print(frame.iloc[0:10,:])

############## split the dataset in train and test dataset ##############
from sklearn.model_selection import train_test_split

for source in frame['source'].unique():
    df_source = frame[frame['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

# split into train/test sets
sentences_train, sentences_val, y_train, y_val = train_test_split(sentences, y, test_size=0.25, random_state=42)

########## Tokenize using CountVectorizer ##############
from sklearn.feature_extraction.text import CountVectorizer

# create the transform
vectorizer = CountVectorizer()
# tokenize an create the vocabulary
vectorizer.fit(sentences_train)
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
#print(X_train.iloc[0:10,:])
#print("Train shape: ", X_train.shape)
#print("Val shape: ", X_val.shape)
#print("Train type: ",type(X_train))

###### Building a simple MLP with some dropout layers for regularization
#(to prevent overfitting to training samples)#######
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

model = models.Sequential()
model.add(Dropout(rate=0.2, input_shape=X_train.shape[1:]))  # rate: float, percentage of input to drop at Dropout layer.
                                                            # input_shape: tuple, shape of input to the model: nr of features
model.add(Dense(units=100, activation='relu'))  # units: int, output dimension of the layers
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))

#### Compiling the model ####

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.summary()  # gives an overview of the model

#### Training model ####

history = model.fit(X_train, y_train, epochs=50, verbose=False, validation_data=(X_val, y_val), batch_size=200)

##### .evaluate() is used to measure the accuracy of the model ########

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print("Validation Accuracy:  {:.4f}".format(accuracy))

####### Visualization of the loss an the accuracy for the training
#and validation data set ############
import matplotlib.pyplot as plt

plt.style.use('ggplot')

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

###### evaluation metrics #########
#roc curve and auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model.predict_proba(X_val)
# keep probabilities for the positive outcome only
#probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(y_val, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(y_val, probs)
# plot no skill
plt.plot([0, 1],[0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
# show the plot
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.show()

####### confusion matrix #########
from sklearn.metrics import confusion_matrix

# predict probabilities for test set
yhat_probs = model.predict(X_val, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_val, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

matrix = confusion_matrix(y_val, yhat_classes)
print(matrix)