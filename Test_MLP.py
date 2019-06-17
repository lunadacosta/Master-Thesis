import pandas as pd

# read data using pandas
df = pd.read_csv('dataset/data_spam_ham.csv',names=['symbols', 'label'], sep=';')

#check if data has been read in correctly
#print(df.iloc[0:10,:])

## randomize dataset
frame = df.sample(frac=1).reset_index(drop=True)
#print(frame.iloc[0:10,:])

# ### split up the dataset into inputs and targets ####
# # create a dataframe with all training data (symbols) except the target column (label)
# train_X = frame.drop(columns=['label'])
#
# # check that the target variable has been removed
# print(train_X.head())
#
# ### insert the colum 'label' into our target variable (train_y) ####
# train_y = frame[['label']]
#
# # view dataframe
# print(train_y.head())

############## split the dataset in train and test dataset ##############
from sklearn.model_selection import train_test_split

sentences = frame['symbols'].values
y = frame['label'].values

sentences_train, sentences_val, y_train, y_val = train_test_split(sentences, y, test_size=0.25, random_state=42)

#print(sentences_train)
#print(y_train)

##### Tokenize using Tokenizer API #######
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

MAX_SEQUENCE_LENGTH = 300

# create the tokenizer
t = Tokenizer(
    char_level=False,
    filters="!\"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n", # will prevent ' _ ' of being removed
    lower=False
)

# build the vocabulary on the sentences of the training set
t.fit_on_texts(sentences_train)

# # summarize what was learned
# print(t.word_counts)
# print(t.document_count)
# print(t.word_index)
# print(t.word_docs)

# vectorize training and validation texts
x_train = t.texts_to_sequences(sentences_train)
x_val = t.texts_to_sequences(sentences_val)

# Get max sequence length
max_len = len(max(x_train, key=len))
if max_len > MAX_SEQUENCE_LENGTH:
    max_len = MAX_SEQUENCE_LENGTH

# Fix sequence length to max value. Sequences shorter than the length are
# padded in the beginning and sequences longer are truncated
# at the beginning
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_val = sequence.pad_sequences(x_val, maxlen=max_len)

print(x_train)
print(x_val)
print(t.word_index)

###### Building a simple MLP with some dropout layers for regularization
# (to prevent overfitting to training samples)#######
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout

model = models.Sequential()
model.add(Dropout(rate=0.2, input_shape=x_train.shape[1:]))  # rate: float, percentage of input to drop at Dropout layer.
                                                             # input_shape: tuple, shape of input to the model
model.add(Dense(units=150, activation='relu'))  # units: int, output dimension of the layers
model.add(Dropout(rate=0.2))
model.add(Dense(units=1, activation='sigmoid'))

#### Compiling the model ####

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()  # gives an overview of the model

#### Training model ####

history = model.fit(x_train, y_train, epochs=50, verbose=False, validation_data=(x_val, y_val), batch_size=200)

##### .evaluate() is used to measure the accuracy of the model ########

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_val, y_val, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

####### Visualization of the loss an the accuracy for the training and testing data set ############
import matplotlib.pyplot as plt

plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

plot_history(history)
plt.show()