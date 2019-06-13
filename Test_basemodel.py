import pandas as pd
##### Load data with Pandas ######

filepath_dict = {'spam': 'dataset/spam.csv',
                 'ham': 'dataset/ham.csv'}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['symbols', 'label'], sep=';')
    df['source'] = source  #Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
##### label=0: ham, label=1_ spam #################
#print(df.iloc[0:10,:])

## randomize dataset
# frame = df.sample(frac=1).reset_index(drop=True)
# print(frame.iloc[0:10,:])

############### split the dataset in train and test dataset ##############

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['symbols'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=42)

# ##### Tokenize using CountVectorizer#######
    vectorizer = CountVectorizer()
    vectorizer.fit(sentences_train)
    vocab = vectorizer.vocabulary_
    X_train = vectorizer.transform(sentences_train)
    X_test = vectorizer.transform(sentences_test)
#    print(vocab)

####### here starts the keras_model #######
from keras.models import Sequential
from keras import layers

input_dim = X_train.shape[1]  # Numbers of features

model = Sequential()
model.add(layers.Dense(25, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
#TensorFlow as backend

### configuration of the learning process ###

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()  # gives an overview of the model

###### training with the .fit() function #######

history = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=10)

##### .evaluate() is used to measure the accuracy of the model ########

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy:  {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
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