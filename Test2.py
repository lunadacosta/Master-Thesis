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

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=42)

#print(sentences_train)
#print(y_train)

##### Tokenize using Tokenizer API #######

from keras.preprocessing.text import Tokenizer

# create the tokenizer
t = Tokenizer(
    char_level=False,
    filters="!\"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n", # will prevent ' _ ' of being removed
    lower=False
)

# build the vocabulary on the sentences of the train set
t.fit_on_texts(sentences_train)

# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

# integer encode documents
train_X = t.texts_to_matrix(sentences_train, mode='binary')
print(train_X[25])

# ### Building the Model ####
# from keras.models import Sequential
# from keras.layers import Dense
#
# # create model
# model = Sequential()
#
# # get number of columns in training data
# n_cols = train_X.shape[1]
#
# #add model layers
# model.add(Dense(200, activation='relu', input_shape=(n_cols,)))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1))
#
# ### Compiling the model ####
# # compile model using mean_squared_error as a measure of model peformance
# model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()
#
# ### Training model ####
# from keras.callbacks import EarlyStopping
#
# #set early stopping monitor so the model stops training when it won't improve anymore
# esm = EarlyStopping(patience=3) # after 3 epochs in a row in which it doens't improve, training will stop
#
# #train model
# model.fit(train_X, y_train, validation_split=0.2, epochs=30, callbacks=[esm])