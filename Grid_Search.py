import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# read data using pandas
sybls = pd.read_csv('dataset/SA/kiesel_symbols.csv')
trainf = pd.read_csv('dataset/Kiesel_Corpus/K_Train.csv',names=['symbols', 'label'], sep=';')
#valf = pd.read_csv('dataset/Kiesel_Corpus/K_Val.csv',names=['symbols', 'label'], sep=';')
# randomize dataset
trainframe = trainf.sample(frac=1).reset_index(drop=True)
#valframe = valf.sample(frac=1).reset_index(drop=True)
#create numpy arrays
symbols = sybls['Symbol'].values
sentences_train = trainframe['symbols'].values
y_train = trainframe['label'].values
#sentences_val = valframe['symbols'].values
#y_val = valframe['label'].values
#check if data has been read in correctly
#print("Symbols: ", symbols[0:5])
#print("Train Sentences: ", sentences_train[0:5])
#print("Y Values: ", y_train[0:5])

########## Tokenize using CountVectorizer ##############
from sklearn.feature_extraction.text import CountVectorizer

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
#X_val = vectorizer.transform(sentences_val)
# transform to binary numpy arrays
X_train = X_train.toarray()
#X_val = X_val.toarray()

# Function to create model, required for KerasClassifier
def create_model():
  # Create Model
  model = Sequential()
  model.add(Dropout(rate=0.2, input_shape=X_train.shape[1:]))
  model.add(Dense(units=100, activation='relu'))
  model.add(Dropout(rate=0.2))
  model.add(Dense(units=1, activation='sigmoid'))
  # Compile Model
  model.compile(loss='binary_crossentropy', optimizer= 'adam', metrics=['accuracy'])
  return model

#Create Model
model = KerasClassifier(build_fn=create_model, verbose=0)
# mini batch sizes:
batch_size = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
# number of epochs:
epochs = [10, 20, 40, 60, 80, 100]
# parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))