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
print(df.iloc[0:10,:])
### randomize dataset
#frame = df.sample(frac=1).reset_index(drop=True)
#print(frame.iloc[0:10,:])

############## split the dataset in train and test dataset ##############

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

for source in df['source'].unique():
    df_source = df[df['source'] == source]
    sentences = df_source['symbols'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=42)

print(sentences_train)

# ##### Tokenize using CountVectorizer#######
#     vectorizer = CountVectorizer()
#     vectorizer.fit(sentences_train)
#     vocab = vectorizer.vocabulary_
#     # X_train = vectorizer.transform(sentences_train)
#     # X_test = vectorizer.transform(sentences_test)
#     print(vocab)


#print(vectorizer.vocabulary_)

#### Tokenize using Hashing_trick #######
    # from sklearn.feature_extraction.text import HashingVectorizer
    #
    # #estimate the size of the vocabulary
    # vocab_size = len(vectorizer.vocabulary_)
    # print("Vocab size: ",vocab_size)
    # #integer encode the document
    # vectorizer = HashingVectorizer(n_features=vocab_size*2)
    # #encode document
    # X_train = vectorizer.transform(sentences_train)
    # X_test = vectorizer.transform(sentences_test)
    #
    # print(X_train.shape)
    # print(X_train.toarray())
    # print(X_test.toarray())

##### Tokenize using Tokenizer API #######

from keras.preprocessing.text import Tokenizer

# create the tokenizer
t = Tokenizer(
    char_level=False,
    filters="!\"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n", # will prevent ' _ ' of being removed
    lower=False
)

# fit the tokenizer on the docs
t.fit_on_texts(sentences_train)

print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

# #### one-hot encoding with scikit-learn ######
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from numpy import argmax
#
# # integer encode
# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(sentences_train)
# print(integer_encoded)
#
# # binary encoded
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
# onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
# print(onehot_encoded)
#
# # invertz first 10 inputs
# inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0:10,:])])
# print(inverted)