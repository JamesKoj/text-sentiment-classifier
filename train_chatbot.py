import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

words=[]
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
    for pattern in intent['patterns']:

        # take each word and tokenize it
        pattern_phrases = nltk.word_tokenize(pattern)

        # take each 2-gram and tokenize it
        twograms = ngrams(pattern.split(), 2)
        for phrase in twograms:
            pattern_phrases.append(phrase[0] + " " + phrase[1])

        # take each 3-gram and tokenize it
        threegrams = ngrams(pattern.split(), 3)
        for phrase in threegrams:
            pattern_phrases.append(phrase[0] + " " + phrase[1] + " " + phrase[2])

        words.extend(pattern_phrases)

        # adding documents
        documents.append((pattern_phrases, intent['tag']))

        # adding classes to our class list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)


pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
    # initializing bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(float(1)) if w in pattern_words else bag.append(float(0))

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = float(1)

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 128 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['categorical_accuracy'])

# fitting and saving the model
print(len(train_x[0]))
print(np.array(train_x).shape)
print(len(train_y[0]))
print(np.array(train_y).shape)
hist = model.fit(np.array(train_x), np.array(train_y), steps_per_epoch=10, epochs=175, batch_size=5, verbose=1)
model.save('./sentimentmodel.h5', hist)


