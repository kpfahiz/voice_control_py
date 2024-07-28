import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras import utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

class Intent:
    def __init__(self):
        self.snips_df = pd.read_json("data/snips.json")
        self.snips_df.columns = ["text", "intent"]
        self.encoder = LabelEncoder()
        self.MAX_SEQ_LEN = 0
        self.X = 0
    
    def Train(self):
        '''
        Prepare data for training.
        '''
        #Split the dataset into train and test
        self.TEST_SPLIT = 0.2
        self.RANDOM_STATE = 10
        np.random.seed(self.RANDOM_STATE)
        tf.random.set_seed(self.RANDOM_STATE)
        X_train, X_test, y_train, y_test = train_test_split(self.snips_df["text"], self.snips_df["intent"], test_size = self.TEST_SPLIT, random_state = self.RANDOM_STATE)
        return X_train, X_test, y_train, y_test
    
    
    
    def Extract(self):
        '''
        Feature Extraction.
        '''
        X_train, X_test, y_train, y_test = self.Train()
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(X_train))

        #Convert text to sequences
        X_seq = tokenizer.texts_to_sequences(list(X_train))
        X_test_seq = tokenizer.texts_to_sequences(list(X_test))

        X_seq_len = [len(x) for x in X_seq]
        X_max_seq_len = max(X_seq_len)

        X_test_seq_len = [len(x) for x in X_test_seq]
        X_max_test_seq_len = max(X_test_seq_len)

        self.MAX_SEQ_LEN = max(X_max_seq_len, X_max_test_seq_len)

        #pad the sequences
        self.X = pad_sequences(X_seq, maxlen = self.MAX_SEQ_LEN, padding = 'post')
        X_test = pad_sequences(X_test_seq, maxlen = self.MAX_SEQ_LEN, padding = 'post')

        #Convert labels to one-hot vectors
        y = y_train.to_numpy()
        self.encoder.fit(y)

        encoded_y = self.encoder.transform(y)
        y_train_encoded = utils.to_categorical(encoded_y)

        y_test = y_test.to_numpy()
        encoded_y_test = self.encoder.transform(y_test)
        y_test_encoded = utils.to_categorical(encoded_y_test)
        return y_train_encoded, y_test_encoded, X_test, tokenizer
    
    def Model_Train(self, sentence):
        '''
        Model Training
        '''
        VAL_SPLIT = 0.1
        BATCH_SIZE = 32
        EPOCHS = 7
        EMBEDDING_DIM = 16
        NUM_UNITS = 16
        NUM_CLASSES = len(self.snips_df['intent'].unique())
        sentence = sentence
        y_train_encoded, y_test_encoded, X_test, tokenizer = self.Extract()
        VOCAB_SIZE = len(tokenizer.word_index) + 1

        #LSTM
        lstm_model = Sequential()
        lstm_model.add(Embedding(input_dim = VOCAB_SIZE, output_dim = EMBEDDING_DIM, input_length = self.MAX_SEQ_LEN, mask_zero = True))
        lstm_model.add(LSTM(NUM_UNITS, activation='relu'))
        lstm_model.add(Dense(NUM_CLASSES, activation='softmax'))

        lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[Precision(), Recall(), 'accuracy'])

        lstm_model.fit(self.X, y_train_encoded, batch_size = BATCH_SIZE, epochs = EPOCHS, verbose = 1, validation_split = VAL_SPLIT)

        #Evaluate the model performance on test data
        lstm_model.evaluate(X_test, y_test_encoded, batch_size = BATCH_SIZE, verbose = 1)
        input_seq = tokenizer.texts_to_sequences([sentence])
        print('LSM Completed')
        return lstm_model, input_seq
    
    def Predict(self,sentence):
        '''
        prediction
        '''
        sentence = sentence
        lstm_model, input_seq = self.Model_Train(sentence=sentence)
        input_features = pad_sequences(input_seq, maxlen = self.MAX_SEQ_LEN, padding = 'post')
        probs = lstm_model.predict(input_features)
        predicted_y = probs.argmax(axis=-1)
        print(self.encoder.classes_[predicted_y][0])
        return self.encoder.classes_[predicted_y][0]

if __name__ == '__main__':
    inte = Intent()
    inte.Predict(sentence="Play music on YouTube Music")