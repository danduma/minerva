from __future__ import print_function
import os
import numpy as np

np.warnings.filterwarnings('ignore')

CUSTOM_SEED = 42
np.random.seed(CUSTOM_SEED)

from keras.models import Sequential
from keras.layers import (Dense, Dropout, Activation)
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras import backend as K
from numpy import argmax

from models.base_model import BaseModel
from models.keyword_features import FeaturesReader, getTrainTestData, filterOutFeatures, getRootDir
from models.models_util import plot_model_performance


def custom_loss(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    print(y_true.shape)
    print(y_pred.shape)
    pos = K.sum(y_true * y_pred, axis=-1) * 0.9
    neg = K.sum((1. - y_true) * y_pred, axis=-1) * 0.1
    # return K.maximum(0., neg - pos + 1.)
    score = neg - pos + 1
    return score


def matrixFromContextFeatures(contexts, dict_vectorizer, MAX_CONTEXT_LEN, dtype=np.float32):
    all_features = [dict_vectorizer.transform(context) for context in contexts]
    all_features = pad_sequences(all_features, maxlen=MAX_CONTEXT_LEN, padding="post", dtype=dtype)
    #     matrix=np.zeros((len(all_features),all_features[0].shape[0],all_features[0].shape[1]))
    #     for index, vect in enumerate(all_features):
    #         matrix[index,:,:]=vect
    # print(dict_vectorizer.get_feature_names())
    matrix = np.stack(all_features, axis=0)
    print(matrix.shape)
    return matrix

class KerasModel(BaseModel):
    def __init__(self, exp_dir, params={},
                 train_data_filename="feature_data.json.gz",
                 test_data_filename="feature_data_test.json.gz"):
        super().__init__(exp_dir, params, train_data_filename, test_data_filename)

    def postProcessLoadedData(self):
        self.MAX_CONTEXT_LEN = max([len(x["tokens"]) for x in self.contexts])

        train_val_cutoff = int(.80 * len(self.contexts))

        self.training_contexts = self.contexts[:train_val_cutoff]
        self.validation_contexts = self.contexts[train_val_cutoff:]

        self.X_train, self.y_train = getTrainTestData(self.training_contexts)
        self.X_val, self.y_val = getTrainTestData(self.validation_contexts)

        self.X_train = matrixFromContextFeatures(self.X_train, self.dict_vectorizer, self.MAX_CONTEXT_LEN)
        self.X_val = matrixFromContextFeatures(self.X_val, self.dict_vectorizer, self.MAX_CONTEXT_LEN)

        self.y_train = pad_sequences(self.y_train, maxlen=self.MAX_CONTEXT_LEN, padding="post", dtype=self.dtype)
        self.y_val = pad_sequences(self.y_val, maxlen=self.MAX_CONTEXT_LEN, padding="post", dtype=self.dtype)

        self.y_train = np_utils.to_categorical(self.y_train)
        self.y_val = np_utils.to_categorical(self.y_val)

    def defineModel(self):
        self.model_params = {
            'input_dim': self.X_train.shape[1],
            'hidden_neurons': 256,
            # 'output_dim': self.y_train.shape[2],
            'epochs': 1,
            'batch_size': 64,
            'verbose': 1,
            'shuffle': True,
            # 'class_weight': class_weights
        }

        # self.model = Sequential([
        #     LSTM(self.model_params["hidden_neurons"],
        #          # batch_input_shape=(model_params["batch_size"], X_train.shape[1], X_train.shape[2]),
        #          input_shape=self.X_train.shape[1:],
        #          return_sequences=True),
        #     BatchNormalization(axis=1),
        #     Activation("relu"),
        #     # Dropout(0.2),
        #     # LSTM(self.model_params["hidden_neurons"], return_sequences=True),
        #     # Activation("relu"),
        #     #     LSTM(model_params["hidden_neurons"], return_sequences=True),
        #     #     Activation("relu"),
        #     TimeDistributed(Dense(self.y_train.shape[-1],
        #                           activation="softmax"),
        #                     input_shape=self.X_train.shape[1:])
        # ])

        self.model = Sequential([
            Dense(self.model_params["hidden_neurons"], input_shape=self.X_train.shape[1:]),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.model_params["hidden_neurons"]),
            Activation('relu'),
            Dropout(0.2),
            Dense(self.y_train.shape[-1], activation='softmax')
        ])

        self.model.compile(
            loss='binary_crossentropy',
            # loss=custom_loss,
            optimizer='rmsprop',
            metrics=["accuracy"]
        )

    def trainModel(self):
        self.hist = self.model.fit(self.X_train,
                                   self.y_train,
                                   batch_size=self.model_params["batch_size"],
                                   epochs=self.model_params["epochs"],
                                   validation_data=(self.X_val, self.y_val)
                                   )

    def testModel(self):
        self.reader = FeaturesReader(self.test_data_filename)
        self.testing_contexts = [c for c in self.reader]
        self.testing_contexts = [filterOutFeatures(context, self.features_to_filter_out, self.corpus) for context in self.testing_contexts]

        self.X_test, self.y_test = getTrainTestData(self.testing_contexts)
        self.X_test = matrixFromContextFeatures(self.X_test, self.dict_vectorizer, self.MAX_CONTEXT_LEN)

        self.y_test = pad_sequences(self.y_test, maxlen=self.MAX_CONTEXT_LEN, padding="post", dtype=self.dtype)
        self.y_test = np_utils.to_categorical(self.y_test)

        predicted_matrix = self.model.predict(self.X_test)

        predicted = argmax(predicted_matrix, axis=2).reshape(-1)
        labels = argmax(self.y_test, axis=2).reshape(-1)

        self.printClassificationResults(self.y_test, predicted)

    def plotPerformance(self):
        """ Plot model loss and accuracy through epochs. """

        train_loss = self.hist.history.get('loss', []),
        train_acc = self.hist.history.get('acc', []),
        train_val_loss = self.hist.history.get('val_loss', []),
        train_val_acc = self.hist.history.get('val_acc', [])
        plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc)


def main():
    params = {}
    # exp_dir = os.path.join(getRootDir("aac"), "experiments", "aac_generate_kw_trace")
    # model = KerasModel(exp_dir, params=params,
    #                    train_data_filename="feature_data.json.gz",
    #                    test_data_filename="feature_data_test.json.gz"
    #                    )
    exp_dir = os.path.join(getRootDir("pmc_coresc"), "experiments", "pmc_generate_kw_trace")
    model = KerasModel(exp_dir, params=params,
                       train_data_filename="feature_data_at_w_min1.json.gz",
                       test_data_filename="feature_data_test_at_w.json.gz"
                       )
    model.run()


if __name__ == '__main__':
    main()
