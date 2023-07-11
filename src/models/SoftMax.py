from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import keras

class SoftMax():
    def __init__(self, input_shape, num_classes, layers, units_per_layer, dropout):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = layers
        self.units_per_layer = units_per_layer
        self.dropout = dropout

    def build(self):
        print(f"{self.input_shape} {self.num_classes}")

        model = Sequential()
        model.add(Input(shape=self.input_shape))

        if type(self.units_per_layer) is list:
            for layer_index in range(len(self.units_per_layer)):
                model.add(Dense(self.units_per_layer[layer_index], activation='relu'))
                model.add(Dropout(self.dropout))
        else:
            for _ in range(self.layers):
                model.add(Dense(self.units_per_layer, activation='relu'))
                model.add(Dropout(self.dropout))


        model.add(Dense(self.num_classes, activation='softmax'))

        optimizer = Adam()
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model
