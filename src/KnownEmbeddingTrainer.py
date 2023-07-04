from src.models.SoftMax import SoftMax
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import numpy as np
import pickle
import time
from tensorflow.keras.callbacks import TensorBoard

class KnownEmbeddingTrainer:

    def __init__(self, labeled_embeddings, layers=2, units_per_layer=4096, dropout=0.2, batch_size=32, epochs=5):
        self.labeled_embeddings = labeled_embeddings
        self.layers = layers
        self.units_per_layer = units_per_layer
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.label_encoder = None

    def __encoded_labels(self):
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(self.labeled_embeddings["labels"])
        num_classes = len(np.unique(labels))
        labels = labels.reshape(-1, 1)
        one_hot_encoder = OneHotEncoder()
        return (num_classes, one_hot_encoder.fit_transform(labels).toarray())

    def train(self, model_name=None, encoded_labels_path="outputs/encoded_labels.pickle"):
        # summaries = open("outputs/conv_models_summaries.txt", 'a+')

        NAME = f"{self.layers}-dense-{self.units_per_layer}-nodes-{self.epochs}-epochs__{int(time.time())}"
        print(model_name or NAME)
        # summaries.write(model_name or NAME)
        # summaries.write("\r\n")

        num_classes, labels = self.__encoded_labels()
        embeddings = np.array(self.labeled_embeddings["embeddings"])
        input_shape = embeddings.shape[1]

        print(f"----- INPUT SHAPE {input_shape} -----")

        model = SoftMax(input_shape=(input_shape,), num_classes=num_classes, layers=self.layers, units_per_layer=self.units_per_layer, dropout=self.dropout)

        training_model = model.build()

        print(training_model.summary())
        # training_model.summary(print_fn=lambda x: summaries.write(x + '\n'))
        # summaries.write("\r\n\r\n")

        # summaries.close()

        # fix
        cv = KFold(n_splits = 2, random_state = 42, shuffle=True)

        tensorboard = TensorBoard(log_dir="logs/{}".format(model_name or NAME))

        # Train
        for train_idx, valid_idx in cv.split(embeddings):
            X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]

            training_model.fit(X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1,
                validation_data=(X_val, y_val),
                callbacks=[tensorboard])

        # write the face recognition model to output
        training_model.save(f"outputs/{model_name or NAME}.h5")
        f = open(encoded_labels_path, "wb")
        f.write(pickle.dumps(self.label_encoder))
        f.close()
