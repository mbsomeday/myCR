import os, glob
import numpy as np
from tensorflow import keras

from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow.keras.backend as K


class MultiGPUModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, alternate_mode, **kwargs):
        self.alternate_model = alternate_mode
        super().__init__(filepath, **kwargs)

    def on_epoch_end(self, epoch, logs=None):
        model_before = self.model
        self.model = self.alternate_model
        super().on_epoch_end(epoch, logs)
        self.model = model_before


class MyCallback(Callback):
    def on_train_batch_begin(self, batch, logs=None):
        print("********** this is train batch begin.************")

    def on_train_batch_end(self, batch, logs=None):
        print("******* this is train batch end. ******")
        self.model.save("my_model")


class PredictionModelCheckpoint(Callback):
    def __init__(self, filepath, prediction_model, monitor='loss', save_best_only=False, mode='min', period=1,
                 save_weights_only=False, verbose=False):
        self.filepath = filepath
        self.prediction_model = prediction_model
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.period = period
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        self.epochs_since_last_save = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch+1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if self.monitor_op(current, self.best):
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch+1, self.monitor, self.best, current, filepath))
                    if self.save_weights_only:
                        self.prediction_model.save_weights(filepath, overwrite=True)
                    else:
                        self.prediction_model.save(filepath, overwrite=True)


class Evaluator(Callback):
    def __init__(self, prediction_model, val_generator, label_len, characters, optimizer, period=2000):
        self.prediction_model = prediction_model
        self.period = period
        self.val_generator = val_generator
        self.label_len = label_len
        self.characters = characters
        self.optimizer = optimizer

    def evaluate(self):
        correct_predictions = 0
        correct_char_predictions = 0

        # x_val, y_val









if __name__ == '__main__':
    a = 1
    b = 5
    c = np.greater(a, b)
    print(c)
