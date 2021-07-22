import os, glob
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow.keras.backend as K
import cv2

from model import CRNN
from config import cfg
from data_generator import TrainGenerator, Val_Generator


# 保存训练模型
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


# 保存测试模型
class PredictionModelCheckpoint(Callback):
    def __init__(self, filepath, prediction_model, monitor='loss', save_best_only=True, mode='min', period=1,
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
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            print("测试模型保存的filepath：", filepath)
            if self.save_best_only:   # 只保存目前为止最好的模型
                current = logs.get(self.monitor)
                if self.monitor_op(current, self.best):
                    print('\nEpoch %05d: %s improved from %0.5f to %0.5f, saving model to %s'
                          % (epoch + 1, self.monitor, self.best, current, filepath))
                    if self.save_weights_only:
                        self.prediction_model.save_weights(filepath, overwrite=True)
                    else:
                        self.prediction_model.save(filepath, overwrite=True)
            else:   # 每次epoch都保存
                if self.save_weights_only:
                    self.prediction_model.save_weights(filepath, overwrite=True)
                else:
                    self.prediction_model.save(filepath, overwrite=True)


# 每period个batch、每个epoch都测一次模型性能
class Evaluator(Callback):
    def __init__(self, prediction_model, val_generator, label_len, characters, optimizer, period=10):
        self.prediction_model = prediction_model
        self.period = period
        self.val_generator = val_generator
        self.label_len = label_len
        self.characters = characters
        self.optimizer = optimizer

    def on_batch_end(self, batch, logs=None):
        if ((batch + 1) % self.period == 0):
            accuracy, correct_char_predictions = self.evaluate()
            print("=========================")
            print("accuracy:", accuracy)
            print("correct_char_predictions:", correct_char_predictions)
            print("=========================")

    def on_epoch_end(self, epoch, logs=None):
        accuracy, correct_char_predictions = self.evaluate()
        print('=====================================')
        print('After epoch %d' % epoch)
        print('Word level accuracy: %.3f' % accuracy)
        print('Correct character level predictions: %d' % correct_char_predictions)

        # 以下为输出学习率相关信息
        if self.optimizer == 'sgd':
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
            print("Decayed learning rate: %.8f" % K.eval(lr_with_decay))
        else:
            print("Learning rate: %.8f" % K.eval(self.model.optimizer.lr))


    def evaluate(self):
        correct_predictions = 0
        correct_char_predictions = 0

        x_val, y_val = self.val_generator[np.random.randint(0, int(self.val_generator.nb_samples / self.val_generator.batch_size))]
        
        # cv2.imshow("img", x_val[0])
        # cv2.waitKey(0)

        # print("real label:", y_val)

        y_pred = self.prediction_model.predict(x_val)
        # print("predict:", y_pred.shape)

        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        ctc_out = K.get_value(ctc_decode)[:, :self.label_len]

        # print("val_generator.batch_size：", self.val_generator.batch_size)
        for i in range(self.val_generator.batch_size):
            # print("***", ctc_out[i], "***")
            result_num = ''.join([self.characters[c] for c in ctc_out[i]])
            result_num = result_num.replace("-", '')
            if result_num == y_val[i]:
                correct_predictions += 1

            for c1, c2 in zip(result_num, y_val[i]):
                if c1 == c2:
                    correct_char_predictions += 1

        return correct_predictions / self.val_generator.batch_size, correct_char_predictions


if __name__ == '__main__':
    training_model, prediction_model = CRNN(cfg)
