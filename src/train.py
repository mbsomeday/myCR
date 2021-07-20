from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import cv2
import os
import json

from config import cfg
from model import CRNN
from data_generator import TrainGenerator, Val_Generator
from utils import MultiGPUModelCheckpoint, PredictionModelCheckpoint, Evaluator

num_classes = 10
input_shape = (28, 28, 1)


def get_models():
    return CRNN(cfg)


def get_callbacks(training_model, prediction_model):
    prediction_model_checkpoint = PredictionModelCheckpoint(cfg.prediction_model_cp_filename, prediction_model,
                                                            monitor='loss',
                                                            save_best_only=cfg.save_best_only, mode='min')
    le_reducer = keras.callbacks.ReduceLROnPlateau(factor=cfg.lr_reduction_factor, patience=3, verbose=1,
                                                   min_lr=0.00001)
    return [prediction_model_checkpoint, le_reducer]


def get_generator():
    train_generator = TrainGenerator(dataset=cfg.o_trainSet,
                                     json_file_path=cfg.o_trainJson,
                                     batch_size=cfg.batch_size,
                                     img_height=cfg.height,
                                     img_width=cfg.width,
                                     channels=cfg.nb_channels,
                                     timesteps=cfg.timesteps,
                                     label_len=cfg.label_len,
                                     characters=cfg.characters)
    val_generator = Val_Generator(dataset=cfg.o_valSet,
                                  json_file_path=cfg.o_valJson,
                                  batch_size=cfg.batch_size,
                                  img_height=cfg.height,
                                  img_width=cfg.width,
                                  channels=cfg.nb_channels,
                                  timesteps=cfg.timesteps,
                                  label_len=cfg.label_len,
                                  characters=cfg.characters)
    return train_generator, val_generator


def get_optimizer():
    if cfg.optimizer == 'sgd':
        opt = keras.optimizers.SGD(lr=cfg.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif cfg.optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=cfg.lr)
    return opt


if __name__ == '__main__':

    training_model, prediction_model = get_models()
    train_generator, val_generator = get_generator()
    # prediction_model, val_generator, label_len, characters, optimizer, period = 2000):
    # e = Evaluator(prediction_model=prediction_model,
    #               val_generator=val_generator,
    #               label_len=cfg.label_len,
    #               characters=cfg.characters,
    #               optimizer=cfg.optimizer)
    # res = e.evaluate()
    
    opt = get_optimizer()
    callbacks = get_callbacks(training_model, prediction_model)
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    training_model.fit_generator(train_generator, steps_per_epoch=10, epochs=5, verbose=1, callbacks=callbacks)
    # training_model.summary()
