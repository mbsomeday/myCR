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


def get_callbacks(training_model, prediction_model, val_generator):
    training_model_checkpoint = MultiGPUModelCheckpoint(os.path.join(cfg.training_model_dir, cfg.training_model_cp),
                                                        training_model,
                                                        save_best_only=True,
                                                        monitor='loss',
                                                        mode='min')
    prediction_model_checkpoint = PredictionModelCheckpoint(os.path.join(cfg.prediction_model_dir, cfg.prediction_model_cp),
                                                            prediction_model,
                                                            monitor='loss',
                                                            save_best_only=cfg.save_best_only, mode='min')
    le_reducer = keras.callbacks.ReduceLROnPlateau(factor=cfg.lr_reduction_factor, patience=3, verbose=1,
                                                   min_lr=0.0000001)
    evaluator = Evaluator(prediction_model, val_generator, cfg.label_len, cfg.characters, cfg.optimizer)
    return [training_model_checkpoint, prediction_model_checkpoint, le_reducer, evaluator]


def get_generator():
    train_generator = TrainGenerator(dataset=cfg.o_trainSet,
                                     json_file_path=cfg.o_trainJson,
                                     batch_size=cfg.batch_size,
                                     img_height=cfg.height,
                                     img_width=cfg.width,
                                     channels=cfg.nb_channels,
                                     timesteps=cfg.timesteps,
                                     label_len=cfg.label_len,
                                     characters=cfg.characters,
                                     shuffle=False
                                     )
    val_generator = Val_Generator(dataset=cfg.o_valSet,
                                  json_file_path=cfg.o_valJson,
                                  batch_size=cfg.batch_size,
                                  img_height=cfg.height,
                                  img_width=cfg.width,
                                  channels=cfg.nb_channels,
                                  timesteps=cfg.timesteps,
                                  label_len=cfg.label_len,
                                  characters=cfg.characters,
                                  shuffle=False
                                  )
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
    # train_generator.data_wash()
    # l = int(train_generator.nb_samples/train_generator.batch_size)
    # x, y = train_generator[7]
    # train_generator.load_image_and_annotation()
    # for i in range(l):  # 251
    #     print(i)
    #     x, y = train_generator[i]


    opt = get_optimizer()
    callbacks = get_callbacks(training_model, prediction_model, val_generator)
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)
    training_model.fit_generator(train_generator,
                                 steps_per_epoch=int(train_generator.nb_samples/train_generator.batch_size),
                                 epochs=1, verbose=1, callbacks=callbacks)
    
    # img = cv2.imread("./a")
    # print(type(img))
    # if img is None:
    #     print("是None")
    # else:
    #     print("else")
    
