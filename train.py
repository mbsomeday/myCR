from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from config import cfg
import cv2
import os

from model import CRNN
from data_generator import ValGenerator
from utils import MultiGPUModelCheckpoint, PredictionModelCheckpoint

num_classes = 10
input_shape = (28, 28, 1)


def get_models():
	return CRNN(cfg)

def get_callbacks(training_model, prediction_model):
	prediction_model_checkpoint = PredictionModelCheckpoint(cfg.prediction_model_cp_filename, prediction_model, monitor='loss',
															save_best_only=cfg.save_best_only, mode='min')
	le_reducer = keras.callbacks.ReduceLROnPlateau(factor=cfg.lr_reduction_factor, patience=3, verbose=1, min_lr=0.00001)
	return [prediction_model_checkpoint]




if __name__ == '__main__':
	val_generator = ValGenerator(dataset=r'../mchar_train',
	                             json_file_path=r'../mchar_train.json',
	                             batch_size=cfg.batch_size,
	                             img_height=cfg.height,
	                             img_width=cfg.width,
	                             channels=cfg.nb_channels,
	                             timesteps=cfg.timesteps,
	                             label_len=cfg.label_len,
	                             characters=cfg.characters,
								 shuffle=False)
	training_model, prediction_model = get_models()
	callbacks = get_callbacks(training_model, prediction_model)
	training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(lr=cfg.lr))
	# training_model.fit_generator(val_generator, steps_per_epoch=10, epochs=5, verbose=1, callbacks=callbacks)
	training_model.summary()


