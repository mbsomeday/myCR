from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from config import cfg


def ctc_lambda_func(args):
	iy_pred, ilabels, iinput_length, ilabel_length = args
	iy_pred = iy_pred[:, 2:, :]
	return keras.backend.ctc_batch_cost(ilabels, iy_pred, iinput_length,ilabel_length)


def CRNN(cfg):
	inputs = keras.Input(shape=(cfg.width, cfg.height, cfg.nb_channels))
	c_1 = layers.Conv2D(filters=cfg.conv_filter_size[0], kernel_size=(3, 3), activation='relu', padding='same', name='conv_1')(inputs)
	c_2 = layers.Conv2D(filters=cfg.conv_filter_size[1], kernel_size=(3, 3), activation='relu', padding='same', name='conv_2')(c_1)
	c_3 = layers.Conv2D(cfg.conv_filter_size[2], (3, 3), activation='relu', padding='same', name='conv_3')(c_2)
	bn_3 = layers.BatchNormalization(name='bn_3')(c_3)
	p_3 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(bn_3)

	c_4 = layers.Conv2D(cfg.conv_filter_size[3], (3, 3), activation='relu', padding='same', name='conv_4')(p_3)
	c_5 = layers.Conv2D(cfg.conv_filter_size[4], (3, 3), activation='relu', padding='same', name='conv_5')(c_4)
	bn_5 = layers.BatchNormalization(name='bn_5')(c_5)
	p_5 = layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(bn_5)

	c_6 = layers.Conv2D(cfg.conv_filter_size[5], (3, 3), activation='relu', padding='same', name='conv_6')(p_5)
	c_7 = layers.Conv2D(cfg.conv_filter_size[6], (3, 3), activation='relu', padding='same', name='conv_7')(c_6)
	bn_7 = layers.BatchNormalization(name='bn_7')(c_7)

	bn_7_shape = bn_7.get_shape()
	reshape = layers.Reshape(target_shape=(int(bn_7_shape[1]), int(bn_7_shape[2] * bn_7_shape[3])), name='reshape')(bn_7)

	fc_9 = layers.Dense(cfg.lstm_nb_units[0], activation='relu', name='fc_9')(reshape)

	# lstm_10的参数量计算：
	#    one_gete_param = (fc_9.shape + lstm_10.shape) * lstm_10.shape + lstm_10.shape
	#    total_param = one_gete_param * 4
	lstm_10 = layers.LSTM(cfg.lstm_nb_units[1], kernel_initializer='he_normal', return_sequences=True, name='lstm_10')(fc_9)
	lstm_10_back = layers.LSTM(cfg.lstm_nb_units[1], kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_10_back')(fc_9)
	lstm_10_add = layers.add([lstm_10, lstm_10_back])

	lstm_11 = layers.LSTM(cfg.lstm_nb_units[1], kernel_initializer="he_normal", return_sequences=True, name='lstm_11')(lstm_10_add)
	lstm_11_back = layers.LSTM(cfg.lstm_nb_units[1], kernel_initializer="he_normal", go_backwards=True, return_sequences=True, name='lstm_11_back')(lstm_10_add)
	lstm_11_concat = layers.concatenate([lstm_11, lstm_11_back])
	do_11 = layers.Dropout(cfg.dropout_rate, name='dropout')(lstm_11_concat)

	fc_12 = layers.Dense(len(cfg.characters), kernel_initializer="he_normal", activation="softmax", name="fc_12")(do_11)

	prediction_model = keras.Model(inputs=inputs, outputs=fc_12)

	labels = keras.Input(name='labels', shape=[cfg.label_len], dtype="float32")
	input_length = keras.Input(name='input_length', shape=[1], dtype='int64')
	label_length = keras.Input(name='label_length', shape=[1], dtype='int64')

	ctc_loss = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([fc_12, labels, input_length, label_length])

	training_model = keras.Model(inputs=[inputs, labels, input_length, label_length], outputs=[ctc_loss])

	# training_model.summary()

	return training_model, prediction_model


if __name__ == '__main__':
    i1 = np.zeros(shape=(3, cfg.width, cfg.height, cfg.nb_channels))
    i2 = np.zeros(shape=[cfg.label_len])
    i3 = np.zeros(shape=[1])
    i4 = np.zeros(shape=[1])
    fake_input = [i1, i2, i3, i4]
    print(fake_input[0].shape)
    training_model, prediction_model = CRNN(cfg)
    opt = keras.optimizers.SGD(lr=cfg.lr, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    training_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=opt)






























