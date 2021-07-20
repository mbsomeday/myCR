import cv2
import os
import numpy as np
import tensorflow.keras.backend as K

from config import cfg
from model import CRNN


def get_batch_img():
    res_images = np.zeros(shape=(cfg.batch_size, cfg.width, cfg.height, cfg.nb_channels), dtype="float")
    # print("batch的尺寸：", res_images.shape)
    h_path = r'../../tfKeras/tianchi/mchar_test_a'
    o_path = r'../../mchar_test_a'
    # 获取文件路径
    for fpath, dirname, fnames in os.walk(o_path):
        # print("fpath:", fpath)
        # print("fnames", fnames)
        for i, f in enumerate(fnames):
            if i == cfg.batch_size:
                break
            file_path = os.path.join(fpath, f)
            img = cv2.imread(file_path)
            img = preprocess_img(img)
            res_images[i] = img
            
    return res_images


def preprocess_img(image):
    if image.shape[0] != cfg.height:
        image = cv2.resize(image, dsize=[int(image.shape[1] * cfg.height / image.shape[0]), cfg.height])
    
    w_h_ratio = image.shape[1] / image.shape[0]
    
    if w_h_ratio <= 6.25:
        pad = np.zeros(shape=(cfg.height, cfg.width - image.shape[1], cfg.nb_channels), dtype=np.uint8)
        image = np.concatenate((image, pad), axis=1)
    else:
        image = cv2.resize(image, dsize=(cfg.width, cfg.height))
    
    # 转换为网络需要的尺寸 [transpose]
    img = image.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    # print("shape", img.shape)
    return img


def predict_num(model, img):
    y_pred = model.predict(img[np.newaxis, :, :, :])
    print("y_pred.shape:", y_pred.shape)
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
    print("ctc_out:", ctc_out)
    result_str = ''.join([cfg.characters[c] for c in ctc_out[0]])
    print("result_str:", result_str)
    result_str = result_str.replace('-', '')
    return result_str


if __name__ == '__main__':
    images = get_batch_img()
    _, prediction_model = CRNN(cfg)
    prediction_model.load_weights(r'./model/prediction_model.005.h5')
    test_img = get_batch_img()
    res = predict_num(prediction_model, test_img[0])
    print("res", res)
    
    