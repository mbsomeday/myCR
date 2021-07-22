import json
import os
import cv2
from config import cfg
import numpy as np
from tensorflow import keras
import abc


class Generator(keras.utils.Sequence):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, dataset, json_file_path, batch_size, img_height, img_width, channels, timesteps, label_len,
                 characters, shuffle=True):
        self.dataset = dataset  # 图片所在文件夹
        self.json_file_path = json_file_path
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.timesteps = timesteps
        self.label_len = label_len
        self.characters = characters
        self.json_names = self._get_json_names()
        self.shuffle = shuffle
        self.on_epoch_end()
        self.nb_samples = len(self.json_names)
    
    def __len__(self):
        return int(np.floor(self._num_samples() / self.batch_size))
    
    def _num_samples(self):
        return len(list(json.load(open(self.json_file_path)).keys()))
    
    def _get_json_names(self):
        return list(json.load(open(self.json_file_path)))
    
    def __getitem__(self, index):
        '''
        :param index:
        :return: 返回一个batch的 x,y
        '''
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        batch_json_names = [self.json_names[i] for i in indices]
        
        X, y = self._generate_data(batch_json_names)
        
        return X, y
    
    @abc.abstractmethod
    def _generate_data(self, batch_json_names):
        pass
    
    def on_epoch_end(self):
        self.indices = np.arange(self._num_samples())
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    
    
    def data_wash(self):
        all_json_info = json.load(open(self.json_file_path))
        img_names = list(all_json_info.keys())
        
        for idx, name in enumerate(img_names):
            cur_json_info = all_json_info[name]
            img_path = os.path.join(self.dataset, name)
            # img.shape: [h, w]
            img = cv2.imread(img_path)
            label_num = len(cur_json_info['label'])
            # print("json info：", cur_json_info)
            for i in range(label_num):
                h = cur_json_info["height"][i]
                w = cur_json_info['width'][i]
                left = cur_json_info['left'][i]
                top = cur_json_info['top'][i]
                # img [y1:y2, x1:x2]
                curImg = img[int(top):int(top + h), int(left):int(left + w)]
                flag = self.check_resize(img_shape=img.shape, y1=int(top), y2=int(top+h), x1=int(left), x2=int(left+w))
                if flag == -1:
                    print("出错的图片信息:", name)
                    continue
                    
                if curImg.shape[0] != self.img_height:
                    curImg = cv2.resize(curImg, dsize=[int(curImg.shape[1] * self.img_height / curImg.shape[0]), self.img_height], interpolation=cv2.INTER_CUBIC)
                
                
    def check_resize(self, img_shape, y1, y2, x1, x2):
        height = img_shape[0]
        width = img_shape[1]
        if y1 < 0 :
            print("标签值top小于0！", y1)
            return -1
        if x1 < 0:
            print("标签值left小于0！", x1)
            return -1
        if y1 > y2:
            print("竖向裁剪尺寸不对！", y1, y2)
            return -1
        if x1 > x2:
            print("横向裁剪尺寸不对！", x1, x2)
            return -1
        if y2 > height:
            print("裁剪高度溢出！", y2, height)
            return -1
        if x2 > width:
            print("裁剪宽度溢出！", y2, height)
            return -1
        return 1
        
        
    
    
    def load_image_and_annotation(self, json_name):
        res_num = []
        res_img = []
        
        all_json_info = json.load(open(self.json_file_path))
        cur_json_info = all_json_info[json_name]
        
        # 将 int 型 label 转换为 str 型
        int_labels = cur_json_info['label']
        for i in int_labels:
            res_num.append(str(i))
        
        label_num = len(res_num)
        
        # 获取图像
        img_path = os.path.join(self.dataset, json_name)
        img = cv2.imread(img_path)
        # print("img_path:", img_path)
        if img is not None:
            for i in range(label_num):
                h = cur_json_info["height"][i]
                w = cur_json_info['width'][i]
                left = cur_json_info['left'][i]
                top = cur_json_info['top'][i]
                # img [y1:y2, x1:x2]
                curImg = img[int(top):int(top + h), int(left):int((left + w))]
                flag = self.check_resize(img_shape=img.shape, y1=int(top), y2=int(top+h), x1=int(left), x2=int(left+w))
                if flag == -1:
                    continue
                # img.shape: [height, width]
                if curImg.shape[0] != self.img_height:
                    # resize: [width, height]
                    curImg = cv2.resize(curImg,
                                        dsize=[int(curImg.shape[1] * self.img_height / curImg.shape[0]), self.img_height],
                                        interpolation=cv2.INTER_CUBIC)
                
                res_img.append(curImg)
            
            if len(res_img) == 1:
                res_img = self.process_image(res_img[0])
            elif len(res_img) > 1:
                tmp = res_img[0]
                for i in range(1, len(res_img)):
                    tmp = np.concatenate((tmp, res_img[i]), axis=1)
                res_img = self.process_image(tmp)
        else:
            print("图片出错，路径为：", img_path)
        
        
        return res_img, res_num
    
    def process_image(self, image):
        w_h_ratio = image.shape[1] / image.shape[0]
        # 200/32 = 6.25
        if w_h_ratio <= 6.25:
            image = self.img_padding(image)
        else:
            image = cv2.resize(image, dsize=(self.img_width, self.img_height))
        return image
    
    def img_padding(self, image):
        pad = np.zeros(shape=(self.img_height, self.img_width - image.shape[1], self.channels), dtype=np.uint8)
        image = np.concatenate((image, pad), axis=1)
        return image
    
    def preprocess(self, img):
        img = img.transpose([1, 0, 2])
        img = np.flip(img, 1)
        img = img / 255.0
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        return img
    

class TrainGenerator(Generator):
    def _generate_data(self, batch_json_names):
        # np.zeros(shape=(h, w))
        x = np.zeros((self.batch_size, self.img_width, self.img_height, self.channels), dtype=np.uint8)
        y = np.zeros((self.batch_size, self.label_len), dtype=np.uint8)
        
        # print("x.shape", x.shape)
        
        for idx, fn in enumerate(batch_json_names):
            img, label = self.load_image_and_annotation(fn)
            img = self.preprocess(img)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            x[idx] = img
            
            while len(label) < self.label_len:
                label += '-'
            
            y[idx] = [self.characters.find(c) for c in label]
        
        return [x, y, np.ones(self.batch_size) * int(self.timesteps - 2), np.ones(self.batch_size) * self.label_len], y


class Val_Generator(Generator):
    def _generate_data(self, batch_json_names):
        x = np.zeros((self.batch_size, self.img_width, self.img_height, self.channels))
        y = []
        
        for i, fn in enumerate(batch_json_names):
            img, word = self.load_image_and_annotation(fn)
            img = self.preprocess(img)
            x[i] = img
            y.append(word)
        return x, y



    
    
    

if __name__ == '__main__':
    pass
