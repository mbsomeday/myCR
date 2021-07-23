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
        
        self.josn_list = self._get_all_json_list()
        self.nb_samples = len(self.josn_list)
        
        self.shuffle = shuffle
        self.on_epoch_end()
    
    
    def __len__(self):
        return int(np.floor(self.nb_samples/ self.batch_size))
    
    
    def _get_all_json_list(self):
        '''
        :return: 以列表形式返回各条josn数据，[{ 'xxx.png': { 'height': [], ...} }]
        '''
        self.json_list = []
        with open(self.json_file_path, 'r') as f:
            for cur_json in f.readlines():
                cur_json = json.loads(cur_json)
                self.json_list.append(cur_json)
        return self.json_list
    
    
    def on_epoch_end(self):
        '''
        :return: 若shuffle为True，则每个epoch最后要将下标打乱顺序
        '''
        self.indices = np.arange(self.nb_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        
    def __getitem__(self, index):
        '''
        :param index:
        :return: 返回一个batch的 x,y
        '''
        indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        cur_batch_json = []      # 当前一个batch的json信息，包含img_name和解析
        for idx in indices:
            cur_batch_json.append(self.json_list[idx])
            
        X, y = self._get_batch_data(cur_batch_json)
        
        # X, y = self._generate_data(batch_json_names)

        return X, y


    # @abc.abstractmethod
    # def _generate_data(self, batch_json_names):
    #     pass


    @abc.abstractmethod
    def _get_batch_data(self, batch_json_info):
        pass


    def load_image_and_annotation(self, json_dict):
        res_num = []
        res_img = []
        
        json_info = list(json_dict.values())[0]
        img_name = list(json_dict.keys())[0]
        
        # 将 int 型 label 转换为 str 型
        int_labels = json_info['label']
        for i in int_labels:
            res_num.append(str(i))

        label_num = len(res_num)
        
        # 获取图像
        img_path = os.path.join(self.dataset, img_name)
        img = cv2.imread(img_path)
        
        for i in range(label_num):
            h = json_info["height"][i]
            w = json_info['width'][i]
            left = json_info['left'][i]
            top = json_info['top'][i]
            # img [y1:y2, x1:x2]
            curImg = img[int(top):int(top + h), int(left):int((left + w))]
            # img.shape: [height, width]
            if curImg.shape[0] != self.img_height:
                # resize: [width, height]
                curImg = cv2.resize(curImg, dsize=[int(curImg.shape[1] * self.img_height / curImg.shape[0]), self.img_height], interpolation=cv2.INTER_CUBIC)
 
            res_img.append(curImg)
            
        if len(res_img) == 1:
            res_img = self.process_image(res_img[0])
        elif len(res_img) > 1:
            tmp = res_img[0]
            for i in range(1, len(res_img)):
                tmp = np.concatenate((tmp, res_img[i]), axis=1)
            res_img = self.process_image(tmp)

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
    # def _generate_data(self, batch_json_names):
    #     # np.zeros(shape=(h, w))
    #     x = np.zeros((self.batch_size, self.img_width, self.img_height, self.channels), dtype=np.uint8)
    #     y = np.zeros((self.batch_size, self.label_len), dtype=np.uint8)
    #
    #     # print("x.shape", x.shape)
    #
    #     for idx, fn in enumerate(batch_json_names):
    #         img, label = self.load_image_and_annotation(fn)
    #         img = self.preprocess(img)
    #         # cv2.imshow("img", img)
    #         # cv2.waitKey(0)
    #         x[idx] = img
    #
    #         while len(label) < self.label_len:
    #             label += '-'
    #
    #         y[idx] = [self.characters.find(c) for c in label]
    #
    #     return [x, y, np.ones(self.batch_size) * int(self.timesteps - 2), np.ones(self.batch_size) * self.label_len], y

    
    def _get_batch_data(self, batch_json_info):
        x = np.zeros((self.batch_size, self.img_width, self.img_height, self.channels), dtype=np.float)
        y = np.zeros((self.batch_size, self.label_len), dtype=np.uint8)
        
        for idx, cur_json_info in enumerate(batch_json_info):
            # print(cur_json_info)
            img, label = self.load_image_and_annotation(cur_json_info)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            
            # print("处理后的图片尺寸：", img.shape)
            img = self.preprocess(img)
            x[idx] = img
            
            while len(label) < self.label_len:
                label += '-'
                
            y[idx] = [self.characters.find(c) for c in label]
            
        return [x, y, np.ones(self.batch_size) * int(self.timesteps - 2), np.ones(self.batch_size) * self.label_len], y
        
    
    
    
class Val_Generator(Generator):
    def _get_batch_data(self, batch_json_info):
        x = np.zeros((self.batch_size, self.img_width, self.img_height, self.channels), dtype=np.float)
        y = []
        
        for i, fn in enumerate(batch_json_info):
            img, word = self.load_image_and_annotation(fn)
            img = self.preprocess(img)
            x[i] = img
            y.append(word)
        return x, y


    

if __name__ == '__main__':
    g = TrainGenerator(dataset=cfg.o_trainSet,
                      json_file_path=cfg.o_trainJson,
                      batch_size=cfg.batch_size,
                      img_height=cfg.height,
                      img_width=cfg.width,
                      channels=cfg.nb_channels,
                      timesteps=cfg.timesteps,
                      label_len=cfg.label_len,
                      characters=cfg.characters)

























