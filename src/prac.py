from config import cfg
import os
import json
import cv2
import shutil


def data_wash(img_dir, json_path, purified_json_path, wasted_dir):
	all_json_info = json.load(open(json_path))
	img_names = list(all_json_info.keys())
	for name in img_names:
		cur_json = all_json_info[name]
		img_path = os.path.join(img_dir, name)
		img = cv2.imread(img_path)

		flag = resize_check(img.shape, cur_json)
		if flag:
			save_json_info(name, cur_json, purified_json_path)
		else:
			shutil.move(img_path, wasted_dir)
			print("error img:", name, cur_json)


def resize_check(img_shape, cur_json):
	height = img_shape[0]
	width = img_shape[1]
	for i in range(len(cur_json['label'])):
		h = int(cur_json["height"][i])
		w = int(cur_json['width'][i])
		left = int(cur_json['left'][i])
		top = int(cur_json['top'][i])

		if h < 0 or w < 0 or left < 0 or top < 0 or (top+h) > height or (left+w) > width:
			return False
	return True


def save_json_info(name, json_info, purified_json_path):
	save_dict = {name: json_info}
	with open(purified_json_path, 'a') as f:
		save_dict = json.dumps(save_dict)
		f.write(save_dict+"\n")



if __name__ == '__main__':
	pass
    # data_wash(cfg.h_trainSet, cfg.h_trainJson, r'./train_purified.json', cfg.h_trainWasted)
    # data_wash(cfg.h_valSet, cfg.h_valJson, r'./val_purified.json', cfg.h_valWasted)



















