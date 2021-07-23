import argparse
import multiprocessing


parser = argparse.ArgumentParser()

parser.add_argument('--h_valSet', type=str, default=r'../../tfKeras/tianchi/mchar_val')
parser.add_argument('--h_valJson', type=str, default=r'../../tfKeras/tianchi/mchar_val.json')
parser.add_argument('--h_valWasted', type=str, default=r'../../tfKeras/tianchi/val_wasted')

parser.add_argument('--h_trainSet', type=str, default=r'../../tfKeras/tianchi/mchar_train')
parser.add_argument('--h_trainJson', type=str, default=r'../../tfKeras/tianchi/mchar_train.json')
parser.add_argument('--h_trainWasted', type=str, default=r'../../tfKeras/tianchi/train_wasted')


parser.add_argument('--o_trainSet', type=str, default=r'../../mchar_train')
# parser.add_argument('--o_trainJson', type=str, default=r'../../mchar_train.json')     没有数据清洗过的json
parser.add_argument('--o_trainJson', type=str, default=r'./o_train_purified.json')     # 数据清洗过的json
parser.add_argument('--o_trainWasted', type=str, default=r'../../train_wasted')

parser.add_argument('--o_valSet', type=str, default=r'../../mchar_val')
# parser.add_argument('--o_valJson', type=str, default=r'../../mchar_val.json')
parser.add_argument('--o_valJson', type=str, default=r'./o_val_purified.json')     # 数据清洗过的json
parser.add_argument('--o_valWasted', type=str, default=r'../../val_wasted')


# 模型保存dir参数
parser.add_argument('--training_model_dir', type=str, default=r'./training_model')
parser.add_argument('--training_model_cp', type=str, default=r'training_model.{epoch:03d}-{loss:.2f}.h5')

parser.add_argument('--prediction_model_dir', type=str, default=r'./prediction_model')
parser.add_argument('--prediction_model_cp', type=str, default=r'prediction_model.{epoch:03d}-{loss:.2f}.h5')


parser.add_argument('--width', type=int, default=200)
parser.add_argument('--height', type=int, default=32)
parser.add_argument('--nb_channels', type=int, default=3)
parser.add_argument('--label_len', type=int, default=7)   # 最长的标签长度
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_CTN', 'CRNN'])
parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
parser.add_argument('--timesteps', type=int, default=50)
parser.add_argument('--dropout_rate', type=float, default=0.25)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--lr_reduction_factor', type=float, default=0.0001)
parser.add_argument('--nb_epochs', type=int, default=1)
parser.add_argument('--nb_workers', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--resume_training', type=bool, default=False)

parser.add_argument('--characters', type=str, default='0123456789-')

parser.add_argument('--base_dir', type=str, default=r'./tianchi')
parser.add_argument('--output_dir', type=str, default=r'./tianchi_result')
parser.add_argument('--save_best_only', type=bool, default=True)
parser.add_argument('--load_model_path', type=str, default='')
parser.add_argument('--tb_lob', type=str, default='tb_log')

cfg = parser.parse_args()


if __name__ == '__main__':
    print(type(cfg))





















