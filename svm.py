from model import get_model, get_features_model
from flower_data import FlowerData
from keras.layers import Input
import argparse
import config as cfg
import os
from bbox import train_bbox
from sklearn.svm import SVC
import numpy as np
import joblib

np.set_printoptions(threshold=np.inf)

svm_path = './svm'
if not os.path.exists(svm_path):
    os.mkdir(svm_path)

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./logs/model_weights.h5', help='weights path')

# load config
classes_num = 2
im_size = cfg.IM_SIZE

# 得到训练数据生成器
flower_data = FlowerData('./data/fine_tune_list.txt', random_shuffle=False)
g_train = flower_data.data_generator_wrapper(is_svm=True)

epoch_length = flower_data.samples_num


def train_svm(features, y):
    features_ = features[y >= 0]
    y_ = y[y >= 0]

    # 难负例挖掘
    # 对于目标检测（object detection）问题，所谓的 hard-negative mining 针对的是训练集中的
    # negative training set（对于目标检测问题就是图像中非不存在目标的样本集合），
    # 对该负样本集中的每一副图像（的每一个可能的尺度），应用滑窗（sliding window）技术。
    # 对每次滑窗捕获的图像区域，计算该区域的 HOG 描述子，并作为分类器的输入。
    # 如果预定义的分类器将其错误地在其中检测出对象，也即 FP（false-positive，伪正），
    # 记录该 FP patch 对应的特征向量及分类器给出的概率。
    # Y_hard = Y[Y < 0]

    features_hard = features[y < 0]
    pred_last = -1
    pred_now = 0
    while pred_now > pred_last:
        clf = SVC(probability=True)
        clf.fit(features_, y_)
        pred_ = clf.predict(features_hard)
        pred_prob = clf.predict_proba(features_hard)

        # 分类错误的样本
        Y_new_hard = pred_prob[pred_ > 0][:, 1]
        features_new_hard_ = features_hard[pred_ > 0]
        index_new_hard = range(Y_new_hard.shape[0])
        # 如果难负例样本过少，停止迭代
        if Y_new_hard.shape[0] // 10 < 1:
            break
        # 统计分类正确的数量
        count = pred_[pred_ == 0].shape[0]
        pred_last = pred_now

        # 计算新的测试正确率
        pred_now = count / features_hard.shape[0]
        idx = np.argsort(Y_new_hard)[::-1][0:len(Y_new_hard) // 10]
        y_ = np.concatenate([y_, np.zeros(len(idx), dtype=np.int32)], axis=0)
        for i in idx:
            features_list = features_.tolist()
            features_list.append(features_new_hard_[i])
            features_ = np.asarray(features_list)
            features_hard_list = features_hard.tolist()
            features_hard_list.pop(index_new_hard[i])
            features_hard = np.asarray(features_hard_list)
    # 将clf序列化，保存svm分类器
    joblib.dump(clf, './svm/svm.pkl')


def main(args):
    # get model
    input_tensor = Input(shape=im_size + (3,))
    model = get_model(input_tensor, classes_num)

    features_model = get_features_model(model)
    if not os.path.exists(args.weights):
        raise Exception('model weights not exists, please check it')
    features_model.load_weights(args.weights, by_name=True)

    # 这里由于采用小数据集 这里就不存入磁盘了
    # 论文中共享特征是先存入磁盘读取的

    total_features = []
    total_rects = []
    total_Y = []
    for i in range(epoch_length):
        X, Y, rects = next(g_train)
        if np.isnan(rects).any():
            print(i)
        features = features_model.predict(X)
        total_features.append(features)
        total_rects.append(rects)
        total_Y.append(Y)
    total_features = np.concatenate(total_features, axis=0)
    total_rects = np.concatenate(total_rects, axis=0)
    total_Y = np.concatenate(total_Y, axis=0)
    # train svm
    train_svm(total_features, total_Y)
    # train bbox
    train_bbox(total_features, total_rects)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
