"""
Each function d⋆(P) (where ⋆ is one of x,y,h,w) is modeled as a linear function of the pool5 features of pro- posal P ,
denoted by φ5 (P ). (The dependence of φ5 (P ) on the image data is implicitly assumed.) Thus we have d⋆(P) = wT⋆φ5(P),
 where w⋆ is a vector of learnable model parameters. We learn w⋆ by optimizing the regu- larized least squares objective
 (ridge regression):
 根据论文所属  边框回归采用的 ridge 回归
 论文中的 共享feature_maps 需要存在磁盘上的 因为数据较大
 这里由于采用小数据集 这里就不存入磁盘了

"""

import numpy as np
from sklearn.linear_model import Ridge
import joblib
from model import get_model, get_features_model
import config as cfg
from flower_data import FlowerData
from keras.layers import Input
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./weights/model_weights.h5', help='weights path')

# load config
classes_num = 2
im_size = cfg.IM_SIZE


def train_bbox(features, deltas):
    """

    :param features: 抽取出来的特征
    :param deltas: anchors 和 gt_boxes 的回归值
    :return:
    """
    clf = Ridge(alpha=0.1)
    clf.fit(features, deltas)
    joblib.dump(clf, './svm/bbox_reg.pkl')
    return clf


def main(args):
    # 得到训练数据生成器
    flower_data = FlowerData('./data/fine_tune_list.txt')
    g_train = flower_data.data_generator_wrapper()

    # get model
    input_tensor = Input(shape=im_size + (3,))
    model = get_model(input_tensor, classes_num)

    features_model = get_features_model(model)
    if not os.path.exists(args.weights):
        raise Exception('model weights not exists, please check it')
    features_model.load_weights(args.weights, by_name=True)

    # 这里由于采用小数据集 这里就不存入磁盘了
    # 论文中共享特征是先存入磁盘读取的
    epoch_length = flower_data.samples_num
    total_features = []
    total_rects = []
    for i in range(epoch_length):
        X, Y, rects = next(g_train)
        features = features_model.predict(X)
        total_features.append(features)
        total_rects.append(rects)
    total_features = np.asarray(total_features)
    total_rects = np.asarray(total_rects)
    train_bbox(total_features, total_rects)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
