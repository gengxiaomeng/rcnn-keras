from PIL import Image
import numpy as np
import config as cfg
from ss.selectivesearch import selective_search
import joblib
import os
from keras.models import Model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import skimage
import cv2
import config as cfg
from model import get_model, get_features_model
from keras.layers import Input
import argparse
from utils.bbox_transform import bbox_transform_inv
from utils.nms import py_cpu_nms

parser = argparse.ArgumentParser()
parser.add_argument('--weights', default='./logs/model_weights.h5', help='weights path')

im_size = cfg.IM_SIZE
classes_num = 2


def show_rect(img_path, regions):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in regions:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()


def get_proposal(img):
    # BGR -> RGB 做简单处理
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img = img / 255.

    # regions 里面 是 x1, y1, x2, y2
    _, regions = selective_search(img, scale=200, sigma=0.9, min_size=50)

    rects = np.asarray([list(region['rect']) for region in regions])
    selected_imgs = []
    candidates = set()
    # 过滤掉一些框
    for r in rects:
        x1, y1, x2, y2 = r
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        if (x1, y1, x2, y2) in candidates:
            continue
        if (x2 - x1) * (y2 - y1) < 220:
            continue
        crop_img = img[y1:y2, x1:x2, :]
        # 裁剪后进行resize
        crop_img = cv2.resize(crop_img, im_size, interpolation=cv2.INTER_CUBIC)
        selected_imgs.append(crop_img)
        candidates.add((x1, y1, x2, y2))

    rects = [list(candidate) for candidate in candidates]
    return np.asarray(selected_imgs), np.asarray(rects)


def main():
    image_path = './data/2flowers/jpg/0/image_0561.jpg'
    img = cv2.imread(image_path)
    # height/width/channel
    height, width, _ = img.shape
    # img resize
    img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)

    imgs, rects = get_proposal(img)

    # get model
    input_tensor = Input(shape=im_size + (3,))
    model = get_model(input_tensor, classes_num)

    features_model = get_features_model(model)
    if not os.path.exists(args.weights):
        raise Exception('model weights not exists, please check it')
    features_model.load_weights(args.weights, by_name=True)

    features = features_model.predict_on_batch()

    # load svm and ridge
    svm_fit = joblib.load('./svm/svm.pkl')
    bbox_fit = joblib.load('./svm/bbox_train.pkl')

    svm_pred = svm_fit.predict(features)
    bbox_pred = bbox_fit.predict(features)

    keep = svm_pred[svm_pred != 0]

    # 取出预测是物体的anchors
    # svm_pred = svm_pred[keep]
    rects = rects[keep]
    bbox_pred = bbox_pred[keep]

    # 边框修复
    pred_boxes = bbox_transform_inv(rects, bbox_pred)

    # 非极大值抑制
    keep_ind = py_cpu_nms(pred_boxes, 0.5)
    #
    pred_boxes = pred_boxes[keep_ind, :]

    # pred_boxes[:, [2, 3]] = pred_boxes[:, [2, 3]] - pred_boxes[:, [0, 1]]
    pred_boxes[:, 2] = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_boxes[:, 3] = pred_boxes[:, 3] - pred_boxes[:, 1]
    # # show img
    show_rect(image_path, pred_boxes)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
