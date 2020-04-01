import numpy as np
import cv2
import os
import config as cfg
from ss.selectivesearch import selective_search
from utils.bbox_overlaps import bbox_overlaps
from utils.bbox_transform import bbox_transform

im_size = cfg.IM_SIZE


class FlowerData(object):
    def __init__(self, flower_data_path, classes_num=2, random_shuffle=True):
        flower_filename = os.path.basename(flower_data_path)
        save_path = './data/%s_annotation.txt' % flower_filename.split('.')[0]
        self._save_path = save_path
        if not os.path.exists(save_path):
            self._parse_flower_data_path(flower_data_path)
        self._annotations = self._get_annotations(save_path)
        self.classes_num = classes_num
        self.samples_num = len(self._annotations)
        self._shuffle = random_shuffle
        if self._shuffle:
            self._random_shuffle()

    def _random_shuffle(self):
        x = np.random.permutation(self.samples_num)
        self._annotations = self._annotations[x]

    def _get_annotations(self, annotation_path):
        annotations = open(annotation_path).readlines()
        annotations = [annotation.strip() for annotation in annotations]
        return np.array(annotations)

    def _parse_flower_data_path(self, flower_data_path):
        annotations = open(flower_data_path).readlines()
        annotations = [annotation.strip() for annotation in annotations]
        list_file = open(self._save_path, 'w')
        for annotation in annotations:
            img_path, cls, gt_box = annotation.split(' ')
            list_file.write('%s %s,%s' % (img_path, gt_box, cls))
            list_file.write('\n')

    def _parse_annotation(self, annotation):
        lines = annotation.strip().split()
        image_path = lines[0]
        gt_boxes = [list(map(float, box.split(','))) for box in lines[1:]]
        return image_path, np.asarray(gt_boxes)

    def data_generator_wrapper(self, batch_size=1, is_svm=False):
        assert batch_size == 1, 'batch_size should be one'
        return self._data_generator(batch_size, is_svm)

    def _data_generator(self, batch_size, is_svm):
        i = 0
        n = self.samples_num
        while True:
            total_img_data = []
            total_labels = []
            total_deltas = []
            for b in range(batch_size):
                if i == 0:
                    self._random_shuffle()
                annotation = self._annotations[i]
                image_path, gt_boxes = self._parse_annotation(annotation)
                img = cv2.imread(image_path)
                # height/width/channel
                height, width, _ = img.shape
                # img resize
                img = cv2.resize(img, im_size, interpolation=cv2.INTER_CUBIC)

                # BGR -> RGB 做简单处理
                img = img[:, :, (2, 1, 0)]
                img = img.astype(np.float32)
                img = img / 255.
                # gt_box resize
                gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * (im_size[0] / width)
                gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * (im_size[1] / height)

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
                # 将 gt_boxes 添加进来
                for idx in range(len(gt_boxes)):
                    x1, y1, x2, y2 = gt_boxes[idx, 0:4]
                    x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                    # 裁剪后进行resize
                    crop_img = img[y1:y2, x1:x2, :]
                    try:
                        crop_img = cv2.resize(crop_img, im_size, interpolation=cv2.INTER_CUBIC)
                        selected_imgs.append(crop_img)
                        rects.append(gt_boxes[idx, 0:4])
                    except:
                        continue

                rects = np.asarray(rects)
                # cal iou
                overlaps = bbox_overlaps(rects, gt_boxes)
                # 选出与哪个gt_box iou最大的索引位置
                argmax_overlaps = np.argmax(overlaps, axis=1)
                # judge cls
                max_overlaps = np.max(overlaps, axis=1)
                threshold = cfg.THRESHOLD if is_svm else cfg.FINE_TUNE_THRESHOLD
                keep = np.where(max_overlaps >= threshold)[0]
                labels = np.empty(len(argmax_overlaps))
                if is_svm:
                    # 用 -1 填充
                    labels.fill(-1)
                    # bg_ids = np.where(max_overlaps < )
                    # ground - truth样本作为正样本  且IoU大于0.3的“hard negatives”，
                    # 背景
                    bg_ids = np.where(max_overlaps < threshold)[0]
                    labels[bg_ids] = 0
                    fg_ids = np.where(max_overlaps > 0.7)
                    labels[fg_ids] = gt_boxes[argmax_overlaps[fg_ids], 4]
                else:
                    labels.fill(0)
                    # 对于大于指定threshold 前景类别
                    labels[keep] = gt_boxes[argmax_overlaps[keep], 4]
                # to something
                deltas = bbox_transform(rects, gt_boxes[argmax_overlaps, 0:4])
                total_deltas.append(deltas)
                total_labels.append(labels)
                total_img_data.append(selected_imgs)
                i = (i + 1) % n
            total_img_data = np.concatenate(total_img_data, axis=0)
            total_labels = np.concatenate(total_labels, axis=0)
            total_deltas = np.concatenate(total_deltas, axis=0)
            yield total_img_data, total_labels, total_deltas