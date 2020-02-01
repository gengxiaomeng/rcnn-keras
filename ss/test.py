import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as mpatches
from ss.selectivesearch import selective_search
import numpy as np


def main():
    img_path = 'study.jpg'
    img = Image.open(img_path)
    img_data = np.asarray(img)
    # perform selective search
    img_lbl, regions = selective_search(img_data)
    # 创建候选框集合candidate
    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])

    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = plt.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in candidates:
        print(x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)

    plt.show()

if __name__ == '__main__':
    main()
