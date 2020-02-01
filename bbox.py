"""
Each function d⋆(P) (where ⋆ is one of x,y,h,w) is modeled as a linear function of the pool5 features of pro- posal P ,
denoted by φ5 (P ). (The dependence of φ5 (P ) on the image data is implicitly assumed.) Thus we have d⋆(P) = wT⋆φ5(P),
 where w⋆ is a vector of learnable model parameters. We learn w⋆ by optimizing the regu- larized least squares objective
 (ridge regression):
 根据论文所属  边框回归采用的 ridge 回归
 论文中的 共享feature_maps 需要存在磁盘上的 因为数据较大
 这里由于采用小数据集 这里就不存入磁盘了

"""

from sklearn.linear_model import Ridge
import joblib


def train_bbox(features, deltas):
    """

    :param features: 抽取出来的特征
    :param deltas: anchors 和 gt_boxes 的回归值
    :return:
    """
    clf = Ridge(alpha=0.1)
    clf.fit(features, deltas)
    # 将ridge模型进行存储
    joblib.dump(clf, './svm/bbox_reg.pkl')
    return clf
