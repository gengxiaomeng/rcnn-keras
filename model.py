from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout


def get_model(input_tensor, classes_num=20, rate=.5, trainable=True):
    vgg16 = VGG16(input_tensor=input_tensor, include_top=False)
    if not trainable:
        for layer in vgg16.layers:
            layer.trainable = False
    share_features = vgg16.output
    x = Flatten()(share_features)
    x = Dense(4096, name='fc1')(x)
    x = Dropout(rate=rate)(x)
    x = Dense(4096, name='fc2')(x)
    x = Dropout(rate=rate)(x)
    # 添加一类背景
    output = Dense(classes_num + 1, activation='softmax', name='cls_output')(x)
    model = Model(vgg16.input, output)
    return model


def get_features_model(model):
    model = Model(model.input, model.get_layer('fc2').output)
    return model
