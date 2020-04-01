from keras.optimizers import SGD
from model import get_model
from keras.layers import Input
import config as cfg
import time
from keras.utils import generic_utils
from flower_data import FlowerData
import tensorflow as tf
import numpy as np
from keras.callbacks import TensorBoard
import os


# tensorboard
def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=value)
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


# load config
classes_num = 2
im_size = cfg.IM_SIZE

# 得到训练数据生成器
flower_data = FlowerData('./data/fine_tune_list.txt')
g_train = flower_data.data_generator_wrapper()

# get model
input_tensor = Input(shape=im_size + (3,))
model = get_model(input_tensor, classes_num)

if os.path.exists('./logs/model_weights.h5'):
    model.load_weights('./logs/model_weights.h5', by_name=True)

optimizer = SGD(lr=1e-3)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

log_path = './logs'
callback = TensorBoard(os.path.join(log_path, '000'))
callback.set_model(model)

# define hyper parameters
epochs_num = 10
train_step = 0
iter_num = 0
epoch_length = flower_data.samples_num
best_loss = np.Inf

for epoch_num in range(epochs_num):

    progbar = generic_utils.Progbar(epoch_length)  # keras progress bar
    print('Epoch {}/{}'.format(epoch_num + 1, epochs_num))
    while True:
        start_time = time.time()
        X, Y, _ = next(g_train)
        loss = model.train_on_batch(X, Y)
        write_log(callback, ['Elapsed time', 'loss', 'best_loss'],
                  [time.time() - start_time, loss, best_loss], train_step)

        train_step += 1
        iter_num += 1
        progbar.update(iter_num, [('loss', loss), ('best_loss', best_loss)])

        if iter_num == epoch_length:
            if loss < best_loss:
                print('curr_loss: {%s} best_loss: {%s} update loss and save bester weights' % (loss, best_loss))
                best_loss = loss
                save_path = os.path.join(log_path, 'model_weights.h5')
                if os.path.exists(save_path):
                    os.remove(save_path)
                model.save_weights(save_path)
            iter_num = 0
            break
