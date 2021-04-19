
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from slim import ops
from slim import scopes
from slim import variables
from net2net import Net2Net
import pickle


EPOCH = 1
N_GPUS = 1

#GPU configurations
device_type = 'GPU'
devices = tf.config.experimental.list_physical_devices(device_type)
devices_names = [d.name.split('e:')[1] for d in devices]
strategy = tf.distribute.MirroredStrategy(devices=devices_names[:N_GPUS])

#Load data
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# split the train set into train and validation set and normalize both dataset
X_train=train_images/255.0
y_train=train_labels

X_valid=test_images/255.0
y_valid=test_labels

X_train=X_train.reshape(-1,32,32,3)
X_valid=X_valid.reshape(-1,32,32,3)

def trainer(model):
    log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return  model.fit(X_train, y_train, epochs=EPOCH, batch_size=64,
                            validation_split=0.1, shuffle = 1, verbose=2, callbacks=[tensorboard_callback])


def train_a_teacher_network():
    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=16, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu", input_shape=[32,32,3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        opt = tf.keras.optimizers.Adam()
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])
    print("Train a teacher...")
    history = trainer(model)
    test = model.evaluate(X_valid, y_valid, verbose=0)

    results = [history.history, { 'test_loss': test[0], 'test_accuracy': test[1]}]
    filename = "teacher_stats.pck"
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model

def train_a_student_network_wider(model_teacher):
    new_width_conv = 32
    l = 0 #first layer will be made wider
    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=new_width_conv, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu", input_shape=[32,32,3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        opt = tf.keras.optimizers.Adam()
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

    # initialize new weights and biases for the first (0) layer that has new width
    model = net_to_wider(model_teacher,model,l,new_width_conv)

    #train the  model
    print("Train a wider student network...")
    history = trainer(model)
    test = model.evaluate(X_valid, y_valid, verbose=0)

    results = [history.history, { 'test_loss': test[0], 'test_accuracy': test[1]}]
    filename = "st_wider_stats.pck"
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model

def train_a_student_network_deeper(model_teacher):

    l=3 #an additional fourth layer is added after the thrid layer, first layer is 0
    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu", input_shape=[32,32,3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        opt = tf.keras.optimizers.Adam()
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

    model = net_to_deeper(model_teacher,model,l)

    print("Train a deeper student network...")
    history = trainer(model)
    test = model.evaluate(X_valid, y_valid, verbose=0)

    results = [history.history, { 'test_loss': test[0], 'test_accuracy': test[1]}]
    filename = "st_deeper_stats.pck"
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model

def train_baseline_network_deeper():
    with strategy.scope():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu", input_shape=[32,32,3]),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, kernel_initializer='he_uniform', padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu", kernel_initializer='he_uniform'),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        opt = tf.keras.optimizers.Adam()
        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=opt,
                      metrics=["accuracy"])

    print("Train a baseline network...")
    history = trainer(model)
    test = model.evaluate(X_valid, y_valid, verbose=0)

    results = [history.history, { 'test_loss': test[0], 'test_accuracy': test[1]}]
    filename = "baseline_stats.pck"
    with open(filename, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model

def net_to_wider(model_teacher,model,l,new_width_conv):
        #Get weights of the teacher model
        ws = []
        bs = []
        for layer in model_teacher.layers:
            if layer.get_weights() != []:
                ws.append(layer.get_weights()[0])
                bs.append(layer.get_weights()[1])
        w1 = ws[l]
        b1 = bs[l]
        w2 = ws[l+1]
        n2n = Net2Net()
        nw1, nb1, nw2 = n2n.wider(w1, b1, w2, new_width_conv, True)
        ws[l] = nw1
        bs[l] = nb1
        ws[l+1] = nw2
        i = 0
        for layer in model.layers:
            if layer.get_weights() != []:
                layer.set_weights([ws[i],bs[i]])
                i += 1
        return model

def net_to_deeper(model_teacher,model,l):
        #Get weights of the teacher model
        ws = []
        bs= []
        for layer_t in model_teacher.layers:
            if layer_t.get_weights() != []:
                ws.append(layer_t.get_weights()[0])
                bs.append(layer_t.get_weights()[1])

        n2n = Net2Net()
        new_w, new_b = n2n.deeper(ws[l-1], True)
        ws.insert(l,new_w)
        bs.insert(l,new_b)

        i = 0
        for layer_s in model.layers:
            if layer_s.get_weights() != []:
                layer_s.set_weights([ws[i],bs[i]])
                i += 1

        return model

if __name__ == '__main__':
    # 1. Train a teacher network
    model_teacher = train_a_teacher_network()
    # 2. Train a student network (Net2Wider)
    model_st_wider = train_a_student_network_wider(model_teacher)
    # 3. Random pad (Net2Wider baseline)
    train_a_student_network_deeper(model_st_wider)
    # 4. Train a baseline network
    train_baseline_network_deeper()
