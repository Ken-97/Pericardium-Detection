import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import Callback



def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = keras.tf.where(keras.tf.equal(y_true, 1), y_pred, keras.tf.ones_like(y_pred))
        pt_0 = keras.tf.where(keras.tf.equal(y_true, 0), y_pred, keras.tf.zeros_like(y_pred))
        return -keras.mean(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1)) - keras.mean((1 - alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0))
    return focal_loss_fixed

"""
original code:
def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
            pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
            pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
            return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
"""    

# def auc(y_true, y_pred):
#     auc = keras.tf.metrics.auc(y_true, y_pred)[1]
#     keras.get_session().run(tf.local_variables_initializer())
#     return auc

# def auroc(y_true, y_pred):
#     return keras.tf.py_func(roc_auc_score, (y_true, y_pred), keras.tf.float32)

def unet(pretrained_weights = None,input_size = (256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    
    drop4_level1 = AveragePooling2D(pool_size=(1, 1))(drop4)
    drop4_level2 = AveragePooling2D(pool_size=(2, 2))(drop4)
    drop4_level3 = AveragePooling2D(pool_size=(4, 4))(drop4)
    drop4_level4 = AveragePooling2D(pool_size=(8, 8))(drop4)
    drop4_level1_Up = UpSampling2D(size=(1, 1))(drop4_level1)
    drop4_level2_Up = UpSampling2D(size=(2, 2))(drop4_level2)
    drop4_level3_Up = UpSampling2D(size=(4, 4))(drop4_level3)
    drop4_level4_Up = UpSampling2D(size=(8, 8))(drop4_level4)
    merge6 = concatenate([drop4, up6, drop4_level1_Up, drop4_level2_Up, drop4_level3_Up, drop4_level4_Up], axis = 3)
    
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model = ParallelModel(model, GPU_COUNT)
    # model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(alpha=.65, gamma=2)], metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(alpha=.65, gamma=2)], metrics = [auc])
    model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss(alpha=.65, gamma=2)], metrics = ['accuracy'])

    #model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


