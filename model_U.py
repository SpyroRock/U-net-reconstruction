import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from tensorflow.keras.callbacks import TensorBoard

def create_model_U(pretrained_weights = None, input_size = (256,256,1), start_neurons=32):
    inputs = Input(input_size)

    #First Conv layer (Concatenate fourth) 
    conv1 = Conv2D(start_neurons * 1, 3, activation = 'relu', padding = 'same')(inputs)                          #cinv1 -> 512
    conv1 = Conv2D(start_neurons * 1, 3, activation = 'relu', padding = 'same')(conv1)                           #conv2 -> 512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)                                                                #pool1 -> 256

    #Second Conv layer (Concatenate third) 256 
    conv2 = Conv2D(start_neurons * 2, 3, activation = 'relu', padding = 'same')(pool1)                           #conv3 -> 256
    conv2 = Conv2D(start_neurons * 2, 3, activation = 'relu', padding = 'same')(conv2)                           #conv4 -> 256
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)                                                                #pool2 -> 128

    #Third Conv layer (Concatenate second) 128 
    conv3 = Conv2D(start_neurons * 4, 3, activation = 'relu', padding = 'same')(pool2)                           #conv5 -> 128
    conv3 = Conv2D(start_neurons * 4, 3, activation = 'relu', padding = 'same')(conv3)                           #conv6 -> 128
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)                                                                #pool3 -> 64

    #Fourth Cinv Layer (Concatenate first) 64 
    conv4 = Conv2D(start_neurons * 8, 3, activation = 'relu', padding = 'same')(pool3)                           #conv7 -> 64
    conv4 = Conv2D(start_neurons * 8, 3, activation = 'relu', padding = 'same')(conv4)                           #conv8 -> 64
    pool4 = MaxPool2D(pool_size=(2,2))(conv4)                                                                    #pool4 -> 32

    #Last Layer 32
    conv5 = Conv2D(start_neurons * 16, 3, activation = 'relu', padding = 'same')(pool4)                          #conv9  -> 32
    conv5 = Conv2D(start_neurons * 16, 3, activation = 'relu', padding = 'same')(conv5)                          #conv10 -> 32

    #Upsampling first Layer 32 
    up1 = Conv2D(start_neurons * 8, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv5)) #up1 conv11 -> 32
    merge1 = concatenate([conv4,up1], axis = 3)
    conv6 = Conv2D(start_neurons * 8, 3, activation = 'relu', padding = 'same')(merge1)                          #conv12     -> 32
    conv6 = Conv2D(start_neurons * 8, 3, activation = 'relu', padding = 'same')(conv6)                           #conv13     -> 32

    #Upsampling second layer 64 
    up2 = Conv2D(start_neurons * 4, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv6)) #up2 conv14 -> 64
    merge2 = concatenate([conv3,up2])
    conv7 = Conv2D(start_neurons * 4, 3, activation = 'relu', padding = 'same')(merge2)                          #conv15     -> 64
    conv7 = Conv2D(start_neurons * 4, 3, activation = 'relu', padding = 'same')(conv7)                           #conv16     -> 64
 
    #Upsampling third layer 128 
    up3 = Conv2D(start_neurons * 2, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv7)) #up3 conv17 -> 128
    merge3 = concatenate([conv2,up3])
    conv8 = Conv2D(start_neurons * 2, 3, activation = 'relu', padding = 'same')(merge3)                          #conv18     -> 128
    conv8 = Conv2D(start_neurons * 2, 3, activation = 'relu', padding = 'same')(conv8)                           #conv19     -> 128

    #Upsampling fourth layer 256 
    up4 = Conv2D(start_neurons * 1, 2, activation = 'relu', padding = 'same')(UpSampling2D(size = (2,2))(conv8)) #up4 conv20 -> 256
    merge4 = concatenate([conv1,up4])
    conv9 = Conv2D(start_neurons * 1, 3, activation = 'relu', padding = 'same')(merge4)                          #conv21     -> 256
    conv9 = Conv2D(start_neurons * 1, 3, activation = 'relu', padding = 'same')(conv9)                           #conv22     -> 256
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same')(conv9)                                           #conv23     -> 256

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)                                                         #conv24     -> 256
    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
