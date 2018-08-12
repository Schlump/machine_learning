from __future__ import absolute_import, division, print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import os
import numpy as np
import subprocess
from dask.array.image import imread
from sklearn.model_selection import train_test_split

def calc_ndvi(X):
    ndvi = (X[:,:,:,7] - X[:,:,:,3]) / (X[:,:,:,7] + X[:,:,:,3])
    return ndvi


def min_max_scaling(X):
    x_min = X.min(axis=(1, 2), keepdims=True)
    x_max = X.max(axis=(1, 2), keepdims=True)
    x = (X - x_min)/(x_max-x_min)
    return x


def z_score(X):
    x = X - X.mean(axis=(1,2),keepdims=True)
    std = X.std(axis=(1,2),keepdims=True)
    std[np.where(std == 0)] = 1
    x = x / std

    return x

def read_data_from_dir(dataDir,extension):
    """ Read a stack of images located in subdirectories into a dask array
        returning X (array of data) and y (array of labels)
    """
    X = np.concatenate([imread(dataDir+subdir+'/*.'+extension).compute() for subdir in os.listdir(dataDir)]) 

    filesdict = {}

    for subdir in sorted(os.listdir(dataDir)):
        files = next(os.walk(dataDir+subdir))[2]
        files = len([fi for fi in files if fi.endswith("."+extension)])
        filesdict.update({subdir:files})

    if sum(filesdict.values()) != X.shape[0]:
        
        raise ValueError('Images and Labels does not Match')

    else:
        y = np.zeros([X.shape[0],1], dtype=np.uint8)
        i = 0
        imagelist = []
        for category in list(filesdict.keys()):
            z = filesdict[category]
            y[sum(imagelist):sum(imagelist)+z] = i
            imagelist.append(z)
            i += 1   

    return X,y
        

def replace_missingvalues_bandmean(X):
    """ Read a numpy array and check for missing values (zeros) and 
        replace zeros with band mean
    """
    if X.ndim != 4:
        raise ValueError('Input not valid, no [pic, row, column, band] data format')
        
    zeros = np.where(X[:,:,:] == 0)

    bandmean = {}

    for i in sorted(np.unique(zeros[3])):
        bandmean.update({i:np.mean(X[:,:,:,i])})
        
    for i in range(0,len(zeros[0])):
        pic, row, column, band = zeros[0][i],zeros[1][i],zeros[2][i],zeros[3][i]
        mean = bandmean.get(band)
        X[pic,row,column,band] = int(mean)
        
    return X
        


def my_model(X):

    input_shape = X.shape[1],X.shape[2],X.shape[3]
    
    lr = 0.001
    
    if X.shape[3] >= 5:
        lr = 0.0005
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    
    
    optimizer = optimizers.adam(lr=lr)


    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


    return model


def VGG_like_model(X):
    
    input_shape = X.shape[1],X.shape[2],X.shape[3]
    
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


def VGG_like_model_batchnorm(X):
    
    input_shape = X.shape[1],X.shape[2],X.shape[3]
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model



def pretrained_VGG16(X):
    
    input_shape = X.shape[1],X.shape[2],X.shape[3]

    pretrained =  VGG16(input_shape=input_shape,weights='imagenet', include_top=False,pooling=max)

    inputs = Input(shape=input_shape,name = 'image_input')

    output_vgg16_conv = pretrained(inputs)

    #Add the fully-connected layers 
    x = Flatten(name='flatten')(output_vgg16_conv)
    x = Dense(512, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(10, activation='softmax', name='predictions')(x)

    #Create own model 
    model = Model(input=inputs, output=x)


    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    return model

    
def get_data(bands=[1,2,3]):

    dataDir = '/home/schlupp/Data/Sentinel2_Trainingdata/Full_Data/'
    X,y = read_data_from_dir(dataDir,'tif')

    num_classes = 10
    X = X.astype('float32')
    print('Replacing missing values')
    X = X[:,:,:,bands]
    X = replace_missingvalues_bandmean(X)

    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    std = X_train.std(axis=(0,1,2))
    std[np.where(std == 0)] = std.mean()
    mean = X_train.mean(axis=(0,1,2))
    
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X = None
    y = None

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return X_train,X_test,y_train,y_test,num_classes

