#CIFAR10 data CNN classification

from __future__ import print_function
import argparse
import datetime
import json
import os

import keras
from keras.callbacks import CSVLogger
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#model variables
num_out_class = 10
batch_size = 50
epochs = 10

#argument parsing
arg_par = argparse.ArgumentParser(prog='CIFAR10 CNN')
arg_par.add_argument('--aug', action='store_true', default=False, dest='aug_enable')
arg_par.add_argument('--augstore', action='store_true', default=False, dest='aug_store_enable')
arg_par.add_argument('--arch', action='store_true', default=False, dest='arch_save_enable')
arg_par.add_argument('--model', action='store_true', default=False, dest='model_save_enable')

args = arg_par.parse_args()
aug_enable = args.aug_enable                #enable data augmentation
aug_store_enable = args.aug_store_enable    #enable data augmentation and store the image files
model_save_enable = args.model_save_enable  #export model data
arch_save_enable = args.arch_save_enable    #export model architecture to json

#save file and folder names and paths preparation
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_folder = os.path.join(os.getcwd(), 'CIFAR10_result')
image_folder = os.path.join(result_folder, 'CIFAR10_image_' + date_time)

if aug_enable or aug_store_enable:
    model_file = 'cifar10_aug_' + date_time + '_model.h5'
    history_file = 'cifar10_aug_' + date_time + '_history.csv'
else:
    model_file = 'cifar10_' + date_time + '_model.h5'
    history_file = 'cifar10_' + date_time +'_history.csv'

if arch_save_enable:
    arch_file = 'cifar10_' + date_time + '_architecture.json'
    arch_point = os.path.join(result_folder, arch_file)
    
model_point = os.path.join(result_folder, model_file)
history_point = os.path.join(result_folder, history_file)
csv_logger = CSVLogger(history_point, append=True, separator=';')

#data load
(train_data, train_label), (test_data, test_label) = cifar10.load_data()

#convert images to tensors
#train_data = train_data.reshape(train_data.shape[0], height, width, 1)
#test_data = test_data.reshape(test_data.shape[0], height, width, 1)
input_shape = train_data.shape[1:]

#normalize data
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
train_data /= 255
test_data /= 255

#convert class vector (int) to binary class - one hot encoding
train_label = keras.utils.to_categorical(train_label, num_out_class)
test_label = keras.utils.to_categorical(test_label, num_out_class)

#model start
model = Sequential()
b_init = keras.initializers.Constant(value=0.05)
model.add(Conv2D(32, kernel_size=(3, 3),
                padding='same',
                activation='relu',
                use_bias=True,
                bias_initializer=b_init,
                input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3),
                padding='same',
                activation='relu',
                use_bias=True,
                bias_initializer=b_init,
                input_shape=input_shape))         
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(Flatten())
model.add(Dense(1024, activation='relu',
                use_bias=True,
                bias_initializer=b_init))
model.add(Dropout(0.25))
model.add(Dense(num_out_class, activation='softmax',
                use_bias=True,
                bias_initializer=b_init))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
print(model.summary())
#model end

#create folder for output data
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

#export architecture to a json file
if arch_save_enable:
    arch_string = model.to_json()
    with open(arch_point, 'w') as outfile:
        json.dump(arch_string, outfile)

#augmented training
if aug_enable or aug_store_enable:
    #definition of data augmentation arguments to be used
    generator = ImageDataGenerator(
        #set input mean to 0 over the dataset, feature-wise
        featurewise_center=False,
        #set each sample mean to 0
        samplewise_center=False,
        #divide inputs by std of the dataset, feature-wise
        featurewise_std_normalization=False,
        #divide each input by its std
        samplewise_std_normalization=False,
        #epsilon for ZCA whitening
        zca_epsilon=1e-6,
        #apply ZCA whitening
        zca_whitening=False,
        #degree range for random rotations
        rotation_range=0,
        #range for random horizontal shifts
        width_shift_range=0.,
        #range for random vertical shifts
        height_shift_range=0.,
        #shear Intensity (Shear angle in counter-clockwise direction as radians)
        shear_range=0.,
        #range for random zoom
        zoom_range=0.,
        #range for random channel shifts
        channel_shift_range=0.,
        #points outside the boundaries of the input are filled according to the given mode ("constant", "nearest", "reflect", "wrap")
        fill_mode='nearest',
        #value used for points outside the boundaries when fill_mode = "constant"
        cval=0.,
        #randomly flip inputs horizontally
        horizontal_flip=False,
        #randomly flip inputs vertically
        vertical_flip=False,
        #rescaling factor
        rescale=None,
        #function that will be implied on each input
        preprocessing_function=None)

    #compute transoformed data set, required for featurewise_center, featurewise_std_normalization, zca_whitening
    generator.fit(train_data)

    #if storing augemnted training images during training is enabled
    if aug_store_enable:
        print("Augmented training - data storing enabled")
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)
        model.fit_generator(generator.flow(train_data, train_label,
                                        save_to_dir=image_folder,
                                        save_prefix=date_time,
                                        batch_size=batch_size),
                            epochs=epochs, 
                            verbose=1, 
                            validation_data=(test_data, test_label),
                            callbacks=[csv_logger])
    #just training
    else:
        print("Augmented training - no data storing")
        model.fit_generator(generator.flow(train_data, train_label,
                                        batch_size=batch_size),
                            epochs=epochs, 
                            verbose=1, 
                            validation_data=(test_data, test_label),
                            callbacks=[csv_logger])
#non-augmented training    
else:
    print("Non-augmented training")
    model.fit(train_data, train_label,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(test_data, test_label),
                callbacks=[csv_logger])
                
#save the trained model
if model_save_enable:
    model.save(model_point)

#evaluate model
score = model.evaluate(test_data, test_label, verbose=0)
print('Loss:', score[0])
print('Acuracy:', score[1])
