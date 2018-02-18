#MINST data CNN classification

from __future__ import print_function
import argparse
import datetime
import os

import keras
from keras.callbacks import CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

#model variables
width = 28
height = 28
num_out_class = 10
batch_size = 50
epochs = 1

#argument parsing
arg_par = argparse.ArgumentParser(prog='MNIST CNN')
arg_par.add_argument('--aug', action='store_true', default=False, dest='data_aug')
arg_par.add_argument('--augstore', action='store_true', default=False, dest='store_aug_pic')

args = arg_par.parse_args()
print(args)
data_aug = args.data_aug
store_aug_pic = args.store_aug_pic

#model, training history and image save files and folders
date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
result_folder = os.path.join(os.getcwd(), 'MNIST_result')
image_folder = os.path.join(result_folder, 'MNIST_image_' + date_time)

if data_aug or store_aug_pic:
    save_file = 'mnist_aug_' + date_time + '_model.h5'
    history_file = 'mnist_aug_' + date_time + '_history.csv'
else:
    save_file = 'mnist_' + date_time + '_model.h5'
    history_file = 'mnist_' + date_time +'_history.csv'

save_point = os.path.join(result_folder, save_file)
history_point = os.path.join(result_folder, history_file)
csv_logger = CSVLogger(history_point, append=True, separator=';')

#data load
(train_data, train_label), (test_data, test_label) = mnist.load_data()

#convert images to tensors
train_data = train_data.reshape(train_data.shape[0], height, width, 1)
test_data = test_data.reshape(test_data.shape[0], height, width, 1)
input_shape = (height, width, 1)

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

#augmented training                
if data_aug or store_aug_pic:
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
    if store_aug_pic:
        print("Augmented training - data store enabled")
        if not os.path.isdir(image_folder):
            os.makedirs(image_folder)
        model.fit_generator(generator.flow(train_data, train_label,
                                        save_to_dir=image_folder,
                                        save_prefix=date_time,
                                        batch_size=batch_size),
                            epochs=epochs, 
                            verbose=2, 
                            validation_data=(test_data, test_label),
                            callbacks=[csv_logger])
    #just training
    else:
        print("Augmented training - no data store")
        model.fit_generator(generator.flow(train_data, train_label,
                                        batch_size=batch_size),
                            epochs=epochs, 
                            verbose=2, 
                            validation_data=(test_data, test_label),
                            callbacks=[csv_logger])
#non-augmented training    
else:
    print("Non-augmented training")
    model.fit(train_data, train_label,
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=(test_data, test_label),
                callbacks=[csv_logger])
                
#save the trained model
model.save(save_point)

#evaluate model
score = model.evaluate(test_data, test_label, verbose=1)
print('Loss:', score[0])
print('Acuracy:', score[1])

