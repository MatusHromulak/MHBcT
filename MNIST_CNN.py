#MNIST data CNN classification

from __future__ import print_function
import argparse
import datetime
import json
import keras
from keras.callbacks import CSVLogger
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys

def parse_arguments():
    #argument parsing
    arg_par = argparse.ArgumentParser(prog='MNIST CNN')
    arg_par.add_argument('--aug', action='store_true', default=False, dest='aug_e')
    arg_par.add_argument('--arch', action='store_true', default=False, dest='arch_se')
    arg_par.add_argument('--model', action='store_true', default=False, dest='mod_se')
    
    args = arg_par.parse_args()
    aug_e = args.aug_e      #enable data augmentation
    mod_se = args.mod_se    #export model data
    arch_se = args.arch_se  #export model architecture to json
    
    return aug_e, mod_se, arch_se

def prepare_files_folders(aug_e):
    #save file and folder names and paths preparation
    d_t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    res_fol = os.path.join(os.getcwd(), 'MNIST_result')
    img_fol = os.path.join(res_fol, 'MNIST_image_' + d_t)

    if aug_e:
        mod_f = 'mnist_aug_' + d_t + '_model.h5'
    else:
        mod_f = 'mnist_' + d_t + '_model.h5'

    mod_p = os.path.join(res_fol, mod_f)
    arch_f = 'mnist_' + d_t + '_architecture.json'
    arch_p = os.path.join(res_fol, arch_f)

    #create folder for output data
    if not os.path.isdir(res_fol):
        os.makedirs(res_fol)
    
    return d_t, res_fol, mod_p, arch_p

def json_export(arch_p, model):
    #export architecture to a json file
    arch_string = model.to_json()
    with open(arch_p, 'w') as outfile:
        json.dump(arch_string, outfile)

def save_model(mod_p, model):
    #save the trained model
    model.save(mod_p)

def process_data(height, width, num_out_class):
    #load data as function arguments
    (trn_dt, trn_lbl), (tst_dt, tst_lbl) = mnist.load_data()

    #convert images to tensors
    trn_dt = trn_dt.reshape(trn_dt.shape[0], height, width, 1)
    tst_dt = tst_dt.reshape(tst_dt.shape[0], height, width, 1)
    in_shape = (height, width, 1)

    #normalize data
    trn_dt = trn_dt.astype('float32')
    tst_dt = tst_dt.astype('float32')
    trn_dt /= 255
    tst_dt /= 255

    #convert class vector (int) to binary class - one hot encoding
    trn_lbl = keras.utils.to_categorical(trn_lbl, num_out_class)
    tst_lbl = keras.utils.to_categorical(tst_lbl, num_out_class)
    
    return trn_dt, trn_lbl, tst_dt, tst_lbl, in_shape

def create_logger(aug_e, d_t, res_fol, it_n):
    if aug_e:
        his_f = 'mnist_aug_' + d_t + '_' + it_n + '_history.csv'
    else:
        his_f = 'mnist_' + d_t + '_' + it_n +'_history.csv'
        
    his_p = os.path.join(res_fol, his_f)
    return(CSVLogger(his_p, append=True, separator=';'))
    
def create_model(in_shape, bias_init, num_out_class, d_t, res_fol, it_n):
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    use_bias=True,
                    bias_initializer=bias_init,
                    input_shape=in_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(64, kernel_size=(3, 3),
                    padding='same',
                    activation='relu',
                    use_bias=True,
                    bias_initializer=bias_init,
                    input_shape=in_shape))         
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
                    use_bias=True,
                    bias_initializer=bias_init))
    model.add(Dropout(0.25))
    model.add(Dense(num_out_class, activation='softmax',
                    use_bias=True,
                    bias_initializer=bias_init))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    mod_f = 'mnist_' + d_t + '_' + it_n + '_model.txt'
    mod_p = os.path.join(res_fol, mod_f)
    with open(mod_p,'w') as out_h:
        std_stdout = sys.stdout
        sys.stdout = out_h
        mod_summ = str(model.summary())
        out_h.write(mod_summ)
        sys.stdout = std_stdout
        
    return model

def train_model(model, aug_e, trn_dt, trn_lbl, batch_size, epochs, tst_dt, tst_lbl, csv_logger):
    #augmented training
    if aug_e:
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
            rotation_range=10,
            #range for random horizontal shifts
            width_shift_range=0.2,
            #range for random vertical shifts
            height_shift_range=0.2,
            #shear Intensity (Shear angle in counter-clockwise direction as radians)
            shear_range=0.3,
            #range for random zoom
            zoom_range=0.3,
            #range for random channel shifts
            channel_shift_range=0.3,
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
        generator.fit(trn_dt)
        
        #training
        model.fit_generator(generator.flow(trn_dt, trn_lbl,
                                        batch_size=batch_size),
                            epochs=epochs, 
                            verbose=0, 
                            validation_data=(tst_dt, tst_lbl),
                            callbacks=[csv_logger])
    #non-augmented training    
    else:
        model.fit(trn_dt, trn_lbl,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(tst_dt, tst_lbl),
                    callbacks=[csv_logger])
    return model

def eval_model(model, tst_dt, tst_lbl, message):
    #evaluate model
    score = model.evaluate(tst_dt, tst_lbl, verbose=0)
    print(message)
    print('Loss: ' + str(score[0]) + ' Acuracy: ' + str(score[1]))
    
def main():
    #fixed variables
    height = 28
    width = 28
    num_out_class = 10
    
    #hyperparameters
    activation = ['relu', 'tanh', 'sigmoid']
    batch_size = [50, 75, 100]
    bias_constant = [0.05, 0.1, 0.2]
    dropout = [0.1, 0.25, 0.5]
    epochs = [5, 7, 10]
    layers = [2, 4, 6]
    #learning_decay = [0.0, 0.01, 0.02, 0.04]
    learning_rate = [0.01, 0.05, 0.1, 0.2]
    loss = ['mean_squared_error', 'categorical_crossentropy']
    neurons = [10, 20, 30, 40, 50, 60, 70, 80]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    pooling = ['MaxPool', 'AvgPool']
    seed = [32, 64, 128]    #np.random.seed(seed[0])
    
    #parse command line arguments
    aug_e, mod_se, arch_se = parse_arguments()
    #aug_e, mod_se, arch_se = [bool(x) for x in comm_line_args]
    
    #save file and folder names and paths preparation
    d_t, res_fol, mod_p, arch_p = prepare_files_folders(aug_e)
    #d_t, res_fol, mod_p, arch_p = [str(x) for x in file_data]
    
    #read in and prepare input data to data. function arguments
    trn_dt, trn_lbl, tst_dt, tst_lbl, in_shape = process_data(height, width, num_out_class)
    
    for x in range (0, 1):
        message = str(x)
        
        #prepare training history logger
        it_n = str(x)
        csv_logger = create_logger(aug_e, d_t, res_fol, it_n)
        
        #create model
        bias_init = keras.initializers.Constant(value=0.05)
        model = create_model(in_shape, bias_init, num_out_class, d_t, res_fol, it_n)
            
        model = train_model(model, aug_e,
                            trn_dt, trn_lbl,
                            batch_size, epochs,
                            tst_dt, tst_lbl,
                            csv_logger)
        
        eval_model(model, tst_dt, tst_lbl, message)
    
    #export model architecture
    if arch_se:
        json_export(arch_p, model)
        
    #export model
    if mod_se:
        save_model(mod_p, model)

#run
if __name__ == "__main__":
    main()
