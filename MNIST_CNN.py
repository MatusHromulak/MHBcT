#MNIST data CNN classification

from __future__ import print_function
import argparse
import datetime
import json
import keras
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys
from time import time

class TimeLog(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.times = time()
        
    def on_train_end(self, logs=None):
        self.times = (time() - self.times)

def parse_arguments():
    #argument parsing
    arg_par = argparse.ArgumentParser(prog='MNIST CNN')
    arg_par.add_argument('--aug', action='store_true', default=False, dest='aug_e')
    arg_par.add_argument('--arch', action='store_true', default=False, dest='arch_se')
    arg_par.add_argument('--board', action='store_true', default=False, dest='board_se')
    arg_par.add_argument('--hist', action='store_true', default=False, dest='hist_se')
    arg_par.add_argument('--hyp', action='store_true', default=False, dest='hyp_se')
    arg_par.add_argument('--model', action='store_true', default=False, dest='mod_se')
    
    args = arg_par.parse_args()
    aug_e = args.aug_e          #enable data augmentation
    board_se = args.board_se    #enable Tensorboard logging
    mod_se = args.mod_se        #export model data
    hist_se = args.hist_se      #export trainning history data
    hyp_se = args.hyp_se        #export hyperparameters
    arch_se = args.arch_se      #export model architecture to json
    
    return aug_e, arch_se, board_se, hist_se, hyp_se, mod_se

def prepare_files_folders():
    #save file and folder names and paths preparation
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    res_fol = os.path.join(os.getcwd(), 'MNIST_result')

    #create folder for output data
    if not os.path.isdir(res_fol):
        os.makedirs(res_fol)
    
    return date_time, res_fol

def json_export(res_fol, model, date_time, iter_name):
    #export architecture to a json file
    arch_f = 'mnist_' + date_time + '_' + ''.join(iter_name) + '_architecture.json'
    arch_p = os.path.join(res_fol, arch_f)
    arch_string = model.to_json()
    with open(arch_p, 'w') as outfile:
        json.dump(arch_string, outfile)

def save_model(res_fol, model, date_time, iter_name):
    #save the trained model
    mod_f = 'mnist_' + date_time + '_' + ''.join(iter_name) + '_model.h5'
    mod_p = os.path.join(res_fol, mod_f)
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
    
def create_model(hyp_se, activation, bias_init, dropout, layers, loss, neurons, optim,
                pooling, in_shape, num_out_class, date_time, res_fol, iter_name):
    model = Sequential()
    
    if layers >= 2:
        model.add(Conv2D(neurons, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))
        model.add(Conv2D(neurons, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))
        if pooling == 'MaxPool':
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        if pooling == 'AvgPool':
            model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    
    if layers >= 4:
        model.add(Conv2D(neurons*2, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))
        model.add(Conv2D(neurons*2, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))
        if pooling == 'MaxPool':
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        if pooling == 'AvgPool':
            model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    
    if layers >= 6:
        model.add(Conv2D(neurons*4, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))
        model.add(Conv2D(neurons*4, kernel_size=(3, 3),
                        padding='same',
                        activation=activation,
                        use_bias=True,
                        bias_initializer=bias_init,
                        input_shape=in_shape))                    
        if pooling == 'MaxPool':
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        if pooling == 'AvgPool':
            model.add(AveragePooling2D(pool_size=(2, 2), padding='same'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation=activation,
                    use_bias=True,
                    bias_initializer=bias_init))
    model.add(Dropout(dropout))
    model.add(Dense(num_out_class, activation='softmax',
                    use_bias=True,
                    bias_initializer=bias_init))
    
    model.compile(loss=loss,
                    optimizer=optim,
                    metrics=['accuracy'])
    
    if hyp_se:
        mod_f = 'mnist_' + date_time + '_' + ''.join(iter_name) + '_model.txt'
        mod_p = os.path.join(res_fol, mod_f)
        with open(mod_p,'w') as out_h:
            std_stdout = sys.stdout
            sys.stdout = out_h
            mod_summ = str(model.summary())
            out_h.write(mod_summ)
            sys.stdout = std_stdout
        
    return model

def train_model(model, board_se, hist_se, date_time, res_fol, iter_name,
                aug_e, trn_dt, trn_lbl, batch_size, epochs, tst_dt, tst_lbl):
    
    #time callback
    time_cb = TimeLog()
    callback = [time_cb]
    
    #TensorBoard callback
    if board_se:
        log_fol = os.path.join(res_fol, 'mnist_' + date_time + '_' + ''.join(iter_name) + '_logs')
        if not os.path.isdir(log_fol):
            os.makedirs(log_fol)
        tensorboard = TensorBoard(log_dir=log_fol.format(time()))
        callback.append(tensorboard)

    #training history logger callback
    if hist_se:
        his_f = 'mnist_' + date_time + '_' + ''.join(iter_name) + '_history.csv'    
        his_p = os.path.join(res_fol, his_f)
        csv_logger = CSVLogger(his_p, append=True, separator=';')
        callback.append(csv_logger)
    
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
                            callbacks=callback)
    #non-augmented training    
    else:
        model.fit(trn_dt, trn_lbl,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(tst_dt, tst_lbl),
                    callbacks=callback)
    res_time = time_cb.times
    return model, res_time

def eval_model(model, tst_dt, tst_lbl, message, res_time):
    #evaluate model
    score = model.evaluate(tst_dt, tst_lbl, verbose=0)
    print(message)
    print('Loss: ' + str(score[0]) + ' Acuracy: ' + str(score[1]) + ' Time: ' + str(res_time))
    
def main():
    #fixed variables
    height = 28
    width = 28
    num_out_class = 10
    
    #hyperparameters
    activation = ['relu', 'tanh']
    batch_size = 50                 #0.08% of the dataset
    dropout = 0.5                   #https://arxiv.org/abs/1207.0580
    epochs = 7
    init_bias = 0.1                 #citation needed
    layers = [2, 4, 6]
    optimizer = ['SGD', 'Adam']     #citation needed
    learn_rate = 0.01               #citation needed
    loss = 'categorical_crossentropy'
    pooling = ['MaxPool', 'AvgPool']
    neurons = [20, 40, 60, 80]
    
    #parse command line arguments
    aug_e, arch_se, board_se, hist_se, hyp_se, mod_se = parse_arguments()
    
    #save file and folder names and paths preparation
    date_time, res_fol = prepare_files_folders()
    
    #read in and prepare input data to data. function arguments
    trn_dt, trn_lbl, tst_dt, tst_lbl, in_shape = process_data(height, width, num_out_class)
    
    #activatio function selection
    for a in activation:
        #iter_name - defines the configuration differentiating handle for file names
        #message - defines the configuration differentiating message for the command line output
        if a == 'relu':
            iter_name = ['r']
            message = ['Activation: relu']
        if a == 'tanh':
            iter_name = ['t']
            message = ['Activation: tanh']
        
        #number of layers selection
        for la in layers:
            iter_name.append(str(la))
            message.append(str('Layers: ' + str(la)))
            
            #optimizer function selection
            for op in optimizer:
                if op == 'SGD':
                    optim = keras.optimizers.SGD(lr=learn_rate)
                    iter_name.append('s')
                    message.append('Optimizer: SGD')
                if op == 'Adam':
                    optim = keras.optimizers.Adam(lr=learn_rate)
                    iter_name.append('a')
                    message.append('Optimizer: Adam')
                
                #pooling layer selection
                for p in pooling:
                    if p == 'MaxPool':
                        iter_name.append('m')
                        message.append('Pooling: MaxPooling')
                    if p == 'AvgPool':
                        iter_name.append('a')
                        message.append('Pooling: AveragePooling')
                    
                    #number of neurons selection
                    for n in neurons:
                        iter_name.append(str(n))
                        message.append(str('Neurons: ' + str(n)))
                        
                        #create model
                        bias_init = keras.initializers.Constant(value=init_bias)
                        model = create_model(hyp_se, a, bias_init, 
                                            dropout, la, loss, n, 
                                            optim, p, in_shape, num_out_class, 
                                            date_time, res_fol, iter_name)
                        
                        #export model architecture
                        if arch_se:
                            json_export(res_fol, model, date_time, iter_name)
                        
                        #train model
                        model, res_time = train_model(model, board_se, hist_se, date_time, res_fol, iter_name,
                                            aug_e, trn_dt, trn_lbl, batch_size, epochs, tst_dt, tst_lbl)
                        
                        #save model architecture, weights, training configuration and optimizer state
                        if mod_se:
                            save_model(res_fol, model, date_time, iter_name)
                        
                        #evaluate model
                        eval_model(model, tst_dt, tst_lbl, message, res_time)
                        
                        iter_name.pop()
                        message.pop()
                    iter_name.pop()
                    message.pop()
                iter_name.pop()
                message.pop()
            iter_name.pop()
            message.pop()
        iter_name.pop()
        message.pop()

#run
if __name__ == "__main__":
    main()
