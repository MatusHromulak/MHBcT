The project consists of four files:
MNIST_CNN.py
CIFAR10_CNN.py
fMRI_CNN.py
fMRI_matfile.py

The program fMRI_matfile converts the files available at: http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/ to .npy files containing cleaned and formated numpy arrays. The program expects the files to be available in the folder ./fMRI_data. The program stores the output data in the same folder.
Run as:
> python fMRI_matfile.py

The programs MNIST_CNN.py, CIFAR10_CNN.py, fMRI_CNN.py implement convolutional neural networks. MNIST and CIFAR download their data for analysis from the internet on the fly. fMRI expects the files provided by fMRI_matfile.py to be available in the folder ./fMRI_data.
Run as:
> python MNIST_CNN.py <arguments>
> python CIFAR10_CNN.py <arguments>
> python fMRI_CNN.py <arguments>

Dependencies:
Python 3.X
TensorFlow
Keras
TensorBoard

Supported optional arguments:
--aug	- enable data augmentation
--arch	- export model architecture to json
--board	- enable TensorBoard logging
--model	- export the trained model data
--hist	- export training history data
--hyp	- export hyper-parameters

For redirecting console output to a file run as:
> python MNIST_CNN.py >console_out.txt 2>&1

If you want to use TensorBoard:
- before running the program, open TensorBoard in a separate command window:
> tensorboard --logdir=C:\...\MNIST_result
- if you want to model multiple runs, put the parent directory as the --logdir argument
- open the TensorBoard dashboard in a browser:
localhost:6006

The program creates a folder (./MNIST_result, ./CIFAR10_result, ./fMRI_result) where all output files are stored.
The created files follow a naming convention: <dataset>_<date in YYYYMMDD format>_<time in HHmmss format>_<model configuration>_<output file type>.xxx
For the model configuration, all changeable parameters provide the first letter of their value. A test with the tanh activation, 4 convolutional layers, SGD optimizer, average pooling and 32 neurons would be encoded as t4sa32.

The parameters produce the following files:
--arch	- export model architecture to json
	Example file: mnist_20180131_125959_r2sa16_architecture.json
--board	- enable TensorBoard logging
	Creates folder: mnist_20180131_125959_r2sa16_logs
--model	- export the trained model data
	Example file: mnist_20180131_125959_r2sa16_model.h5
--hist	- export training history data
	Example file: mnist_20180131_125959_r2sa16_history.csv
--hyp	- export hyper-parameters
	Example file: mnist_20180131_125959_r2sa16_model.txt