# MHBcT
#MNIST_CNN.py
Run as:
> python MNIST_CNN.py <arguments>

Dependencies:
TensorFlow
Keras
TensorBoard

Supported optional arguments:
--aug	- enable data augmentation
--arch	- export model architecture to json
--board	- enable TensorBoard logging
--model	- export model data
--hist	- export trainning history data
--hyp	- export hyperparameters


It is recommended to redirect console output to a file:
> python MNIST_CNN.py >console_out.txt 2>&1

If you want to use TensorBoard:
- before running the program, open TensorBoard in a separate command window:
> tensorboard --logdir=C:\...\MNIST_result
- if you want to model multiple runs, put the parent directory as the --logdir arguments
- open the TensorBoard dashboard in a browser:
localhost:6006