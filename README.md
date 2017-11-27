# Artifical Neural Network for Classifying Handwritten Data

The following, simple, easy to follow C++ code implements code to train and run 
a feed-forward neural network to perform classification of 28x28 pixel images of handwritten charecters.

The training and evaulation datasets are taken from the MNIST database.
The training set contains 60,000 labeled examples and the test set contain 10,000 labeled examples.
The MNIST database is not distributed with the source distribution for the ann.
However, it can be obtained by running getdata.sh in the data/ directory of the source distribution.

After obtaining the training dataset, the neural network can be built, trained and run with the following
commands from the root directory of the source distribution.  

````bash
mkdir build
cmake ..
make
./ann_mnist
````
