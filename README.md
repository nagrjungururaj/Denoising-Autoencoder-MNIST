# Unsupervised-learning-on-MNIST
Information on how to use auto-encoders on the MNIST dataset

# Overview
The code samples and short explanations helps a user to understand how to use an autoencoder practically and build on the ideas presented here. Autoencoders have gained a lot of popularity in the field of image processing and computer vision in recent years. They are specially used for reconstruction of images and segmentation problems.

# General info about autoencoders
Motivating from neural network, the architecture of autoencoders resembles that of the previous. Autoencoders generally have a 'encoder' unit which is similar to the hidden layer of a neural network with some weights, and a 'decoder' unit where the weights of decoder unit are in such a way that the dimensions / pixels of input image/images are recovered. For a clearer picture, see the figure in the link below.

https://user-images.githubusercontent.com/31846605/30348343-4a2676ec-980f-11e7-965a-82d2c59c46b4.png

# Must Read
Vincent, Pascal, et al. “Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion.” The Journal of Machine Learning Research 11 (2010): 3371-3408. This paper is useful and covers all aspects of autoencoder and unsupervised deep learning.

You can download the paper here : http://jmlr.csail.mit.edu/papers/volume11/vincent10a/vincent10a.pdf

# Denoising is the thing !
So what is denoising ? Denoising actually is 'NOT' removal of noise but instead the opposite. So what does this have to do with autoencoders ? Interestingly, corrupting the input images before feeding them to the encoder enchances the training and quality of reconstruction of the input images. But wait, how is this possible ? The trick is this type of noisy training optimizes the loss between the output of the decoder with noisy inputs and the original input images. Generally three types of noise are used, 

1. Additive gaussian noise : adding random gaussian distributed signal to our images with a specified mean and variance
2. Masking noise : randomly set the pixels to 'zero' on a selected percentage area to our images leaving the remaining area of images untouched
3. Salt & pepper noise : In our input images, a threshold is selected and pixels are set to 'zero' which are below the threshold and the  remaining are set to 'one/255'

# How to use this code
The code shows on how to use the autoencoders on the custom MNIST dataset to reconstruct them. Also, denoising techniques like gaussian and masking noise are applied which results in a denoised autoencoder. The user can implement deep stacked autoencoders and deep denoised autoencoders on their own image data after working on this demonstration.

# Pre-requisites 
1. Python 2.7 or greater
2. Tensorflow 0.8 or greater
3. Scipy
4. Numpy
5. skimage (optional)
