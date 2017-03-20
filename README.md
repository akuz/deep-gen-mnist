# generative-mnist
Generative model for MNIST handwritten digits data, built using TensorFlow's standard components from the Neural Network toolbox. 

This model is probabilistic rather than deterministic in nature. This is expressed through the inversion of the convolutions, making them go from the latent variables to the image, rather than from the image to the latent variables.

The process of inference and learning requires two optimizers: an inner optimizer for inferring the state of the latent variables given a batch of images, and an outer optimizer for learning the parameters of the model.