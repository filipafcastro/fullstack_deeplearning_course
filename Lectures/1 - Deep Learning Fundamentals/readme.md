- [1️⃣ Deep Learning Fundamentals](#1---deep-learning-fundamentals)
  * [Neural Networks](#neural-networks)
    + [Biological logic/inspiration of a neuron:](#biological-logic-inspiration-of-a-neuron-)
    + [From biology to its mathematical representation -  the perceptron:](#from-biology-to-its-mathematical-representation----the-perceptron-)
    + [Activation functions:](#activation-functions-)
    + [What's indeed a neural network?](#what-s-indeed-a-neural-network-)
  * [Universality](#universality)
  * [Learning Problems](#learning-problems)
  * [Loss Functions](#loss-functions)
  * [Gradient Descent](#gradient-descent)
  * [Architectures](#architectures)
  * [CUDA](#cuda)

# 1️⃣ Deep Learning Fundamentals
📼 [Video](https://www.youtube.com/watch?v=fGxWfEuUu0w&feature=emb_title&ab_channel=FullStackDeepLearning) | 📖 [Slides](https://github.com/filipafcastro/fullstack_deeplearning_course/blob/main/Lectures/1%20-%20Deep%20Learning%20Fundamentals/1.%20Deep%20Learning%20Fundamentals.pdf) | 📋 [Notebook](https://github.com/filipafcastro/fullstack_deeplearning_course/blob/main/Lectures/1%20-%20Deep%20Learning%20Fundamentals/neural_network_coding.ipynb)

## Neural Networks

### Biological logic/inspiration of a neuron: 
+ dendrites as receptors of info;
+ if enough stimulation is received by the dendrites, then the entire neuron fires;
+ an electrical impulse is generated that propagates through the axon. This will pass to other neurons through the dendrites. 
+ then there's a network of these neurons getting stimulated. If there's enough stimulation, they fire and there's an electrical potential going through them.

### From biology to its mathematical representation -  the perceptron: 
+ inputs are the stimulus that arrive to the dendrite; 
+ the dendrite gets stimulated by x (weights);
+ you sum  these stimulus to get the stimulus of the entire neuron; 
+ activation of the neuron is either on or off. If it's enought stimulated —> ON; if it doesn't not pass the threshold —> OFF.

### Activation functions:
+ sigmoid
+ hyperbolic tangent
+ **rectified linear unit (RELU): the most used one right now.**

### What's indeed a neural network? 
We've just talked about a neuron/perceptron. It's **neurons arranged in layers: input layer, hidden layers, output layers**. Each of these neurons has its own weights and bias and these will determine how the neural network will work.

## Universality
One can prove that any two layer network (one hidden layer), if given enough units in the hidden layer, can approximate any function. The intuition can be obtained [in this chapetr of this book](http://neuralnetworksanddeeplearning.com/chap4.html). **NNs are universal/general, because theoretically you can represent any function using a neural network**.

## Learning Problems
**Unsupervised Learning:** learn the structure of the data; Eg. predict the next character on a sentence (charRNN); predict similarity/relationships between words (eg. man and woman have the same kind of relationship as queen and king); predict the next pixel/autocomplete images; compress and decompress images (VAEs); GANs: generate fake images which are indistinguishable from the real ones: [https://thispersondoesnotexist.com/](https://thispersondoesnotexist.com/)

**Supervised Learning:** learn to make predictions on the data; 

**Reinforcement Learning:** learn how to take actions according to the data/input/environment. Reality/feedback/reaction is also an input. 

Supervised is the comercially viable at the moment. Reinforcement will be the next.

## Loss Functions
It's about finding the parameters (weights and biases) that minimize the loss function (eg. MSE, cross-entropy).

## Gradient Descent
+ **1.** We have some random weights and bias, random parameters.
**2.** We evaluate these parameters on the data we observe, this is, see their loss. 
**3.** We can compute the loss function in respect to these weights, which means thinking like: if I change this parameters by X, how much does my loss changes? which is the same as computing the loss function derivative with respect to the weights. 
**4.** And then, in order to improve the fit of the NN to the data, we will update the weights by subtracting the derivative of the gradient to its current value, multiplied by some learning rate. 
By doing this, we're changing the weight in the direction of loss minimization and we expect that we're heading towards a local minima. 
We also want to walk towards the steepest direction in terms of minimizing loss.

+ Gradient descent is a first-order approximation (you just compute 1st order derivative). There are methods which are more expensive (2nd order) but more powerful. For instance, Adam is a 2nd order approximation.

+ Stochastic Gradient Descent (SGD): Instead of updating each weight at a time, we do it in batches. Stochastic gradient descent aka batch gradient descent. Less compute per optimization step (you just use one batch, not the entire data) for each optimization step. But of course it's more noisy than using all the data.

+ How to compute these derivatives efficiently? Chain rule, because the neural net is made of computations which always have a gradient, this is, functions that always have a derivative (eg. ax+b). 
This is called the **backpropagation step**.

+ we don't have to do it by hand, we have **automatic differentiation software** like PyTorch, Theano or Tensorflow. We just need to program the function/forward pass and the software automatically computes the derivatives in the backward pass.

## Architectures

NNs can represent anything but we should encode prior knowledge of the world to encode our data, for instance:

+ For computer vision purposes, there are CNNs. 
+ For sequence processing (eg. NLP, there are RNNs). 

So the logic behind will always be neural networks, but we adapt them, we adapt the input, we adapt the architecture to go towards the logic/biology (eg. CNN/eye). 
Other adaptations: depth vs width, skip connections, batch/weight/layer normalization.

## CUDA
Deep learning's kick-off on 2013 was not only caused by bigger datasets, but also by good libraries for matrix computations on GPUs (eg. Nvidia). 
Because all the computations in deep learning are just matrix multiplications which are easy to paralelize over the computational cores of a GPU.
