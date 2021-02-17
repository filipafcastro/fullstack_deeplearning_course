# 2️⃣🅰️ Convolutional Networks (CNNs)
📼 [Video](https://www.youtube.com/watch?v=hO3kOdShwsI&ab_channel=FullStackDeepLearning) | 📖 [Slides](https://github.com/filipafcastro/fullstack_deeplearning_course/blob/main/Lectures/2%20-%20Convolutional%20Networks/2A%20-%20Convolutional%20Networks.pdf)

## Convolutional Filters
+ Instead of taking the entire image flattened into a vector and then multiplied by a matrix, you extract a single patch of the image (eg. 5x5), which corresponds to a smaller vector, multiply it by a smaller vector as well, and obtain a one dimensional output/value. And you can slide this patch/window, always using the same weights. The outcome is a group of single values (one per each 5x5 patch), which together make a new matrix.

+ Convolutional operations are the basis of image filters, such as blur, when filters are carefully chosen. [See here](https://setosa.io/ev/image-kernels/).

+ But the idea in deep learning is to **learn these filters/weights**, instead of using pre-determined ones.

+ Because images have 3 channels, our filters will need to be not 5x5, but instead 5x5x3. So, instead of having 25 weights, we'll have 75.

+ And instead of having just one learnable convolutional filter, you can have several ones, which learn independently from each other. If you have one, the 3rd dimension/depth of your output will be one. But if you try to apply/learn several convolutional filters for the same image, the number of outputs, this is, the depth of your output will correspond to the number of filters.

+ Then, we can apply another convolutional filter to this output, this is, stack convolutional filters to build a convolutional neural network.

## Strides and Padding

These define how the convolutional filter slides over the image.

+ **Stride**: convolutions which subsample the image by jumping across some locations. If stride=1, there are no jumps and no subsampling.

+ **Padding**: add default values to the border of the images (extra rows/cols), so that you can still apply the convolution/operations near the borders. It solves the problem of filters running out of the image. Usually default=0.

## Filter Math

Every time you apply a filter you need to know what will be the size of the output, as this will be the input for the next layer and you need to design the next filter of the network.

+ **input**: W x H x D volume

+ **parameters**: 

    + K filters, each with size (F,F), normally set to powers of 2 (eg. 32, 64, etc)
    + stride (S,S), commonly (5,5), (3,3), (2,2), (1,1)
    + padding P

+ **output**: W'x H' x K volume

    + W' = (W-F+2P) / S + 1

    + H' = (H-F+2P) / S + 1

Check visualizations [here](https://github.com/vdumoulin/conv_arithmetic)

## Receptive Field
+ For a particular pixel/position in the output, which pixels/positions in the original input contributed to that outputs' value? Eg. if we have a 3x3 filter with stride 1, we say that the receptive field is 3x3. Each pixel in the output sees a 3x3 patch of pixels in the input.

+ If we stack convolutional operations, we'll increase the receptive field. Eg. input 5x5, apply 2 convolutions of 3x3. The output has a receptive field of 5x5.

+ Stacking two 3x3 convolutions has the same RF as one 5x5, but it has fewer parameters and normally works better.

## Dilated Convolutions
+ Another alternative to increase the receptive field is to use **dilated convolutions**. They see a greater portion of the image by skipping pixels in their convolutions. A 3x3 dilated convolution has 9 weights and a receptive field of 5x5.

+ We can also stack dilated convolutions.

## Other ConvNet Operations

+ To decrease the size of the input, one can use **poling** by subsampling the input through average or max of each region. More info [here](https://cs231n.github.io/convolutional-networks/#pool)

+ To decrease the nr of channels/depth of a tensor, we can apply **1x1 convolutions**. It corresponds to applying a MLP to every pixels in the convolutional output. The receptive field of each convolution is a single pixel but you're seeing the entire depth.. This operation is very popular on ConvNet architectures (eg. Inception).

## ConvNet Architectures

**LeNet(-like)**: [(Conv filter + Non-linearity) x many times --> pooling] x many times --> (FC + non-linearity) x many times --> Softmax. [Check slide 64](https://github.com/filipafcastro/fullstack_deeplearning_course/blob/main/Lectures/2%20-%20Convolutional%20Networks/2%20-%20Convolutional%20Networks.pdf)

# 2️⃣🅱️ Computer Vision Applications
📼 [Video](https://www.youtube.com/watch?v=rHGUVo6GjVA&ab_channel=FullStackDeepLearning) | 📖 [Slides](https://github.com/filipafcastro/fullstack_deeplearning_course/blob/main/Lectures/2%20-%20Convolutional%20Networks/2A%20-%20Convolutional%20Networks.pdf)

## 