# Diffusion Model
A diffusion model following [this](https://huggingface.co/blog/annotated-diffusion) huggingface blog post 
written by a native Rust programmer who is being forced to write Python.

# Walkthrough:

We begin at the start of `model.py`.  
The `exists()` and `default()` functions can be glossed over for the time being, 
they do exactly what they look like they do.

## Residual
The `Residual` class can be initialized with some conversion function `Residual(convert_fn)`.  
The `Residual` will then use this function to change the input `x`, which is initially images
or a batch of images, but later on it is converted so much it may just be edges or some features
so it is just some `x`.  
`fn(x)` is a "change" on `x` with a change, it is only the change. Adding this "residual" onto
`x` again gives you the next layer. This is why `fn(x)` must also return the same shape as `x`,
the change must map cleanly on top of the original input.

Overall: `x` is data in, `fn(x)` is the change applied, `fn(x) + x` is the data out

## Sinusoidal Position Embeddings
[This](https://www.youtube.com/watch?v=dWkm4nFikgM) video offers a good explanation.  
Generally, we will use this to encode the timesteps. Remember, we have one model for all our timesteps, 
not 1 model for each particular transition.  
The same weights have to work for both ends of the noising scale, from barely noisy to almost pure noise  
In this case, it is obvious the two ends need completely different behavior, and this is why we tell the 
model about the timestep, `t`, by passing it in as an input.

Unfortunately, it turns out just passing in `t` as something like a raw integer won't work very well.  
Neural networks like to work with inputs centered around 0, and just one `t` scalar may easily get 
lost in the network's processing of countless more image features.

And so, `t` is expanded from just a number scalar to a large vector, where the numbers may be used 
to encode different values of different magnitudes.

In the video, the guy uses binary numbers as an analogy, but the gist of it is each place encodes 
a change in a different scale, the 2's place encodes very small changes, the 64's place, for instance,
would convey very large changes.  
Similarly, it can be thought of as a vector of inputs, like `[0, 0, 1, 0, 1]` (9)  
Each number encodes a different magnitude 

A clock is also typically used, the hands all spin at different speeds, and combining all of them 
together gives you the exact time of day. 

Of course, we will be using sine and cosine. Their property of being bounded between -1 and 1 
makes them naturally scaled well.

## ResNet
ResNet, or Residual Network is a deep learning architecture that solves the "degradation problem", 
the phenomenon that causes neural networks to lose accuracy when too many layers are trained on. 
When networks get too deep, the gradients stack onto each other and the gradient might explode or 
shrink to nothing.

Normalization helps by resetting the scale, and this is done by the standardization operation:
normalized = (value - mean) / stddev
Subtracting the mean centers it on 0, dividing by the standard deviation forces it to have a spread of 1
Whatever is inputted now has a mean of 0 and a variance of 1, which forces numbers to stay close to 0

## GroupNorm VS BatchNorm
Normalization as explained before is a technique that scales and centers data as it passes through 
a neural network. It stabilizes gradients and helps with training.

Batch Normalization normalizes features across the whole batch, creating one mean/stddev, 
and is the default for most Convolutional Neural Networks (neural networks designed to
process grid-like data like images and video).  
However, in our case, it is not applicable because:
1. All the images are coupled together
- This makes it so generating causes the output to be just one normalized sample
2. It works poorly with small batches
- Normalizing a few images with high noise between gets you mostly junk
3. Training mismatches with Eval
- During training, the neural network gets the whole batch to work with, 
  but it only gets a running average that was computed during training when 
  it comes to evaluation

GroupNorm sidesteps these issues by averaging in a single image, splitting channels across a group of channels into some groups, each group getting their own mean/stddev and computed from the values of the image.  

## Weight Standardization
Standard normalization normalizes the input data while weight standardization normalizes the weights  
Before each forward pass, the convolution kernel is restandardized to 0 mean 1 variance.

The Convolution Kernel is a small matrix of weights that maps an image onto some output

Anyway, weight standardization and groupnorm match or beat batchnorm without batchnorm's disadvantages, even if groupnorm usually underperforms batchnorm

## Feature wise Linear Modulation (FiLM)
We have positional embeddings for time, but it should influence image processing by controlling the scale and shift of each channel by this:
`x = x * (scale + 1) + shift`
scale and shift are computed from the time embedding, with 1 pair per channel, allowing the timestep to control the "weight" and "bias" of each feature, which is FiLM.

we multiply by `scale + 1` instead of `scale` because `scale` can be 0.

We do FiLM right after norm because normalization just erased all the mean and scale information, because thats its job.  
FiLM re-adds that scale and shift the timestep wants, and determines which features fire
