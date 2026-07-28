# Diffusion Model
A diffusion model following [this](https://huggingface.co/blog/annotated-diffusion) huggingface blog post 
written by a native Rust programmer who is being forced to write Python.

# Diffusion - In a Nutshell
In diffusion, we add noise to an image, then we ask a computer to denoise it for us. More specifically:  

Forward Process: Take a real image from a real dataset, add a little Gaussian noise, repeat a few hundred times 
until it becomes pure noise.  
Reverse Process: Train a neural network to look at a noisy image and guess what noise was added. We can subtract the noise until the static turns back into an image that could reasonably fit in our real dataset.  
This network is a [U-Net](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/).  
The U-Net takes two inputs - the noisy image at timestep `t`, and the noisy image `x_t` at timestep `t`. The network only denoises a bit at a time, it's way easier to denoise out a tiny difference from `x_t` to `x_(t-1)` than just going from `x_T` (pure noise) to `x_0` (input data).


## U-Net
From the linked article, here is a brief sumamry:  
A U-Net is a neural network that segments an image into different parts to identify specific objects.  
It is called a U-Net because the architecture looks like a "U"  

There are three parts:  
**CONTRACTION**: Uses 3x3 convolutional layers to shrink images by preserving important details.  
This "Kernel" is controlled by a network itself, it is initialized as a 3x3 matrix containing random weights.  
During training, the network adjusts these weights to look for specific details like lines, edges, colors.  

This 3x3 window usually starts in the top left, 
and slides right 1 pixel at a time (overlapping the 6 of the 9 pixels on the right of the last window).  
For each window, the window adds all 3 numbers (after multiplying by weights) into a single pixel, 
and puts it on a new smaller image. For a 256x256 image, 1 contraction step would only shrink the image to  
254x254.  
By shifting right 1 pixel only and overlapping pixels, we retain the information of how areas relate 
to each other, which is important to neural networks.  

After the convolution layer is done, a 2x2 Max Pooling layer takes the output (maybe 254x254) image and selects 
the highest values in each 2x2 window, which turns the image into the "bigger picture", general image.  

Every time the image shrinks, whether that be due to 3x3 convolutional layers or the 2x2 max pooling layer,  
the number of channels increases. It is no longer R, G, B, (3 channels), it's R, G, B, ... some other channels 
that are layered on top of that. It goes from 3 to 64, to 128 to 256 if the size halved in the case of the 
max pooling layer.  

**BOTTLENECK**: This isn't really a place where much goes on, it's just the middle 
of contraction and expansion.

**EXPANSION**: This is the opposite of contraction, it upsamples the small image raw, then goes over with 
convolutional layers to add detail back into the image. During this process, the image is compared against 
the corresponding contraction images (before contraction), trying to get close to what it was 

## Convolution
Our `Block` uses a 3x3 convolution. Each output pixel is computed from a 3x3 matrix "neighborhood" of 
the input image. It is local, it must be - with 3x3 neighborhoods stacked next to each other, one at 
the top left has very little to no information about the neighborhood in the bottom right. Conversely, 
the 3x3 neighborhood that overlaps the 3x3 matrix in the top left corner knows 6 out of the 9 pixels of 
that neighborhood. Note that I am using "matrix" and "neighborhood" interchangedly, but I will start using 
"matrix" more.

As continuation, reference [Attention](#attention)

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

## Attention
For background, see [Convolution](#1x1-convolution-layer).  

In this example, we are following Annotated Diffusion and denoising shirts. Locality doesn't do us favors here.  
Imagine a shirt is coming out to be turquoise on one sleeve. It should be turquoise on the other sleeve, but the 
sleeves are like 20 pixels apart. There is no way the locality of raw convolution can 
carry information that far.  

This is what Attention is for, it lets every position talk to every other position through something like 
a search engine:  
Query, Key, Value.

The analogy of a search engine actually is quite accurate.  
**Query** is what the position/pixel is looking for.  
**Key** is what the pixel says about itself.  
**Value** is what each pixel hands over if you believe its Key.

For example, a pixel on the left sleeve edge *queries*, "I am a sleeve edge, where is the rest of the shirt"?  
This right sleeve pixel also *says*: "I am a turquoise sleeve edge"
It then hands over some information, due to the fact that it is a turquoise sleeve edge.
It's like a map/dictionary.

Each pixel produces a query, key and value. For each pixel, compare the key to the query, 
and this outputs a *relevance score*, which is turned into weights. Then a weighted average is taken 
of all the pixels' scores. Pixels with high relevance contribute more than those with low relevance.

Q, K, V aren't all as abstract as "i'm a sleeve edge" and "where's the rest of the shirt?", but 
rather just linear projections of the same input `x`. The network learns 3 different "views" of each 
pixel, but what makes it work is that they're different. Q is likely not V, K is likely not V.  

To compare how relevant 2 vectors are, we use a dot product. Dot products literally compare how similar 
two vectors are, so it works here. If two vectors point in the same direction, the output is positive and big. 

When this is done for each Q and K pair, the output is a matrix with shape **N**x**N**, where row `i`, column `j` 
gives information about how much pixel `i` should care about pixel `j`.

### Softmax
The raw scores are arbitrary numbers, and to convert them into weights we need something that is positive and sums to 1. 

The [Softmax](https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/) function can do this 
for us.  
It in short keeps everything positive and bounded between 0 and 1.  

if you input a vector of numbers, softmax them (requires the sum of the exponentials of the numbers 
in the function itself), 
you will get a vector where summing everything gives you 1.

### sqrt(d)
Softmax cares about the gap between 2 numbers. `[9, 10]` and `[1, 2]` plugged into Softmax behave identically, 
as they have the same gap.  
If the gap plugged into Softmax is too big, say `[90, 100]` (gap of 10), it quickly saturates to 0% or 100%. 
When a model reaches that point, it stops learning because it thinks it has reached perfection, the gradients 
disappear.  
A score is a sum of `d` products, where `d` is the length of each query/key vector, 
statistically half positive and half negative. When they cancel, you 
get a residual that grows roughly like `sqrt(d)`. A good analogy is a drunk person stumbling around, they 
could go in any direction each step. When they take 100 steps from a dropoff point because they got kicked 
off a car they hitchhiked, they don't end up perfectly 100 steps away from their original location, it would be 
extremely rare since they are randomly wandering, they would end up about 10 steps away from their original 
position. Similarly, if you flip a coin 100 times, add 1 for heads, subtract 1 for tails, you get a score that 
is about + / - 10 from 0, not 100. That would be really rare. Overall, randomness over steps will usually 
cancel out.  
The typical gap between 2 scores is `sqrt(d)`. If you divide by `sqrt(d)`, the gap comes back to 
~1, which Softmax really likes.

## Back to Attention
One head of attention allows a pixel to query for other pixels with the same one trait, but if we need a pixel 
to match itself to several different traits, we need multi-head attention. For example, if we only had a single 
head, we could maybe only focus on say, color.

Imagine you were a detective matching a crime to a crime scene, and you were only allowed to investigate one 
thing at a time. If you only look at the broken window glass on the inside, you would miss all the other details 
like blood splatters on the wall, broken display cases, and you would not be able to solve the crime. However, 
if you could look at all the other details, you could match the crime pretty easily.

To feed these multiple heads, the computer needs Queries, Keys, Values. If you've forgotten from a block before, 
Queries are the questions, Keys are the labels, Values are the content. This is obtained through the 
1x1 convolution layer.

## 1x1 Convolution Layer
The 1x1 convolution layer just looks at 1 pixel and mixes all its channels.  
Imagine the image is made out of LEGOs. Each "pixel" is actually a stack of legos, with each stack 
containing many different colors of legos, each one representing a channel. The 1x1 convolution 
layer operates on only one of these stacks at a time, not looking at the neighboring pixels but 
only inside the one pixel. For our case, we need 3 different views from each single pixel, 
Queries, Keys, Values. We could process the image three times, but that would be inefficient. 
So, we just create a giant channel mega-brick with all three views in it at once with 1 operation, 
which can be sliced up using `.chunk(3)`.

Q, K, V are made with `nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)` creating one `conv` that is 
later split into 3 with `.chunk(3, dim=1)`.

# `einsum`
`torch.einsum` is notation for multiplying and summing over 
axes. The rules are:  
- Same letter in 2 inputs -> axes line up
- Letter appears in input but not output -> summed over
- Letters in the output -> kept

e.g.

`torch.einsum("b h d i, b h d j -> b h i j", q, k)`

`d` (dimension of the feature), is missing from the output, 
so it is summed as the dot product. `i` and `j` are indices 
of pixels and both survive, giving us the full pixel x pixel 
score matrix.  

This is a batched matrix multiplication.

# `amax`
`sim = sim - sim.amax(dim=-1, keepdim=True).detach()`

Softmax doesn't change if you subtract a constant from each 
score because the constant cancels in the numerator and 
denominator. If you are not familiar with Softmax, go 
look at [Softmax](#softmax).

Subtracting the row max makes the largest exponent 
`exp(0)`, or `1`, instead of something like `exp(50)`
which would blow up to infinity and overflow. 

The `.detach()` says that this operation should not be backpropagated, 
because it's not part of the model.
