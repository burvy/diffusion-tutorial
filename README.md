# Diffusion Model
A diffusion model following [this](https://huggingface.co/blog/annotated-diffusion) huggingface blog post 
written by a native Rust programmer who is being forced to write Python.

# Diffusion - In a Nutshell
In diffusion, we add noise to an image, then we ask a computer to denoise it for us. More specifically:  

Forward Process: Take a real image from a real dataset, add a little Gaussian noise, repeat a few hundred times 
until it becomes pure noise.  
Reverse Process: Train a neural network to look at a noisy image and guess what noise was added. We can subtract the noise until the static turns back into an image that could reasonably fit in our real dataset.  
This network is a [U-Net](https://www.geeksforgeeks.org/machine-learning/u-net-architecture-explained/).  
The U-Net takes two inputs - the timestep `t`, and the noisy image `x_t` at timestep `t`. 
The network only denoises a bit at a time, it's way easier to denoise out a tiny difference 
from `x_t` to `x_(t-1)` than just going from `x_T` (pure noise) to `x_0` (input data).


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

According to the generic U-Net articles,
After the convolution layer is done, a 2x2 Max Pooling layer takes the output 
(maybe 254x254) image and selects the highest values in each 2x2 window, which 
turns the image into the "bigger picture", general image.  

According to the code:
We don't just blindly downscale, rather we compress 4 pixels into 1 channel, and 
the 1x1 convolution layer learns what to keep. 

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

GroupNorm sidesteps these issues by averaging in a single image, 
splitting channels across a group of channels into some groups, 
each group getting their own mean/stddev and computed from the values 
of the image.  

## Weight Standardization
Standard normalization normalizes the input data while weight 
standardization normalizes the weights  
Before each forward pass, the convolution 
kernel is restandardized to 0 mean 1 variance.

The Convolution Kernel is a small matrix of weights that maps an image onto some output

Anyway, weight standardization and groupnorm match or beat 
batchnorm without batchnorm's disadvantages, 
even if groupnorm usually underperforms batchnorm

## Feature wise Linear Modulation (FiLM)
We have positional embeddings for time, but it should influence image processing by controlling the scale and shift of each channel by this:
`x = x * (scale + 1) + shift`
scale and shift are computed from the time embedding, with 1 pair per channel, allowing the timestep to control the "weight" and "bias" of each feature, which is FiLM.

we multiply by `scale + 1` instead of `scale` because `scale` is 0 in the 
default state before we train it. When we multiply `x` by 0, it just 
kills training since all the information is gone before even getting to 
learn anything.  
`scale + 1` is about `1`

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

## `einsum`
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

## `amax`
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

## LinearAttention
`Attention` has a problem. It builds `sim` with shape `(n, n)` where `n = height * width`, because 
every pixel is scored against every other pixel. This is an issue, because imagine a 16x16 image.  
Image Size: 16x16  
Pixel Count (n): 256  
Entries in `sim`: 65536 (sim is the matrix of scores against each other)

However, we can solve this blow-up by using the associative property of multiplication:  
`(3 * 4) * 5 = 3 * (4 * 5) = 60`  
Both equal 60, but the intermediate step is different, `3 * 4 = 12`, different from `4 * 5 = 20`.  
In this case, 20 might be slightly easier to work with, since we can just do `3 * 2 * 10` which is 
`6 * 10` which is 60. The same thing works with matrix multiplication, and the intermediates here 
differ a lot from just 12 and 20.  

e.g.
Regular attention is like making guests at a party get to know every other guest, and 
for 200 guests, thats 40000 interactions. Linear attention would be like a whiteboard in the 
corner, having everyone write their info on there, and having every guest read from that. For
200 guests, that is 200 writes, 200 reads. 

This whiteboard would be K^T @ V.  
Regular Attention computes (Q @ K^T) @ V, which causes (n, d) @ (d, n) -> (n, n), which 
grows really fast and was the whole issue.  
Linear Attention does Q @ (K^T @ V) which causes (d, n) @ (n, d) -> (32, 32), 
the dimensions.  
`d = dim_head = 32`  
When you matrix multiply, for example (d, n) @ (n, d), the inner `n` disappears, 
it is summed away. 

Importantly, to define `n` and `d`, `n` can be thought of a count of contributors, 
and `d` how many numbers describe each contributor.  
When averaging 10 numbers, 10 is the `n`, and 1 is the `d`. The 10 numbers are summed and 
divided by 10, and stored in one output. When averaging 100 numbers, 100 is the `n`, and 
1 is still the `d`. No matter how many contributors contribute to a value, the output is 
just one value. As a result, `d` never changes in size.

However, this isn't exactly like Attention. This whiteboard cannot contain each 
individual pixel's knowledge of every other pixel, but rather it contains every 
pixel's knowledge of the rest of the image filtered through those `d`x`d` slots.  

As a result, we still must use Attention when it is affordable to do so, at the 
deepest 7x7 layer in the U-Net.

Worth looking at this too:
```python
q = q.softmax(dim=-2) # over features: how pixels split attention during READ
k = k.softmax(dim=-1) # over pixels: how pixels write to slots together during WRITE
```

## Normalization
We are going back to this for reinforcement:  
Normalization takes a pile of numbers, subtracts the mean off all of them, 
and divides by the standard deviation, causing the pile to be centered around 0.  

For example:  
[10, 12, 14] - 12 => [-2, 0, 2] => divide by std => [-1.22, 0, 1.22]  

This produces the same shape of the data but with standard scale.

The tension is on what axis we should normalize over, and there are two options: 
## BatchNorm
BatchNorm creates one pile per channel, pooling across every image in a batch. 
which causes all images to be coupled together and messy, as explained before.
## GroupNorm -> LayerNorm
We normalize over the other axis, chopping channels into groups and piling in 
each single image. No image knows about any other image like BatchNorm.

## Why LinearAttention needs normalization
Attention's softmax guarantees that weights are positive and sum to 1. The output 
is a weighted average of V rows; it cannot possibly exceed the largest value in V.  

LinearAttention softmaxes the two inputs independently, and multiplying them together 
doesn't give you a normalized product. However, the output is still probably well 
behaved, but still drifts over time with Residual being added. 
Normalization is what forces the output back to being well behaved like 
Attention would be.

------
NOTE:  
You may notice we usually start with an identity, and add some residual onto 
it frequently. For example, in FiLM, 1 + scale when scale begins at about 0, so we multiply 
by 1. Each layer starts out doing nothing (no-op), and it learns to be useful later on.
------

## PreNorm
PreNorm takes a copy of the data, then makes it processable for Attention. Residual is 
added to the data that didn't get  PreNormed, not ONTO the PreNormed copy, which means 
it shouldn't be normalized with each iteration. 

PreNorm simply does this:
```python
return self.fn(self.norm(x))
```
Normalizes, then runs the function on it. It simply guarantees that the input is readable 
before passing into the Attention function. 

## Views and Copies
In Attention and LinearAttention, you will see:  
`q = q * self.scale`  
We do this instead of `q *= self.scale` because `*=` does in-place mutation, which is bad 
because q originally gave us a view into data that PyTorch is still using. We have to create 
a new copy of the tensor with self.scale applied to it.


# U-Net
We are finally done with everything required to construct the U Net.  
There are a few specific things related to diffusion:  
## Time Embeddings
`SinusoidalPositionEmbeddings` is constructed once and injected in all layers through 
a small MLP that encodes it on one vector. Each `ResnetBlock` `FiLM`s on it, allowing 
this single U-Net to handle all timesteps of diffusion.  
### `nn.Linear`
Starting from the basics, `nn.Linear` is a layer itself. Mathematically:  
`y=Wx+b` where `W` is the weight and `b` is the bias.  
Each output number is a weighted sum of each input number.  

Contrast this with `Conv2d`. A convolution layer sees a 3x3 neighborhood and slides 
across an image to process images.  
A linear layer just processes vectors with no spatial structure.
### MLP
MLP, or Multi-Layer Perception stacks several `Linear` layers with nonlinearity in between.  
Running something through multiple linear layers does the same thing a single linear layer 
could, like mixing buckets of paint, you could have always mixed paint together in one step.  

However, an activation layer is placed in between the `Linear` layers which allows the 
second layer to do something the first couldn't. Here is an example in our code:  
`nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))`

We will eventually use this in our U-Net like so:  
```python
nn.Sequential(
    SinusoidalPositionEmbeddings(dim), # fixed formula (no learning)
    nn.Linear(dim, time_dim), # first linear
    nn.GELU(), # activation function to allow learning and prevent collapse
    nn.Linear(time_dim, time_dim), # second linear
)
```
We need this because the position embeddings are a hardcoded formula, but meaningless to 
the network. This layer allows the network to learn what the position, or in our case, 
the timestep is.

## Attention
Attention happens for every level, but the difference is `LinearAttention` is used for 
higher resolutions (as it is faster), and normal `Attention` is used for the smallest 
layers in the middle where precision is essential.  

## Concatenation in skips
Skips are cat, not +.
Using + destroys the information that was there before. For example, 8 could be 
3 + 5, 4 + 4, 6 + 2, etc.. the information on what we used is gone.  
Concatenation stacks without mixing. Something like representing 8 using 6 + 2.  
As a result, we might have to store some more data, which is why ResnetBlock takes 
`dim * 2` as the input. When `dim * 2` appears in the U-Net, this is what is happening.

## Symmetry
When downsampling, the skip should match the upsampling dimensions exactly.  
When we upsample and the dimensions aren't the same as what the skip saved, 
the program will crash, and good that it does because we need the training 
to be exact.
  
## Output
Our network predicts noise, not an image. Output is the same shape as input, which is 
the noise.

# The actual U-Net
We can assemble everything into the actual U-Net now.
## 1x1 Convs
```python
self.init_conv: nn.Conv2d = nn.Conv2d(channels, init_dim, 1, padding=0)
```
We are just stacking channels on a single pixel without blending things together 
like a 3x3 convolution would. We would like to keep all the details before we run 
a skip connection right after it.
## `dims`, `in_out`
We start the tensor out with a bunch of channel widths:
```python
dim: int, # 64
dim_mults: tuple[int, ...] = (1, 2, 4),
```
We multiply those into 64, 128, 256.  
These are the channel widths.  
So, the tensor visits: 64 (init_conv), 64, 128, 256  
So, we do:  
64 -> 64 -> 128 -> 256  
We store this as [(64, 64), (64, 128), (128, 256)]  

These are like checkpoints, like if you are planning a drive home, you must stop at certain 
places, which means you track when you transition from one place to another, not the places 
you stop at. Just storing the checkpoints doesn't really give the direction.

## Linear Beta Schedule
