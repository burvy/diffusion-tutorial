# Diffusion Model
A diffusion model following (this)[https://huggingface.co/blog/annotated-diffusion] huggingface blog post 
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
(This)[https://www.youtube.com/watch?v=dWkm4nFikgM] video offers a good explanation.  
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
