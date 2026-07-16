# Diffusion Model
A diffusion model following (this)[https://huggingface.co/blog/annotated-diffusion] huggingface blog post 
written by a native Rust programmer who is being forced to write Python.

# Walkthrough:

We begin at the start of `model.py`.
The exists() and default() functions can be glossed over for the time being, 
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
