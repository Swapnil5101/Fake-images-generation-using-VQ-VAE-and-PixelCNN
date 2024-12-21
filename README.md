# Fake-images-generation-using-VQ-VAE-and-PixelCNN
Vector Quantization plus auregressive CNN model training for creating fake images of given ones.


In this project, the ISIC 2018 dataset is chosen. Task is to generate realistic skin lesion
images by applying advanced generative modeling techniques like VQ-VAEs and then applying
autoregressive modeling techniques like PixelCNN, LSTM, etc.

Link to dataset: https://challenge.isic-archive.com/data/#2018


VQ-VAE Model: Combines the encoder, vector quantization, and decoder into a single
model. It takes an input image, encodes it into a continuous latent space, quantizes the latent
space, and decodes it back to reconstruct the input. It also computes the embedding loss
(quantization loss) and perplexity.

PixelCNN: This generative neural network generates new images pixel by pixel by predicting the likelihood of the next pixel, based on the pixels preceding it. It takes in the current pixels and outputs the probability distribution for the next pixel, i.e., a probability is outputted for each possible pixel value.
