# Generative Adversarial Network (GAN)
***
## What is a Generative Adversarial Network (GAN)?
- A GAN (Generative Adversarial Network) is a type of generative model where two neural networks, the **Generator** and the **Discriminator**, compete against each other .
- The **Generator**'s job is to create artificial data (like images) from random noise, trying to make it as realistic as possible .
- The **Discriminator**'s job is to evaluate the data it receives and determine whether it is real (from an actual dataset) or fake (created by the Generator) .
- Through this competitive process, the Generator becomes progressively better at creating convincing fakes, while the Discriminator gets better at spotting them .

## Core Concepts of GAN
- **Generator (G)**: This network takes a random noise vector (from a latent space, `Z-Dim`) as input and attempts to transform it into a sample that resembles the true data distribution (e.g., a 28x28 image).
- **Discriminator (D)**: This network acts as a binary classifier. It takes a data sample (either real or fake) as input and outputs a probability of the sample being real. It is trained to output `1` for real images and `0` for fake ones.

## Training Process
The training of a GAN follows an adversarial strategy broken down into three main steps:
1.  The Discriminator is trained on a batch of real images from the dataset, learning to classify them as real (output `1`).
2.  The Discriminator is then trained on a batch of fake images produced by the Generator, learning to classify them as fake (output `0`).
3.  The Generator is trained to produce images that fool the Discriminator. In this step, the Discriminator's weights are frozen, and the Generator's weights are updated to maximize the Discriminator's error (i.e., make it output `1` for a fake image).

## Python Implementation Overview
- **Goal**: To build and train a GAN to generate new images based on Google's "Quick, Draw!" dataset (specifically, 28x28 images of camels).
- **Process**:
    - **Model Building**: The architecture consists of a Generator and a Discriminator built with `Conv2D` and `Conv2DTranspose` layers. The document specifies hyperparameters for both networks, including filter sizes, kernel sizes, strides, and learning rates.
    - **Custom GAN Class**: A custom `GAN` class encapsulates the logic for building the generator, discriminator, and the combined adversarial model. It also contains the training methods.
    - **Training**: The model is compiled with an optimizer (e.g., `rmsprop`) and trained on the image dataset. A custom `train` method alternates between training the discriminator and the generator for a set number of epochs.
    - **Visualization**: During training, the loss and accuracy for both the Generator and Discriminator are plotted to monitor performance.
- **Example GAN Training Code Snippet**:
    ```python
    # Define the GAN model with specified hyperparameters
    gan = GAN(
        input_dim = (28,28,1),
        discriminator_learning_rate = 0.0008,
        generator_learning_rate = 0.0004,
        z_dim = 100
    )

    # Compile and train the model
    gan.train(
        images,
        batch_size = 64,
        epochs = 2,
        run_folder = RUN_FOLDER
    )

    # Plot the losses to visualize training progress
    plt.plot([x[0] for x in gan.d_losses], color='black') # Discriminator loss
    plt.plot([x[0] for x in gan.g_losses], color='orange') # Generator loss
    plt.show()
    ```

## Challenges and Advanced Models
- The document notes that training standard GANs can be unstable, leading to issues like:
    - **Oscillating Loss**: The loss values for the Generator and Discriminator can fluctuate wildly instead of converging .
    - **Mode Collapse**: The Generator produces a very limited variety of samples, failing to capture the diversity of the training data .
- **Improved Models**: To address these stability issues, more advanced models were developed:
    - **WGAN (Wasserstein GAN)**: Uses Wasserstein loss and weight clipping to stabilize training .
    - **WGAN-GP (WGAN with Gradient Penalty)**: An improvement on WGAN that uses a gradient penalty instead of weight clipping and avoids batch normalization in the critic (discriminator) .
