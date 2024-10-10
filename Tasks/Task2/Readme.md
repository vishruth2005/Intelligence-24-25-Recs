# Underwater Image Enhancement Project

This project implements various image generation techniques to enhance the quality of underwater images. The project is divided into four subtasks, each exploring a different approach to image enhancement.

## Project Structure

The project consists of four main Jupyter notebooks, each corresponding to a subtask:

- **subtask1.ipynb**: Variational Autoencoders (VAEs)
- **subtask2.ipynb**: Generative Adversarial Networks (GANs) - MNIST Practice
- **subtask3.ipynb**: GANs for Underwater Image Enhancement
- **subtask4.ipynb**: Diffusion Models

## Subtasks and Architectures

### Subtask 1: Variational Autoencoders (VAEs)

- **Architecture**: Encoder-Decoder network with a latent space
- **Encoder**: Convolutional layers followed by dense layers
- **Decoder**: Dense layers followed by transposed convolutional layers
- **Loss**: Reconstruction loss (MSE) + KL divergence

### Subtask 2: Generative Adversarial Networks (GANs) - MNIST Practice

- **Generator**: Fully connected layers with ReLU activation
- **Discriminator**: Fully connected layers with LeakyReLU activation
- **Loss**: Binary Cross Entropy

### Subtask 3: GANs for Underwater Image Enhancement

- **Generator**: U-Net-like architecture with skip connections
- **Discriminator**: PatchGAN discriminator
- **Loss**: Adversarial loss + L1 loss

### Subtask 4: Diffusion Models

- **Architecture**: Simple U-Net-like model for denoising
- **Forward process**: Linear noise schedule
- **Reverse process**: Iterative denoising
- **Loss**: Mean Squared Error (MSE)

## Evaluation Metrics

Each subtask includes the following evaluation metrics:

- Mean Squared Error (MSE)
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)

## Conclusion

Based on the evaluation metrics, here's a comparison of the four models:

| Subtask                               | Mean MSE                                                             | Mean PSNR | Mean SSIM |
| ------------------------------------- | -------------------------------------------------------------------- | --------- | --------- |
| VAE (Subtask 1)                       | 0.05975                                                              | 12.5022   | 0.1606    |
| GAN - MNIST (Subtask 2)               | Not applicable for underwater image enhancement (MNIST dataset used) |
| GAN for Underwater Images (Subtask 3) | 0.0172                                                               | 19.3156   | 0.8254    |
| Diffusion Model (Subtask 4)           | 0.0251                                                               | 17.1502   | 0.6282    |

Based on these results, we can conclude that:

- The GAN for Underwater Images (Subtask 3) performed the best overall, with the lowest MSE, highest PSNR, and highest SSIM scores.
- The Diffusion Model (Subtask 4) came in second, showing promising results across all metrics.
- The VAE (Subtask 1) had the lowest performance among the three applicable models for underwater image enhancement.
- The GAN trained on MNIST (Subtask 2) was not directly applicable to underwater image enhancement but served as a learning exercise for GAN implementation.

The GAN architecture seems to be the most effective for this specific task of underwater image enhancement. However, I feel that the diffusion model has more scope because I was only able to train the model for 10 epochs where as the other two were trained for more. This was due to lack of computational resources. I feel if trained properly diffusion model might outperform the other two in terms of performance.
