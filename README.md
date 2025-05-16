# Handwritten Digits GAN with WGAN-GP Architecture

This repository contains an implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP) designed to generate realistic handwritten digits. The model is built using PyTorch and includes sophisticated techniques to stabilize GAN training.

## Overview

This GAN implementation uses:
- **WGAN-GP Architecture**: For more stable training compared to traditional GANs
- **Instance Noise**: Gradually decreasing noise to prevent discriminator overfitting
- **Learning Rate Scheduling**: Step-based LR reduction to fine-tune training
- **Batch Normalization**: In the generator for improved stability
- **Smaller Discriminator Network**: To balance generator/discriminator competition

The model generates convincing 28×28 grayscale handwritten digits that resemble those in the training dataset.

## Dataset

The code requires a dataset of handwritten digits stored in the following structure:

```
D:\HandwrittenDigitsDataset\dataset
├── 0\0\*.png
├── 1\1\*.png
├── ...
└── 9\9\*.png
```

Each PNG should be a 28×28 grayscale image with a transparent background and black ink. You can modify the dataset path in the code if needed.

## Requirements

- Python 3.8+
- PyTorch (with CUDA support recommended)
- torchvision
- Pillow (PIL)
- Matplotlib
- NumPy

Install dependencies:

```bash
pip install torch torchvision pillow matplotlib numpy
```

## Technical Details

### Model Architecture

- **Generator**: 
  - Features batch normalization and dropout for stability
  - Three-layer network with 512→1024→784 neurons
  - tanh activation for normalized output images

- **Discriminator**: 
  - Purposely designed smaller than the generator (256→128→1)
  - Includes dropout layers to prevent overfitting
  - No sigmoid activation (in line with WGAN principles)

### Training Techniques

- **Gradient Penalty**: Enforces 1-Lipschitz constraint on discriminator
- **Instance Noise**: Applied to both real and fake images, decreasing over time
- **Learning Rate Scheduling**: Reduces learning rates by half every 5 epochs
- **Differential Learning Rates**: Generator learns 10× faster than discriminator
- **Gradient Clipping**: Prevents extreme gradients for better stability

## Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/handwritten-digits-wgan.git
   cd handwritten-digits-wgan
   ```

2. **Train the Model**:
Run all the cells in the `DigitsGAN.ipynb` file one by one.

   - Training runs for 50 epochs by default
   - Generated samples are saved every 10 epochs
   - Loss values are printed every 100 batches

3. **Generate Images**:
   After training, load the saved model to generate new digits:
   ```python
   generator = Generator(latent_dim=100)
   generator.load_state_dict(torch.load('generator.pth'))
   z = torch.randn(16, 100)
   fake_imgs = generator(z)
   ```

## Results

Training progress shows discriminator loss fluctuating around zero (expected in WGAN) while the generator loss gradually improves. The discriminator provides a balanced training signal without completely overwhelming the generator.

## Troubleshooting

- **Mode Collapse**: If generated images lack diversity, try:
  - Increasing instance noise
  - Further reducing the discriminator learning rate
  - Adding more dropout to the discriminator

- **Poor Image Quality**: If generated images are blurry or unrealistic:
  - Train for more epochs
  - Increase the generator capacity
  - Adjust the learning rate scheduling

## References

- [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [Instance Noise: A technique for stabilizing GAN training](https://arxiv.org/abs/1406.2661)

## ⚠️ Warning
This code is generated using LLMs (**Grok 3** and **Claude 3.7 Sonnet**) and may have certain problems and/or inaccuracies. Please run this code at your own risk. This code is published for **educational purposes** only. The author is **not** responsible for any damage caused by running this code.
