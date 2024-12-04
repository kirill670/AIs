AIs: A Collection of AI Experiments
This repository contains a collection of experimental AI models implemented in PyTorch.  The focus is on exploring different architectures and approaches to various AI tasks.  Currently, the repository includes implementations of:


ByteAI.py: A Generative Adversarial Network (GAN) designed to generate MNIST images represented as byte arrays.  The model utilizes a Liquid VQ-VAE (Vector Quantized Variational Autoencoder) for encoding and decoding, and a standard GAN architecture (generator and discriminator) for image generation. The training process uses binary cross-entropy loss.  Link to ByteAI.py


BytesButLiquidAI.py: A hybrid model combining a Liquid State Machine (LSM), Liquid Neural Network (LNN), and a Spiking Neural Network (SNN) to process MNIST images represented as byte arrays. The images are saved in PNG format. The model is trained using Mean Squared Error (MSE) loss. Link to BytesButLiquidAI.py


EvoAI.py: A neuroevolutionary model employing Covariance Matrix Adaptation Evolution Strategy (CMA-ES) and Particle Swarm Optimization (PSO) to optimize the parameters of a convolutional neural network.  The network architecture includes convolutional layers, fully connected layers, Layer Normalization, ReLU activation, and Dropout. The fitness function is the negative MSE loss. Link to EvoAI.py


LiquidAI.py: A Liquid AI model based on a Mixture of Experts (MoE) and a Transformer layer (Reasoning Layer). The MoE layer selects the best experts (fully connected layers) using a gating network. The Transformer is used for reasoning. The model is trained using MSE loss.  A LiquidAISwarmFoundation class is implemented, representing a swarm of LiquidAI models trained in parallel. Link to LiquidAI.py


Requirements
The code requires PyTorch and several related libraries.  You can install them using pip:

                                                         pip install torch torchvision numpy matplotlib Pillow

                        
Usage
Each Python file contains its own implementation and usage instructions within the code.  Refer to the individual files for details on how to run each model.  Note that some models may require specific datasets (e.g., MNIST) to be downloaded and preprocessed.
Contributing
Contributions are welcome!  Feel free to open issues or submit pull requests.
License
This project is licensed under the MIT License - see the LICENSE file for details.
