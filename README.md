
# Diffusion Model for Image Denoising and Classification

This project implements a diffusion model for image denoising and classification tasks using PyTorch. The diffusion model is trained on the CIFAR-10 dataset to denoise noisy images and classify them into different categories.

## How it Works

1. **Diffusion Model Architecture**:
   - Utilizes a UNet architecture with sinusoidal position embeddings and label conditioning.
   - Sinusoidal position embeddings encode temporal information, aiding in image denoising.
   - Label conditioning enables the model to perform image classification tasks.
    
2. **Image Nosing and Denoising**:
   - Trains the diffusion model to denoise images corrupted by noise.
   - During training, the model predicts noise in input images and generates denoised outputs.
   - Minimizes Mean Squared Error (MSE) loss between predicted noise and ground truth noise.
     
    *Noising and Denoising*
   
    ![Forward Diffusion Evolution](https://github.com/gk-gokul/image_diffusion/assets/108075631/e1c7b9aa-cafa-48a2-bbe1-f700ae3e603c)
    ![Reverse diffusion 1](https://github.com/gk-gokul/image_diffusion/assets/108075631/50bb4e55-a3cc-4b56-902e-a78460e5231c)
    ![Reverse Diffusion 2](https://github.com/gk-gokul/image_diffusion/assets/108075631/42777359-f1df-40db-a135-f4fdcd3340fc)
    ![Reverse Diffusion 3](https://github.com/gk-gokul/image_diffusion/assets/108075631/acb203af-73b5-47fe-870c-3fb1d918e503)
    ![Reverse Diffusion 4](https://github.com/gk-gokul/image_diffusion/assets/108075631/09615c4a-d576-4e67-af64-cbe3470bcbd1)
    ![Reverse Diffusion 5](https://github.com/gk-gokul/image_diffusion/assets/108075631/86a52285-016e-4867-b519-191ebf9e1cd6)
    ![Reverse Diffusion 6](https://github.com/gk-gokul/image_diffusion/assets/108075631/0234af07-0ce7-4d05-af1f-43c7b3e3ac72)

   
4. **Image Classification**:
   - Trains the diffusion model to classify images into different categories.
   - Utilizes label conditioning to provide class information during training.
   - Learns to generate images corresponding to different classes.
   ![Diffusion Evaluation](https://github.com/gk-gokul/image_diffusion/assets/108075631/f8d5b7c6-952f-4e13-85a6-fc1613a7f7a8)

5. **Training Process**:
   - Includes code for training the diffusion model using the CIFAR-10 dataset.
   - Generates noisy images by adding random noise to input images.
   - Trains the model to predict noise in noisy images and generate denoised outputs.
   - For classification, trains the model to generate images corresponding to different classes.

   ![Training](https://github.com/gk-gokul/image_diffusion/assets/108075631/70ae1409-07b1-48f4-88ce-36ec27c9f4ff)

6. **Evaluation**:
   - Evaluates the trained model on the test dataset to assess denoising and classification performance.
   - Uses metrics like Mean Squared Error (MSE) loss and classification accuracy for evaluation.
  ![Training](https://github.com/gk-gokul/image_diffusion/assets/108075631/70ae1409-07b1-48f4-88ce-36ec27c9f4ff)
     

7. **Visualization**:
   - Provides code for visualizing denoised images and classification results.
   - Uses Matplotlib to plot histograms of noise distributions and display denoised images.
  
   ![Epoch 0 graph](https://github.com/gk-gokul/image_diffusion/assets/108075631/b4df352d-e0d9-43cc-8192-0cef28c54237)
   ![Epoch 0 noises](https://github.com/gk-gokul/image_diffusion/assets/108075631/e29077c0-bc3a-4d00-b71d-87c1c38121fd)
   ![epoch 400 graph](https://github.com/gk-gokul/image_diffusion/assets/108075631/dfcef7d5-2b85-420e-a29c-6a4231654814)
   ![epoch 400 noise](https://github.com/gk-gokul/image_diffusion/assets/108075631/f5570877-3c0d-402c-949f-f07b7c8430e1)
   ![epoch 1200 graph](https://github.com/gk-gokul/image_diffusion/assets/108075631/73de4f88-901a-4354-aa93-4e68388bb664)
   ![epoch 1200 noise](https://github.com/gk-gokul/image_diffusion/assets/108075631/77e7e40a-d817-412a-9115-864bbc8b6e86)
   ![epoch 1600 graph](https://github.com/gk-gokul/image_diffusion/assets/108075631/93035cc9-0eab-45fe-90ea-742460733046)
   ![epoch 1600 noise](https://github.com/gk-gokul/image_diffusion/assets/108075631/6f5903c6-cfd6-4184-91f9-24ce0fe78b55)

## Installation

1. Install Python 3.x.
2. Install required dependencies using `pip`:
   ```
   pip install torch torchvision matplotlib numpy pillow
   ```

## Usage

1. Open the provided Jupyter Notebook (`DiffusionModel.ipynb`).
2. Run the notebook cells sequentially to train the diffusion model and evaluate its performance.
3. Customize the model parameters and training settings as needed for your specific use case.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by research in deep learning and diffusion models.
- Thanks to the developers of PyTorch, Matplotlib, NumPy, and PIL for their invaluable libraries.

---
