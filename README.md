# Face Emotion Detection

**Authors**: Sayali Lokhande, Bhuvana Ravikumar, Anish Kataria  
**Project Report**: [MLD_Final_Report.pdf](./MLD_Final_Report.pdf)

## Overview
This project explores techniques to improve face emotion detection accuracy using **ResNet-18** and **Tiny-ViT** models.  
Face emotion detection has applications in **mental health analysis, human-computer interaction, and surveillance**.  
The objective is to evaluate different **optimization strategies** to improve model performance.

## Dataset
- **FER2013 Dataset** ([Kaggle Link](https://www.kaggle.com/datasets/msambare/fer2013))
- Contains **48x48 grayscale facial images** categorized into seven emotions:
  - Anger  
  - Disgust  
  - Fear  
  - Happiness  
  - Sadness  
  - Surprise  
  - Neutral  
- The dataset consists of **28,709 training images** and **3,589 testing images**.

## Models Used
- **ResNet-18** ([PyTorch Model](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html))
- **Tiny-ViT (ViT-B/16)** ([Hugging Face Model](https://huggingface.co/google/vit-base-patch16-224))

## Optimization Techniques
Five different optimization techniques were applied to improve accuracy:

1. **Learning Rate Scheduling**: StepLR reduces learning rate at fixed intervals.  
2. **Regularization**: Dropout and L2 weight decay were used to prevent overfitting.  
3. **Data Augmentation**: Random flipping, rotation, and color jittering were applied to improve generalization.  
4. **Increased Training Epochs**: Extended training up to 50 epochs.  
5. **Hyperparameter Tuning**: Adjustments to optimizers, batch sizes, and learning rates.  

## Results
ResNet-18 performed better with **data augmentation** and **hyperparameter tuning**, while Tiny-ViT performed best with **regularization techniques**.
