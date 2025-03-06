Face Emotion Detection
📌 Authors: Sayali Lokhande, Bhuvana Ravikumar, Anish Kataria
📄 Report: MLD_Final_Report.pdf

📌 Overview
This project explores techniques to improve face emotion detection accuracy using ResNet-18 and Tiny-ViT models.
Face emotion detection is crucial in mental health analysis, human-computer interaction, and surveillance.
Our goal is to evaluate different optimization strategies and improve model performance.

📌 Dataset
FER2013 Dataset (Kaggle Link)
Contains 48x48 grayscale facial images categorized into 7 emotions:
Anger, Disgust, Fear, Happiness, Sadness, Surprise, Neutral.
28,709 training images & 3,589 testing images.
📌 Models Used
🔹 ResNet-18 (PyTorch Model)
🔹 Tiny-ViT (ViT-B/16) (Hugging Face Model)

📌 Optimization Techniques
We applied five different optimization techniques for improving accuracy:
1️⃣ Learning Rate Scheduling: StepLR reduces learning rate at fixed intervals.
2️⃣ Regularization: Dropout & L2 weight decay to prevent overfitting.
3️⃣ Data Augmentation: Random flipping, rotation, color jittering for improved generalization.
4️⃣ Increased Training Epochs: Extended training up to 50 epochs.
5️⃣ Hyperparameter Tuning: Adjusting optimizers, batch sizes, and learning rates.

📌 Results
📌 ResNet-18 performed better with data augmentation and hyperparameter tuning, while Tiny-ViT performed best with regularization techniques.
