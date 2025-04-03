# Image Classification using CNN (CIFAR-10)

This project demonstrates image classification using a Convolutional Neural Network (CNN) built with TensorFlow and trained on the CIFAR-10 dataset.

## Overview

- **Framework**: TensorFlow / Keras  
- **Dataset**: CIFAR-10 (60,000 images, 10 classes)  
- **Model**: CNN with 3 convolutional layers and data augmentation  
- **Performance**:  
  - Test Accuracy: ~54.5% after 5 epochs  
  - Test Loss: ~1.33

## Sample Output

### Sample CIFAR-10 Images  
*Save and upload the sample image plot as `samples.png` to see it here.*

```python
# Add this to your script to save the image:
plt.savefig("samples.png")
```

### Model Performance  
*Save and upload the training accuracy/loss plot as `accuracy.png` to display it here.*

```python
# Add this to your script to save the training plot:
plt.savefig("accuracy.png")
```

## Features

- Data loading and normalization  
- Real-time image augmentation using `ImageDataGenerator`  
- CNN with dropout for regularization  
- Accuracy and loss tracking during training  
- Trained model saved as `image_classifier_cifar10.h5`

## Setup

```bash
# Clone the repository
git clone https://github.com/kushitec15691/image-classification-cnn.git
cd image-classification-cnn

# (Optional) Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the training script
python main.py
```

## License

This project is licensed under the [MIT License](LICENSE).
