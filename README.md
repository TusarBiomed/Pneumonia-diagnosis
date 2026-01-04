## Pediatric Pneumonia Diagnosis using CNN

## Project Overview
This project implements a Convolutional Neural Network (CNN) to diagnose pediatric pneumonia from chest X-ray images. The model is trained to classify X-ray scans into two categories: **Normal** and **Pneumonia**. The system leverages deep learning techniques to assist in medical diagnosis, achieving high accuracy in identifying pneumonia cases.

## Dataset
The dataset consists of pediatric chest X-ray images divided into two classes:
- **Training Set**: 2,461 images (2 classes)
- **Test Set**: 446 images (2 classes)

The images are preprocessed and resized to 224×224 pixels for compatibility with the CNN architecture.

## CNN Architecture
The model follows a sequential CNN architecture with the following layers:

1. **Convolutional Layer 1**: 32 filters, 3×3 kernel, ReLU activation
2. **Max Pooling Layer 1**: 2×2 pool size, stride 2
3. **Convolutional Layer 2**: 32 filters, 3×3 kernel, ReLU activation
4. **Max Pooling Layer 2**: 2×2 pool size, stride 2
5. **Flattening Layer**
6. **Fully Connected Layer**: 128 neurons, ReLU activation
7. **Output Layer**: 1 neuron, Sigmoid activation (binary classification)

**Total Parameters**: 11,954,337 (all trainable)

## Training Process
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 32

### Data Augmentation (Training Set)
- Rescaling (1./255)
- Shear transformation (0.2)
- Zoom (0.2)
- Horizontal flip

### Preprocessing (Test Set)
- Rescaling only (1./255)

## Results
The model achieves promising results in classifying pediatric pneumonia cases. The training process includes comprehensive evaluation metrics:

### Performance Metrics
- **Confusion Matrix**: Visual representation of true vs. predicted classifications
- **Accuracy**: Overall diagnostic accuracy of the model
- **ROC Curve**: Receiver Operating Characteristic curve demonstrating the model's ability to distinguish between pneumonia and normal cases

### Training History
The training process shows:
- Consistent improvement in training accuracy
- Validation accuracy reaching up to 95.07%
- Stable loss reduction over 30 epochs

## Key Features
1. **Medical Application**: Specifically designed for pediatric pneumonia diagnosis
2. **Deep Learning**: Utilizes CNN architecture optimized for image classification
3. **Comprehensive Evaluation**: Includes confusion matrix, accuracy metrics, and ROC curve analysis
4. **Data Augmentation**: Enhances model generalization through image transformations
5. **Binary Classification**: Clear distinction between pneumonia and normal cases

## Technical Details
- **Framework**: TensorFlow 1.14.0 with Keras backend
- **Image Size**: 224×224 pixels
- **Color Channels**: 3 (RGB)
- **Image Processing**: OpenCV/PIL (via Keras preprocessing)
- **Visualization**: Matplotlib
- **Evaluation**: sklearn metrics integration
- **Total Parameters**: 11.95 million
- **Output**: Binary classification (Pneumonia/Normal)

## Usage
The notebook provides a complete pipeline for:
1. Data preprocessing and augmentation
2. CNN model building and training
3. Model evaluation and performance visualization
4. Generation of diagnostic metrics (confusion matrix, accuracy, ROC curve)

## Applications
This system can be used as:
- An assistive tool for radiologists in diagnosing pediatric pneumonia
- A screening mechanism in healthcare facilities with limited access to specialist care
- A research benchmark for medical image classification using deep learning

## Limitations
- Model performance depends on dataset quality and diversity
- Requires further validation with external datasets
- Should be used as an assistive tool alongside clinical expertise

## Future Improvements
- Integration of additional imaging modalities
- Multi-class classification for different pneumonia types
- Real-time inference capabilities
- Transfer learning with pre-trained models
- Explainable AI features for clinical transparency

## Conclusion
This CNN-based system demonstrates effective pneumonia diagnosis from pediatric chest X-rays, offering a promising approach to assist medical professionals in early detection and treatment. The comprehensive evaluation metrics provide transparent performance assessment, making it suitable for clinical research applications.

## Author: Rezaul Karim Tusar,
- MSc Epidemiology
- LMU München
