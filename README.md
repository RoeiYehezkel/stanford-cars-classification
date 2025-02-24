## Introduction
This assignment focuses on the **Stanford Cars dataset**, developing a **classification model** to predict car types using **CNN architectures**. Several strategies, including custom CNN models, pretrained models, **K-fold cross-validation**, and **augmentation techniques**, were explored. Additionally, a new car class was introduced to evaluate the model's extensibility.

---

## Exploratory Data Analysis (EDA)
### Dataset Overview
The dataset contains labeled images of various car models, divided into training and testing sets:
- **Training set size:** 8144 images
- **Testing set size:** 8041 images

Each image comes with **bounding box information** to focus on the car itself.

### Training & Testing Class Distribution
- **Most populated class:** Class 118 (68 images)
- **Least populated class:** Class 135 (24 images)

### Data Content
Each image includes:
1. **Image Information**
   - **Resolution:** Varies across images; resized to **224 × 224** for training.
   - **Channels:** RGB (3 channels)
   - **Bounding Boxes:** (x1, y1, x2, y2) provided for cropping.
2. **Labels**
   - **196 car model classes** (e.g., *Audi 100 Sedan 1994, Infiniti QX56 SUV 2011*).
   - Imbalanced class distribution.

---

## Preprocessing
### Steps Applied:
1. **Bounding Box Cropping:** Cropped images using bounding box coordinates.
2. **Resizing:** All images resized to **224 × 224** pixels.
3. **Normalization:** Pixel values normalized using dataset mean & standard deviation.
4. **Data Type Conversion:** Converted images to tensors for PyTorch processing.

---

## Data Augmentation
To improve generalization and address class imbalance, the following augmentations were applied:

### Geometric Transformations:
- **Random Cropping** (varied viewpoints)
- **Resizing with Scaling Variations**
- **Horizontal Flipping** (leveraging symmetry)

### Color Transformations:
- **Color Jittering** (brightness, contrast, saturation variations)
- **Grayscale Conversion** (to prevent reliance on color features)

### Affine Transformations:
- **Rotation** (±10°)
- **Translation** (horizontal/vertical shifts)

### Noise & Blurring:
- **Gaussian Blur** (simulating out-of-focus images)
- **Random Noise** (robustness to real-world data)

---

## Benchmark Comparison
We evaluated **custom CNN models** and **pretrained architectures**:

| Model      | Accuracy (%) |
|------------|--------------|
| ResNet34   | 88           |
| ResNet50   | 89           |
| VGG16      | 84           |
| EfficientNet | 81        |

---

## Custom CNN Architecture
### Structure:
- **4 convolutional blocks**, each with:
  - Convolution layers for feature extraction
  - Batch normalization for stability
  - ReLU activation for non-linearity
  - Max-pooling for dimensionality reduction

Feature map depth:
- **Block 1:** 3 → 64
- **Block 2:** 64 → 128
- **Block 3:** 128 → 256
- **Block 4:** 256 → 256

### Fully Connected Layers:
- **FC1:** Flattened features → 512 neurons
- **Output Layer:** 512 → Number of classes
- **Dropout (50%)** and **Batch Normalization** applied to prevent overfitting

---

## Results for Custom Model
### Baseline Performance
- **Training Loss:** Improved across epochs.
- **Validation Accuracy:** Plateaued at **26.56%** (overfitting present).
- **Test Accuracy:** **27.09%**, confirming limited generalization.

### Suggestions for Improvement
1. **Data Augmentation**
   - Initial augmentation **decreased** accuracy to **6.16%**.
2. **Modified Architecture**
   - Reducing fully connected neurons to 128 and adding convolution layers improved accuracy to **9.26%**.
3. **Regularization**
   - **Increased Dropout** (60%) & **L2 Regularization** to prevent overfitting.
4. **Inference-Time Augmentation (ITA)**
   - Averaged predictions over multiple augmentations → **30.02% test accuracy**.

---

## K-Fold Cross-Validation
| Fold | Validation Accuracy (%) | Test Accuracy (%) | Test Accuracy with ITA (%) |
|------|-------------------------|-------------------|---------------------------|
| 1    | 26.68                   | 26.01             | 28.53                     |
| 2    | 25.28                   | 27.34             | 31.70                     |
| 3    | 27.11                   | 27.30             | 30.15                     |
| 4    | 25.89                   | 27.00             | 30.02                     |
| 5    | 27.79                   | 27.80             | 30.02                     |

---

## Adding a New Class
To evaluate extensibility, **Toyota Tacoma** was added:
- **Train Dataset:** 80 images
- **Test Dataset:** 20 images
- **Test Accuracy:** **85%** (only 3 misclassified images)

---

## Pretrained Models
Fine-tuned **ResNet50, VGG16, DenseNet169, EfficientNet-B4** on the dataset.

| Model         | Parameters | Validation Loss | Validation Accuracy (%) | Test Loss | Test Accuracy (%) |
|--------------|------------|----------------|-------------------------|-----------|-------------------|
| VGG16       | 135M       | 1.0085         | 71.15                   | 1.0338    | 71.12             |
| ResNet50    | 23.9M      | 0.4780         | 86.49                   | 0.4939    | 87.10             |
| DenseNet169 | 12.8M      | 0.4794         | 86.86                   | 0.4780    | 87.51             |
| EfficientNet | 17.9M      | 0.9476         | 74.28                   | 0.9468    | 73.87             |

---

## Observations
1. **Pretrained models outperform custom CNNs** in both accuracy & efficiency.
2. **DenseNet169 achieves the best performance** (87.51% accuracy).
3. **Feature extraction with DenseNet169 & Logistic Regression** performed well but slightly underperformed compared to full DenseNet.

---

## Conclusion
1. **Custom Model**
   - Data augmentation decreased accuracy.
   - **Inference-Time Augmentation** improved test accuracy to **30%**.
2. **Pretrained Models**
   - Fine-tuned architectures significantly **outperformed custom models**.
   - **DenseNet169 and ResNet50** achieved the highest test accuracy.
3. **New Class Addition**
   - The framework successfully **extended to new classes** with **85% accuracy**.

---


