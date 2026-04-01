# Multi-Class Gravitational Lens Substructure Classification

## 1. Overview

In this project, we developed a deep learning model using PyTorch to classify gravitational lensing images based on the type of substructure they exhibit. The task was to distinguish between three categories:

- **No Substructure (`no`)**: Images showing strong lensing without noticeable substructure.
- **Sphere Substructure (`sphere`)**: Images with spherical substructures, analogous to cold dark matter subhalos.
- **Vortex Substructure (`vort`)**: Images exhibiting vortex-like patterns, potentially linked to alternative dark matter physics (e.g., axion or superfluid models).

The primary objective was to build a **highly accurate and robust classifier** capable of distinguishing subtle physical features across these categories.

---

## 2. The Dataset

- Balanced dataset with **~2500 samples per class**
- Total dataset size: **~7500 images**
- Images represent simulated gravitational lensing systems with different substructure morphologies
- Data split into **training, and validation sets**
- Labels correspond to physical substructure classes (`no`, `sphere`, `vort`)

---

## 3. Our Approach

### Model Selection:
- Efficient B1 for image classification
- Final fully connected layer adapted for **3-class softmax output**

### Data Preprocessing and Augmentation:
- Image normalization
- Separate transforms for training and validation
- Basic augmentations to improve generalization

### Training Strategy:
- Loss: **Cross-Entropy Loss**
- Optimizer: **Adam**
- Multi-epoch training with validation monitoring
- Best model saved based on validation performance

---

## 4. Evaluation Metrics

- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC (One-vs-Rest)

---

## 5. Results

### Overall Performance
- **AUC:** 99.784%
- **Test Accuracy:** ~98.04%
- **Macro F1-score:** 0.98
- **Weighted F1-score:** 0.98

### Class-wise Performance

| Class    | Precision| Recall | F1-score | Support |
|----------|----------|--------|----------|---------|
| no       | 0.97     | 1.00   | 0.98     | 2500    |
| sphere   | 0.99     | 0.96   | 0.97     | 2500    |
| vort     | 0.99     | 0.99   | 0.99     | 2500    |

|accuracy  |                       0.98   | 7500    |
|macro avg | 0.98     | 0.98   |   0.98   | 7500    |
|wtd avg   | 0.98     | 0.98   |   0.98   | 7500    |



---

## Model Weights

- Saved trained model weights after training
- Uploaded in the github as ![Trained Weights][test 1/](test_I_trained_wts_best_40ep.pth)

---

## Visualizations

### Confusion Matrix
![Confusion Matrix](test 1/Test_I_confusion.png)

### ROC Curve
![ROC Curve](test 1/Test_I_ROC.png)

### ROC-AUC Scores
- **total:** 99.784%
- **no:** 99.8%
- **sphere:** 99.7%  
- **vort:** 99.9%  

These confirm excellent separability and strong model confidence.

---


## 7. Dependencies

- Python >= 3.8  
- PyTorch  
- Torchvision  
- NumPy  
- Matplotlib  
- Scikit-learn  

---

## Author

**Milind Sarkar**  
IISER Mohali  