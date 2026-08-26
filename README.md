

# **Alzheimer’s Disease Detection Model (ResNet-18 + SHAP)**

*A hybrid research–engineering implementation for MRI-based Alzheimer’s classification.*


---

## **Overview**

This project implements a complete end-to-end deep learning pipeline for classifying structural MRI brain scans into four Alzheimer’s disease categories: Non-Demented, Very Mild Dementia, Mild Dementia, and Moderate Dementia.
The system combines a modified **SafeResNet-18** architecture with **SHAP GradientExplainer** for interpretable predictions. The backend includes preprocessing, weighted loss optimization for class imbalance, visualization modules, and model evaluation tools.

This repository contains the codebase, model training workflow, SHAP explainability utilities, and all experiment logs aligned with the methodology described in the project’s final report.


---

## **Abstract**

This work presents a ResNet-18–based classifier trained on the OASIS MRI dataset to detect Alzheimer’s disease severity across four classes. A weighted cross-entropy loss addresses class imbalance, and images are preprocessed to 224×224 with ImageNet standardization. SHAP GradientExplainer is integrated for model interpretability by disabling inplace ReLU operations.

The model achieves **98.90% accuracy**, **0.9787 macro precision**, **0.9954 macro recall**, and **0.9868 macro F1-score**, demonstrating strong discriminative performance across all classes. SHAP value maps provide clinically interpretable visualizations of brain regions contributing to predictions.


---

## **Architecture**

### **System Flowchart**

```mermaid
flowchart TD

A[Raw MRI Image] --> B[Data Loader]
B --> C[Preprocessing\nResize, Crop, Normalize]

C --> D[SafeResNet-18\nResidual Blocks]
D --> E[Global Average Pooling]

E --> F[Fully Connected Layer\nClass Scores]
F --> G[Softmax]
G --> H[Predicted Class]

D --> I[SHAP GradientExplainer\nBackground Set]
I --> J[SHAP Value Computation]
J --> K[SHAP Maps]

H --> L[Evaluation Metrics]
K --> M[Explainability Reports]

subgraph Data_Pipeline
B --> C
end

subgraph Model_Inference
C --> D --> E --> F --> G --> H
end

subgraph Explainability
D --> I --> J --> K
end

subgraph Outputs
H --> L
K --> M
end


```

---

## **Key Features**

### **Model**

* Modified **SafeResNet-18** architecture
* ReLU layers set to `inplace=False` for SHAP compatibility
* Fully connected layer replaced with 512 → 4 classifier head

### **Explainability**

* SHAP GradientExplainer on PyTorch model
* Background sampling from validation set
* Generates attribution heatmaps and class-wise contribution plots

### **Training**

* Weighted Cross-Entropy
* Adam optimizer (lr = 1e-4)
* StepLR scheduler (step_size=5, gamma=0.1)
* Batch size: 32
* Image input: 224 × 224
* Metrics: Accuracy, Precision (macro), Recall (macro), F1-score (macro)

All values sourced from the final report.


---

## **Dataset**

* **Dataset:** OASIS MRI (Kaggle mirror)
* **Total images:** 86,437
* **Classes:** 4 (Non-Demented → Moderate Dementia)
* **Train/Validation Split:** 80/20
* **Preprocessing:**

  * Resize to 248×496
  * CenterCrop to 224×224
  * Normalize using ImageNet statistics

Dataset information derived from the report, pages 1–4.


---

## **Model Architecture**

The SafeResNet-18 architecture follows the ResNet-18 backbone with modifications documented on page 2 of the report:

* Initial convolution → batch normalization → ReLU → max pooling
* Four residual layers with identity shortcuts
* Global average pooling
* Linear layer producing four class logits

The architecture diagram in the report illustrates all functional blocks used in this repository’s implementation.


---

## **Results**

Final evaluation metrics as reported (page 5):

| Metric          | Value  |
| --------------- | ------ |
| Accuracy        | 0.9890 |
| Macro Precision | 0.9787 |
| Macro Recall    | 0.9954 |
| Macro F1-Score  | 0.9868 |
| Mean Confidence | 0.9882 |
| Validation Loss | 0.0170 |
| Training Loss   | 0.0099 |

### **Confusion Matrix Overview**

* Non-Demented: 13,271 correct
* Very Mild Dementia: 2,753 correct
* Mild Dementia: 993 correct
* Moderate Dementia: 80 correct

These values indicate strong separation across all classes with minimal confusion, as shown in the confusion matrix on pages 5–6.


---

## **Project Structure**

```
Alzhiemer-Detection-Model/
│
├── data/                         # Dataset (not included)
├── models/                       # Model weights
├── outputs_tcc_resnet18/         # Metrics, logs, plots
├── shap_analysis/                # SHAP heatmaps and analysis
├── train.py                      # Training script
├── model.py                      # SafeResNet18 implementation
├── shap_explain.py               # SHAP integration
├── utils.py                      # Helper functions
└── README.md
```

---

## **Installation**

### Clone the repository

```
git clone https://github.com/Saurabh89580/Alzhiemer-Detection-Model
cd Alzhiemer-Detection-Model
```

### Install dependencies

```
pip install -r requirements.txt
```

---

## **Usage**

### Train the model

```
python train.py
```

### Generate SHAP explanations

```
python shap_explain.py
```

Outputs will appear in:

* `outputs_tcc_resnet18/`
* `shap_analysis/`

---

## **Roadmap**

* Extend SHAP to multi-sample summaries
* Add Grad-CAM comparison module
* Implement EfficientNet / ensemble benchmarking
* Introduce automated hyperparameter tuning
* Add inference API endpoint for deployment

---

## **References**

[1] D. S. Marcus et al., “The OASIS project: Cross-sectional MRI data...,” Journal of Cognitive Neuroscience, 2007.
[2] H.-I. Suk, S.-W. Lee, and D. Shen, “Hierarchical feature representation...” NeuroImage, 2014.
[3] S. Korolev et al., “Residual and plain CNNs for 3D brain MRI classification,” ISBI, 2017.
[4] J. Islam and Y. Zhang, “Brain MRI analysis for AD diagnosis...,” Brain Informatics, 2018.
[5] S. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.
[6] V. Arvidsson et al., “Explainable AI in medical imaging,” Insights into Imaging, 2023.
[7] M. Böhle et al., “Layer-wise relevance propagation...,” Frontiers in Aging Neuroscience, 2019.
[8] K. He et al., “Deep Residual Learning for Image Recognition,” CVPR, 2016.
[9] N. Tajbakhsh et al., “CNNs for medical image analysis...” IEEE TMI, 2016.
[10] J. Johnson and T. M. Khoshgoftaar, “Survey on deep learning with class imbalance,” Journal of Big Data, 2019.
[11] J. Wen et al., “CNNs for Alzheimer's classification,” Medical Image Analysis, 2020.



---


