🫀 EchoNet-Dynamic: Ejection Fraction Classification
====================================================

This repository provides tools and code to utilize the EchoNet-Dynamic dataset (https://echonet.github.io/dynamic/) for classifying cardiac function based on left ventricular ejection fraction (LVEF). The dataset comprises over 10,000 apical-4-chamber echocardiogram videos, each annotated with clinical measurements such as EF, end-systolic volume (ESV), and end-diastolic volume (EDV).

📊 Dataset Overview
-------------------

- Source: Stanford University School of Medicine
- Content: 10,030 apical-4-chamber echocardiogram videos
- Annotations: Each video includes:
  - Left ventricular ejection fraction (EF)
  - End-systolic volume (ESV)
  - End-diastolic volume (EDV)
  - Expert tracings of the left ventricle at end-systole and end-diastole
- Format: Videos are standardized to 112×112 pixels and de-identified to ensure patient privacy

🎯 Objective
------------

The primary goal is to classify echocardiogram videos into categories based on their ejection fraction:

- Normal EF: EF ≥ 50%
- Reduced EF: EF < 50%

This classification aids in the diagnosis of conditions like heart failure with reduced ejection fraction (HFrEF).

📈 Results
==========

We evaluated the performance of ResNet18 and ResNet152 models for classifying ejection fraction status on the EchoNet-Dynamic dataset across different input video resolutions. The classification threshold for EF was set at 50%.

ResNet18 Results
----------------
| Input Size    | Train Acc | Val Acc | Test Acc | AUROC | AUPRC |
|---------------|-----------|---------|----------|--------|--------|
| 28x28         | 0.97      | 0.87    | 0.85     | 0.87   | 0.83   |
| 56x56         | 0.93      | 0.87    | 0.87     | 0.90   | 0.87   |
| 112x112       | 0.92      | 0.88    | 0.87     | 0.90   | 0.88   |
| 28x28 (w)     | 0.94      | 0.83    | 0.81     | 0.85   | 0.80   |
| 56x56 (w)     | 0.92      | 0.86    | 0.85     | 0.90   | 0.86   |
| 112x112 (w)   | 0.94      | 0.88    | 0.86     | 0.88   | 0.86   |

Notes:
- (w) = weighted evaluation under class imbalance.
- Trainable Parameters: ~66.35M

ResNet152 Results
-----------------
| Input Size    | Train Acc | Val Acc | Test Acc | AUROC | AUPRC |
|---------------|-----------|---------|----------|--------|--------|
| 28x28         | 0.87      | 0.86    | 0.88     | 0.88   | 0.72   |
| 56x56         | 0.92      | 0.87    | 0.86     | 0.89   | 0.75   |
| 112x112       | 0.98      | 0.86    | 0.85     | 0.84   | 0.68   |

- Trainable Parameters: ~236.19M
- FLOPs: from 1.8B (28x28) to 11.3B (112x112)

Training Info:
- Epochs ranged from 2 to 9
- Batch sizes of 8 or 16 were used

📌 Conclusion
============

Our results demonstrate that large models like ResNet152 are not strictly necessary to achieve competitive performance on the EchoNet-Dynamic classification task. Smaller architectures such as ResNet18 performed comparably well in terms of accuracy and AUROC, while requiring significantly fewer parameters and computational resources.

We also evaluated the impact of applying class weighting to address class imbalance. However, the use of weighted loss functions did not lead to noticeable improvements in AUROC or AUPRC, suggesting that class imbalance in this dataset may not substantially hinder model performance.

In summary, compact models paired with standard training techniques can be sufficient for ejection fraction classification, offering a more computationally efficient and scalable alternative for clinical applications.
