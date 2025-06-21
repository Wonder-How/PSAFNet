# PSAFNet 🧠🎯

## 📌 Disclaimer
This repository contains code developed to support the paper **"Brain-inspired deep learning model for EEG-based low-quality video target detection with phased encoding and aligned fusion"**, now **published in *Expert Systems with Applications (ESWA)*** 🎉.  
👉 [https://doi.org/10.1016/j.eswa.2025.128189](https://doi.org/10.1016/j.eswa.2025.128189)

Please note that this code reflects the final state of the research as published. We encourage the community to build upon and extend this work.

---

## 📁 File Description

- `cross_subject_evaluation.py`  
  Code for cross-subject training and evaluation.

- `PSAFNet.py`  
  PSAFNet: Phase Segment and Aligned Fusion Net (proposed model)  
  Input shape: `[batchsize, 1, channels, timepoints]`

- `data_processing.py`  
  Functions for data loading and segmentation.

- `train.py`  
  Training, validation, and testing logic.

- `utils.py`  
  Utility functions used across the codebase.

- `my_config.py`  
  Hyperparameter configuration file.

- `requirements.txt`  
  Package versions used in the project.

---

## ⚙️ Install

```bash
git clone https://github.com/Wonder-How/PSAFNet.git
cd PSAFNet
pip install -r requirements.txt

If you have any questions about the code or research methods, feel free to reach out:

📧 Email: wonderhow@bit.edu.cn

We welcome collaborations, discussions, and any valuable feedback from the research community!