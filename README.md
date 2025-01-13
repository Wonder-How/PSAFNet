# README

## Disclaimer
This repository contains code developed to support a scholarly paper currently under review. Please note that the paper is currently under review and the results, methods, or conclusions may change based on the final review process. The provided code reflects the state of the research at the time of submission and may be subject to further revisions.


## File discription
- `cross_subject_evaluation.py`

  The code of cross subject training and evaluation 

- `PSAFNet.py`

  PSAFNet: Phase Segment and Aligned Fusion Net (proposed model)

  The size of input: [batchsize, 1, channels, timepoints]

- `data_processing.py`

  The function of data loading and segmentation

- `train.py`

  The function of training, validation and testing

- `utils.py`

  Other universal function

- `my_config.py`

  Hyperparameters configuration file

- `requirements.txt`

  Version of the package to use

## Install
```bash
git clone https://github.com/Wonder-How/PSAFNet.git
cd PSAFNet
pip install -r requirements.txt
```


## Contact Information

If you have any questions about the code or research methods, please contact us via:

- Email: [wonderhow@bit.edu.cn](mailto:wonderhow@bit.edu.cn)

We encourage other researchers to use and extend these methods, and we look forward to your valuable feedback.