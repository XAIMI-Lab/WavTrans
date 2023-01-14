### Environment and Dependencies
Requirements:
* Python 3.6
* Pytorch 1.4.0 
* scipy
* scikit-image
* opencv-python
* tqdm

Our code has been tested with Python 3.6, Pytorch 1.4.0, torchvision 0.5.0, CUDA 10.0 on Ubuntu 18.04.


### To Run Our Code
- Train the model
```bash
python train.py
```

- Test the model
```bash
python test.py
```
### Citation
```bash
@inproceedings{Li2022WavTransSW,
  title={WavTrans: Synergizing Wavelet and Cross-Attention Transformer for Multi-contrast MRI Super-Resolution},
  author={Guangyuan Li and Jun Lyu and Chengyan Wang and Qi Dou and Jin Qin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  year={2022}
}
```
Our code is built on  [DuDoRNet](https://github.com/bbbbbbzhou/DuDoRNet) and [SwinIR](https://github.com/JingyunLiang/SwinIR). Thank the authors for sharing their codes!