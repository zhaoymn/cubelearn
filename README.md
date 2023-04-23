# CubeLearn
This is the official repository for paper 

P. Zhao, C. X. Lu, B. Wang, N. Trigoni and A. Markham, "CubeLearn: End-to-end Learning for Human Motion Recognition from Raw mmWave Radar Signals," in IEEE Internet of Things Journal, doi: 10.1109/JIOT.2023.3237494.

## Code
The training, validation and testing set split is provided in csv format so that you can replicate the results in the paper if you want to. The code is provided mianly as a reference. If you want to run the code you'll have to modify the dataset path and select the model which you want to run. The complex layers are based on https://github.com/ivannz/cplxmodule.

## Dataset
The dataset can be accessed at 

https://www.dropbox.com/sh/qafpm5tstong4ll/AABc4F_moM9LpiKKX9OhBXO3a?dl=0

For each file, it is a numpy array of size (2, T, 128, 12, 256), where 2 is the real and complex part of the raw data, T is the timestamps (10 for HGR and AGR, 20 for HAR), 128 is the number of chirps in a frame, 12 is the virtual antennas with the following arrangement:
```
      8  9  10 11
0  1  2  3  4  5  6  7
```
and 256 is the number of samples per virtual antenna per chirp.

The format of the file name is {user id}\_{gesture/activity}\_{sample id}.npy



## Citation

If you find the paper or the code/data helpful to your research/project, please consider citing our paper:

```bibtex
@article{zhao2023cubelearn,
  title={CubeLearn: End-to-end learning for human motion recognition from raw mmWave radar signals},
  author={Zhao, Peijun and Lu, Chris Xiaoxuan and Wang, Bing and Trigoni, Niki and Markham, Andrew},
  journal={IEEE Internet of Things Journal},
  year={2023},
  publisher={IEEE}
}
```
