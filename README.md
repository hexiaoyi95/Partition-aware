# Partition-Aware Adaptive Switching Neural Networks for Post-Processing in HEVC

This repository releases the test code for our paper

**Partition-Aware Adaptive Switching Neural Networks for Post-Processing in HEVC (TMM 2020)**

**Weiyao Lin, Xiaoyi He, Xintong Han, Dong Liu, John See, Junni Zou, Hongkai Xiong, Feng Wu**


## Preparation

1. Clone this repository and install the necessary python packages
```Shell
git clone https://github.com/hexiaoyi95/Partition-aware

cd Partition-aware

# visit the requirements.txt for the necessary packages for python2.7
pip install -r requirements.txt
```

2. Prepare test sequences. We provide an example on [One Drive](). The original yuv sequences and compressed sequences are put into two different directories. If the original yuv sequence is named as *seq.yuv*, please name the compressed sequence at QP=37 as *seq_QP37.yuv*.


## Deploy a pre-trained model

- for yuv input posr-processing:
```Shell
usage: inference.py [-h] [--QP QP] [--checkpoint CHECKPOINT]
                    [--test_num TEST_NUM] [--info INFO]
                    [--recYuv_path RECYUV_PATH] [--origYuv_path ORIGYUV_PATH]
                    [--patch_size PATCH_SIZE] [--Yonly]

optional arguments:
  -h, --help            show this help message and exit
  --QP QP, -q QP        test QP value
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        checkpoint to be evaluted
  --test_num TEST_NUM, -n TEST_NUM
                        test frames number, default is 32
  --info INFO           string to describle the result
  --recYuv_path RECYUV_PATH
                        reconstructed yuv dir
  --origYuv_path ORIGYUV_PATH
                        original yuv dir
  --patch_size PATCH_SIZE
                        patch_size, default is 64
  --Yonly               only test Y channel if specified
```

## Released model

We released models for our partition-aware network and VRCNN+partition(i.e., 2-in+MM+AF and VRCNN+MM+AF in our paper) trained at QP=37 on [Google Drive]()


## Citation

If you think this work is helpful for your own research, please consider add following bibtex config in your latex file

```Latex
@article{lin2020partition,
  title={Partition-Aware Adaptive Switching Neural Networks for Post-Processing in HEVC},
  author={Lin, Weiyao and He, Xiaoyi and Han, Xintong and Liu, Dong and John, See and Zou, Junni and Xiong, Hongkai and Wu, Feng},
  journal={IEEE Transaction on Multimedia},
  year={2020},
  organization={IEEE}
}

```
