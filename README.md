This is an unofficial code for implementing MobileFaceSwap.

You can install all envs by the provided requiremnets.txt.

### [MobileFaceSwap: A Lightweight Framework for Video Face Swapping (AAAI 2022)](https://arxiv.org/abs/2201.03808)

---

**Dependencies**

- CUDA 11.3
- cuDNN 8.2
- paddlepaddle==2.1.2.post112
- insightface==0.2.1
- opencv==4.7.0.72
- numpy==1.19.2
- torch==1.9.0

**Getting Started**

1. The pretrained models can be downloaded from [Baidu Drive](https://pan.baidu.com/s/14_Wat-OA6ljGfR3Hk8Fk6A) (passward:f6wu) or [Google Drive](https://drive.google.com/file/d/1ZIzGLDB15GRAZAbkfNR0hNWdgQpxeA_r/view?usp=sharing).

2. Run the codes as follows for image or video tests.

```
python image_test.py --target_img_path data/xxx.png --source_img_path data/xxx.png --output_dir results --use_gpu True

### video_test.py is under testing.
##  python video_test.py --target_video_path data/xxx.mp4 --source_img_path data/xxx.png --output_dir results --use_gpu True
```


**Results**

![](docs/demo.png)

![](docs/video.gif)

Target:

![](images/source.png)

Source:

![](images/target.png)

Swapped:

![](results/source.png)

**Citation**
```
@inproceedings{xu2022MobileFaceSwap,
  title={MobileFaceSwap: A Lightweight Framework for Video Face Swapping},
  author={Xu, Zhiliang and Hong, Zhibin and Ding, Changxing and Zhu, Zhen and Han, Junyu and Liu, Jingtuo and Ding, Errui},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2022}
}
```
