## Official-PyTorch-Implementation-of-TransMEF

<p align="center">
  <img src="https://github.com/miccaiif/TransMEF/blob/main/method.png" width="480">
</p>

This is a PyTorch/GPU implementation of the AAAI2022 paper [TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning](https://arxiv.org/abs/2112.01030):

### For training
* We use the [MS-COCO dataset](https://voxel51.com/docs/fiftyone/user_guide/dataset_zoo/datasets.html#dataset-zoo-coco-2017) for self-supervised training 
and all images are converted to 256 * 256 grayscale images.
* For a quick start, please run 
```shell
python train_TransMEF.py --root './coco' --batch_size 24 --save_path './train_result_TransMEF' --summary_name 'TransMEF_qiuck_start_'
```

### For fusion
* We use the [benchmark dataset MEFB](https://github.com/xingchenzhang/MEFB) for evaluation, and all images are converted to 256 * 256 grayscale images.
We provide a convenient implementation of the transformation. Please refer to [resize_all.py](https://github.com/miccaiif/TransMEF/blob/main/resize_all.py) for details.

* For a quick start, please run 
```shell
python fusion_gray_TransMEF.py --model_path './best_model.pth' --test_path './MEFB_dataset/' --result_path './TransMEF_result' 
```
* Managing RGB Input

    We refer to the [code of hanna-xu](https://github.com/hanna-xu/utils/tree/master/fusedY2RGB) to convert the fused image into a color image.
    
* Managing Arbitrary input size images
  
    We recommend to use the sliding window strategy to fuse input images of arbitrary non-256 * 256 size, i.e., fusing images of 256 * 256 window size at a time.
    
* Best model in this paper

    Please refer to this [link](https://drive.google.com/file/d/1a-i_M7i-rns9pyu-PxkOKuL3RWoza8em/view?usp=sharing) for the best model employed in this paper.

### Fusion results of TransMEF

![Main fusion results](https://github.com/miccaiif/TransMEF/blob/main/main_results.png)

![Supplementary outdoor fusion results](https://github.com/miccaiif/TransMEF/blob/main/more_result.png)

![Supplementary indoor fusion results](https://github.com/miccaiif/TransMEF/blob/main/more_result_indoor.png)

### Evaluation metrics of TransMEF

![Main metrics](https://github.com/miccaiif/TransMEF/blob/main/main_metric.png)


### Citation
If this work is helpful to you, please cite it as:
```
@article{qu2021transmef,
  title={TransMEF: A Transformer-Based Multi-Exposure Image Fusion Framework using Self-Supervised Multi-Task Learning},
  author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
  journal={arXiv preprint arXiv:2112.01030},
  year={2021}
}
```

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
