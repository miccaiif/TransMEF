## 🌟 Official-PyTorch-Implementation-of-TransMEF

<p align="center">
  <img src="https://github.com/miccaiif/TransMEF/blob/main/method.png" width="720">
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
* We use the [benchmark dataset MEFB](https://github.com/xingchenzhang/MEFB) for evaluation, and all images are converted to 256 * 256 grayscale png images. Note that our test metrics may not be consistent with those reported in the MEFB research due to the resizing and format conversion.

  We provide a convenient implementation of the transformation. Please refer to [resize_all.py](https://github.com/miccaiif/TransMEF/blob/main/resize_all.py) for         details.
  
  We provide an example of the dataset [here](https://github.com/miccaiif/TransMEF/tree/main/MEFB_dataset_example). Please note the data path and format!

* For a quick start, please run 
```shell
python fusion_gray_TransMEF.py --model_path './best_model.pth' --test_path './MEFB_dataset/' --result_path './TransMEF_result' 
```
* Managing RGB Input

    We refer to the [code of hanna-xu](https://github.com/hanna-xu/utils/tree/master/fusedY2RGB) to convert the fused image into a color image.
    
* Managing Arbitrary input size images
  
    We recommend to use the sliding window strategy to fuse input images of arbitrary non-256 * 256 size, i.e., fusing images of 256 * 256 window size at a time.
    
    The code is available now at [fusing intput images with arbitary size](https://github.com/miccaiif/TransMEF/blob/main/fusion_arbitary_size_TransMEF_gray.py)
    
* Best model in this paper

    Please refer to this [link for Google Drive](https://drive.google.com/file/d/1a-i_M7i-rns9pyu-PxkOKuL3RWoza8em/view?usp=sharing)/ [link for Baidu Disk](https://pan.baidu.com/s/1PDUkL_z6DLnHa6mIQy-HPA?pwd=jcx3) for the best model employed in this paper.

### Fusion results of TransMEF

![Main fusion results](https://github.com/miccaiif/TransMEF/blob/main/main_results.png)

![Supplementary outdoor fusion results](https://github.com/miccaiif/TransMEF/blob/main/more_result.png)

![Supplementary indoor fusion results](https://github.com/miccaiif/TransMEF/blob/main/more_result_indoor.png)

### Evaluation metrics of TransMEF

![Main metrics](https://github.com/miccaiif/TransMEF/blob/main/main_metric.png)


### Frequently Asked Questions

* The fused images are all black.

  Please mind your input path. It is likely that the input images are not loaded due to the wrong path. You could attempt to pass the test path to the class test_gray and use an absolute path instead of a relative one.

* Fusion of arbitary size images.

  We recommend to use the sliding window strategy to fuse input images of arbitrary non-256 * 256 size, i.e., fusing images of 256 * 256 window size at a time. The code is available now at [fusing intput images with arbitary size](https://github.com/miccaiif/TransMEF/blob/main/fusion_arbitary_size_TransMEF_gray.py)
    
* Error loading pretrained model.

  There are two points to note. 1) Please check the downloaded models, as many errors for the models are caused by download problems. 2) Please do not change the argument '--gpus' in the argparse at the beginning of the code. Its default value is '0,1', which means the provided model was trained using two gpus. Changing this value will cause errors in loading model parameters.
  
* For evaluation metrics.

  You can refer to Zhang's [MEFB](https://github.com/xingchenzhang/MEFB) for details. But several metrics may not work with my practice.

### Citation
If this work is helpful to you, please cite it as:
```
@inproceedings{qu2022transmef,
  title={Transmef: A transformer-based multi-exposure image fusion framework using self-supervised multi-task learning},
  author={Qu, Linhao and Liu, Shaolei and Wang, Manning and Song, Zhijian},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={36},
  number={2},
  pages={2126--2134},
  year={2022}
}
```

### Contact Information
If you have any question, please email to me [lhqu20@fudan.edu.cn](lhqu20@fudan.edu.cn).
