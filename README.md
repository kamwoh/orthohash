# OrthoHash

[ArXiv](https://arxiv.org/abs/2109.14449) | <a href="https://github.com/kamwoh/orthohash/blob/main/docs/suppmat.pdf">Supplementary Material</a>

### Official pytorch implementation of the paper: "One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective"

#### NeurIPS 2021

Released on September 29, 2021

# Description

This paper proposes a novel deep hashing model with only a single learning objective which is a simplification from most state of the art papers generally use lots of losses and regularizer. Specifically, it maximizes the cosine similarity between the continuous codes and their corresponding binary orthogonal codes to ensure both the discriminative capability of hash codes and the quantization error minimization. Besides, it adopts a Batch Normalization layer to ensure code balance and leverages the Label Smoothing strategy to modify the Cross-Entropy loss to tackle multi-labels classification. Extensive experiments show that the proposed method achieves better performance compared with the state-of-the-art multi-loss hashing methods on several benchmark datasets.

[comment]: <> (![animation]&#40;static/orthohash_animation.gif&#41;)

<img src="https://github.com/kamwoh/orthohash/blob/main/static/orthohash_animation.gif?raw=true" width="250" height="250" />

https://user-images.githubusercontent.com/23725126/136496588-a79646ee-6d91-4c6b-a8cf-957e3b4bd175.mp4



# How to run

### Training
```bash
python main.py --codebook-method B --ds cifar10 --margin 0.3 --seed 59495
```

Run `python main.py --help` to check what hyperparameters to run with. All the hyperparameters are the default parameters to get the performance in the paper.

The above command should obtain mAP of 0.824 at best for CIFAR-10. 

### Testing

```bash
python val.py -l /path/to/logdir
```

# Dataset

### Category-level Retrieval (ImageNet, NUS-WIDE, MS-COCO)

- You may refer to this repo (https://github.com/swuxyj/DeepHash-pytorch) to download the datasets. We were using the same dataset format as [HashNet](https://github.com/thuml/HashNet). See `utils/datasets.py` to understand how to save the data folder.

- Dataset sample: https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/imagenet/test.txt

- For CIFAR-10, the code will auto generate a dataset at the first run. See `utils/datasets.py`.

### Instance-level Retrieval (GLDv2, ROxf, RPar)

- This code base is a simplified version and we did not include everything yet. We will release a version that will include the dataset we have generated and also the corresponding evaluation metrics, stay tune.

(Update) The datasets for instance-level retrieval are uploaded here (https://drive.google.com/drive/folders/1yVAC501OAFJac0EyQDrl6otLkLysCRjt?usp=sharing). We still haven't updated the codebase for loading the embedding and the evaluation, but you can still refer to our supplementary metarial for the evaluation protocol.

# Performance Tuning (Some Tricks)

We have found some tricks to further improve the mAP score.  

### Avoid Overfitting

- As set by the previous protocols, the dataset is small in size (e.g., 13k training images for ImageNet100) and hence overfitting can easily happen during the training. 

#### An appropriate learning rate for backbone

- We set a 10x lower learning rate for the backbone to avoid overfitting.

#### Cosine Margin

- An appropriate higher cosine margin should be able to get higher performance as it slow down the overfitting. 

#### Data Augmentation

- We did not tune the data augmentation, but we believe that appropriate data augmentation can obtain a little bit of improvement in mAP.

### Database Shuffling

- If you shuffle the order of database before `calculate_mAP`, you might get 1~2% improvement in mAP.

- It is because many items with same hamming distance will not be sorted properly, hence it will affect the mAP calculation.

### Codebook Method

- Run with `--codebook-method O` might help to improve mAP by 1~2%. The improvement is explained in our paper. 

# Feedback

Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to `jiuntian at gmail.com` or `kamwoh at gmail.com` or `cs.chan at um.edu.my`.

# Related Work

1. Deep Polarized Network (DPN) - (https://github.com/kamwoh/DPN)

# Notes

1. You may get slightly different performance as compared with the paper, the random seed sometime affect the performance a lot, but should be very close.
2. We re-run the training (64-bit ImageNet100) with this simplified version can obtain 0.709~0.710 on average (paper: 0.711).

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2021 Universiti Malaya.
