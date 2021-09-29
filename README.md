# OrthoHash

[ArXiv](Coming Soon)

### Official pytorch implementation of the paper: "One Loss for All: Deep Hashing with a Single Cosine Similarity based Learning Objective"

#### NeurIPS 2021

Released on September 29, 2021

# How to run

### Training
```bash
python main.py --codebook-method B --ds cifar10
```

Run `python main.py --help` to check what hyperparameters to run with. All the hyperparameters are the default parameters to get the performance in the paper.

### Testing

```bash
python val.py -l /path/to/logdir
```

# Dataset

### Category-level Retrieval (ImageNet, NUS-WIDE, MS-COCO)

You may refer to this repo (https://github.com/swuxyj/DeepHash-pytorch) to download the datasets. I was using the same dataset format as [HashNet](https://github.com/thuml/HashNet).

Dataset sample: https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/imagenet/test.txt

For CIFAR-10, the code will auto generate a dataset at the first run. See `utils/datasets.py`.

### Instance-level Retrieval (GLDv2, ROxf, RPar)

This code base is a simplified version and we did not include everything yet. We will release a version that will include the dataset we have generated and also the corresponding evaluation metrics, stay tune.

# Performance Tuning (Some Tricks)

I have found some tricks to further improve the mAP score.

### Database Shuffling

If you shuffle the order of database before `calculate_mAP`, you might get 1~2% improvement in mAP.

It is because many items with same hamming distance will not be sorted properly, hence it will affect the mAP calculation.

### Codebook Method

Run with `--codebook-method O` might help to improve mAP by 1~2%. The improvement is explained in our paper. 

# Feedback

Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to `jiuntian at gmail.com` or `kamwoh at gmail.com` or `cs.chan at um.edu.my`.

# Related Work

1. Deep Polarized Network (DPN) - (https://github.com/kamwoh/DPN)

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2021 Universiti Malaya.
