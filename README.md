# Deep Polarized Network (IJCAI 2020)

Official pytorch implementation of the paper:

- Deep Polarized Network for Supervised Learning of Accurate Binary Hashing Codes ([IJCAI 2020](https://www.ijcai.org/proceedings/2020/115))

Released on July 7, 2021

### Other Implementations

https://github.com/swuxyj/DeepHash-pytorch/blob/master/DPN.py

# How to run

### Training
```bash
python main.py
```

Run `python main.py --help` to check what hyperparameters to run with. All the hyperparameters are the default parameters to get the performance in the paper.

### Testing

```bash
python val.py -l /path/to/logdir -m 0  # normal
python val.py -l /path/to/logdir -m 1  # ternary
```

# Dataset

You may refer to this repo (https://github.com/swuxyj/DeepHash-pytorch) to download the datasets. I was using the same dataset format as [HashNet](https://github.com/thuml/HashNet).

Dataset sample: https://raw.githubusercontent.com/swuxyj/DeepHash-pytorch/master/data/imagenet/test.txt

# Performance Tuning (Some Tricks)

I have found some tricks to further improve the mAP score.

### Database Shuffling

If you shuffle the order of database before `calculate_mAP`, you might get 1~2% improvement in mAP.

It is because many items with same hamming distance will not be sorted properly, hence it will affect the mAP calculation.

### Regularization on hash layer output

Run with `--reg 0.001` might help to improve mAP a little bit.

### Centroids Method

Run with `--centroid-method O` might help to improve mAP by 1~2%.

# Notes

The original code base is the private asset of Webank. This repo is *a re-implementation of the paper*, therefore the performance you computed from this repo might not be exactly the same as the paper (but should be quite close with only +-1% difference)

# Feedback

Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to `lixinfan at webank.com` or `kamwoh at gmail.com` or `cs.chan at um.edu.my`.

# License and Copyright

The project is open source under BSD-3 license (see the `LICENSE` file).

©2020 Webank and University of Malaya.