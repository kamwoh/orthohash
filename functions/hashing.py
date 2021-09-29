import logging

import numpy as np
import torch

import configs
from utils.misc import Timer


def get_hamm_dist(codes, centroids, margin=0, normalize=False):
    with torch.no_grad():
        nbit = centroids.size(1)
        dist = 0.5 * (nbit - torch.matmul(codes.sign(), centroids.sign().t()))

        if normalize:
            dist = dist / nbit

        if margin == 0:
            return dist
        else:
            codes_clone = codes.clone()
            codes_clone[codes_clone.abs() < margin] = 0
            dist_margin = 0.5 * (nbit - torch.matmul(codes_clone.sign(), centroids.sign().t()))
            if normalize:
                dist_margin = dist_margin / nbit
            return dist, dist_margin


def get_codes_and_labels(model, loader):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    vs = []
    ts = []
    for e, (d, t) in enumerate(loader):
        print(f'[{e + 1}/{len(loader)}]', end='\r')
        with torch.no_grad():
            # model forward
            d, t = d.to(device), t.to(device)
            v = model(d)
            if isinstance(v, tuple):
                v = v[0]

            vs.append(v)
            ts.append(t)

    print()
    vs = torch.cat(vs)
    ts = torch.cat(ts)
    return vs, ts


def calculate_mAP(db_codes, db_labels,
                  test_codes, test_labels,
                  R, threshold=0.):
    # clone in case changing value of the original codes
    db_codes = db_codes.clone()
    test_codes = test_codes.clone()

    # if value within margin, set to 0
    if threshold != 0:
        db_codes[db_codes.abs() < threshold] = 0
        test_codes[test_codes.abs() < threshold] = 0

    # binarized
    db_codes = torch.sign(db_codes)  # (ndb, nbit)
    test_codes = torch.sign(test_codes)  # (nq, nbit)

    db_labels = db_labels.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    dist = []
    nbit = db_codes.size(1)

    timer = Timer()
    total_timer = Timer()

    timer.tick()
    total_timer.tick()

    with torch.no_grad():
        db_codes_ttd = configs.tensor_to_dataset(db_codes)
        db_codes_loader = configs.dataloader(db_codes_ttd, 32, False, 0, False)

        # calculate hamming distance
        for i, db_code in enumerate(db_codes_loader):
            dist.append(0.5 * (nbit - torch.matmul(test_codes, db_code.t())).cpu())
            timer.toc()
            print(f'Distance [{i + 1}/{len(db_codes_loader)}] ({timer.total:.2f}s)', end='\r')

        dist = torch.cat(dist, 1)  # .numpy()
        print()

    # fast sort
    timer.tick()
    # different sorting will have affect on mAP score! because the order with same hamming distance might be diff.
    # unsorted_ids = np.argpartition(dist, R - 1)[:, :R]

    # torch sorting is quite fast, pytorch ftw!!!
    topk_ids = torch.topk(dist, R, dim=1, largest=False)[1].cpu()
    timer.toc()
    print(f'Sorting ({timer.total:.2f}s)')

    # calculate mAP
    timer.tick()
    APx = []
    for i in range(dist.shape[0]):
        label = test_labels[i, :]
        label[label == 0] = -1
        idx = topk_ids[i, :]
        # idx = idx[np.argsort(dist[i, :][idx])]
        imatch = np.sum(np.equal(db_labels[idx[0: R], :], label), 1) > 0
        rel = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if rel != 0:
            APx.append(np.sum(Px * imatch) / rel)
        else:  # didn't retrieve anything relevant
            APx.append(0)
        timer.toc()
        print(f'Query [{i + 1}/{dist.shape[0]}] ({timer.total:.2f}s)', end='\r')

    print()
    total_timer.toc()
    logging.info(f'Total time usage for calculating mAP: {total_timer.total:.2f}s')

    return np.mean(np.array(APx))


def sign_dist(inputs, centroids, margin=0):
    n, b1 = inputs.size()
    nclass, b2 = centroids.size()

    assert b1 == b2, 'inputs and centroids must have same number of bit'

    # sl = relu(margin - x*y)
    out = inputs.view(n, 1, b1) * centroids.sign().view(1, nclass, b1)
    out = torch.relu(margin - out)  # (n, nclass, nbit)

    return out


def calculate_similarity_matrix(centroids):
    nclass = centroids.size(0)
    sim = torch.zeros(nclass, nclass, device=centroids.device)

    for rc in range(nclass):
        for cc in range(nclass):
            sim[rc, cc] = (centroids[rc] == centroids[cc]).float().mean()

    return sim
