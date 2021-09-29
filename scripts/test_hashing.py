import json
import os
import time
from datetime import datetime
from pprint import pprint

import torch

import configs
from functions.hashing import calculate_mAP
from scripts.train_hashing import prepare_dataloader, prepare_model, test_hashing
from utils import io


def main(config):
    device = torch.device('cuda')
    io.init_save_queue()

    start_time = time.time()
    configs.seeding(config['seed'])

    logdir = config['logdir']
    pprint(config)

    test_loader, db_loader = prepare_dataloader(config)[1:]

    if config['dataset'] == 'cifar10':
        config['R'] = len(db_loader.dataset)

    print('Test Dataset', len(test_loader.dataset))
    print('DB Dataset', len(db_loader.dataset))

    codebook = torch.load(f'{logdir}/outputs/codebook.pth').to(device)
    model, extrabit = prepare_model(config, device, codebook)
    model.load_state_dict(torch.load(f'{logdir}/models/best.pth'))

    result_logdir = logdir + '/testing_results'
    count = 0
    orig_logdir = result_logdir
    result_logdir = orig_logdir + f'/{count:03d}'
    while os.path.isdir(result_logdir):
        count += 1
        result_logdir = orig_logdir + f'/{count:03d}'
    os.makedirs(result_logdir, exist_ok=True)

    json.dump(config, open(os.path.join(result_logdir, 'eval_history.json'), 'w+'))

    loss_param = config.copy()
    loss_param['device'] = device

    print('Testing Start')

    res = {}

    test_meters, test_out = test_hashing(model, codebook, test_loader, loss_param, True)
    db_meters, db_out = test_hashing(model, codebook, db_loader, loss_param, True)

    for key in test_meters: res['test_' + key] = test_meters[key].avg
    for key in db_meters: res['db_' + key] = db_meters[key].avg

    res['mAP'] = calculate_mAP(db_out['codes'], db_out['labels'],
                               test_out['codes'], test_out['labels'],
                               loss_param['R'], loss_param['map_threshold'])

    json.dump(res, open(result_logdir + '/history.json', 'w+'))
    io.fast_save({'test': test_out, 'db': db_out}, result_logdir + '/outputs.pth')

    total_time = time.time() - start_time
    print(f'Testing End at {datetime.today().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'mAP: {res["mAP"]:.4f}')
    print(f'Total time used: {total_time / (60 * 60):.2f} hours')
    print('Waiting for save queue to end')
    io.join_save_queue()
    print(f'Done: {logdir}')
