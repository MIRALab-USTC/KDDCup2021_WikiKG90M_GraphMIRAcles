import os
import pickle
import json
import numpy as np
import sys
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
import pdb
from collections import defaultdict
import torch.nn.functional as F
import torch
import ipdb
import subprocess

def eval_h10(valid_result_dict):

    t_correct_index = torch.from_numpy(valid_result_dict['h,r->t']['t_correct_index'])
    t_pred_top10 = torch.from_numpy(valid_result_dict['h,r->t']['t_pred_top10'])

    h10 = ((t_correct_index.view(-1,1) == t_pred_top10).sum() / t_correct_index.shape[0]).item()

    return h10

# python evaluate_single.py $AVE_PATH $NUM_PROC $ENSEMBLE_PATH
if __name__ == '__main__':
    path = sys.argv[1]
    num_proc = int(sys.argv[2])
    ensemble_path = sys.argv[3]

    all_file_names = os.listdir(path)
    test_file_names = [name for name in all_file_names if '.pkl' in name and 'test' in name]
    valid_file_names = [name for name in all_file_names if '.pkl' in name and 'valid' in name]
    steps = [int(name.split('.')[0].split('_')[-1]) for name in valid_file_names if 'valid_0' in name]
    steps.sort()
    evaluator = WikiKG90MEvaluator()
    device = torch.device('cpu')

    all_test_dicts = []
    best_valid_mrr = -1
    best_valid_idx = -1
    best_step = -1

    mrr_list = []
    h10_list = []

    print("evaluating {} ...".format(os.path.basename(path)))
    for i, step in enumerate(steps):
        valid_result_dict = defaultdict(lambda: defaultdict(list))
        for proc in range(num_proc):
            valid_result_dict_proc = torch.load(os.path.join(path, "valid_{}_{}.pkl".format(proc, step)),
                                                map_location=device)

            for result_dict_proc, result_dict in zip([valid_result_dict_proc],
                                                     [valid_result_dict]):
                for key in result_dict_proc['h,r->t']:
                    result_dict['h,r->t'][key].append(result_dict_proc['h,r->t'][key].numpy())
        for result_dict in [valid_result_dict]:
            for key in result_dict['h,r->t']:
                result_dict['h,r->t'][key] = np.concatenate(result_dict['h,r->t'][key], 0)

        metrics = evaluator.eval(valid_result_dict)
        metric = 'mrr'
        # print("valid-{} at step {}: {}".format(metric, step, metrics[metric]))
        # print(metrics[metric])
        # print(step)
        mrr_list.append(metrics[metric])
        h10_list.append(eval_h10(valid_result_dict))

        if metrics[metric] > best_valid_mrr:
            best_valid_mrr = metrics[metric]
            best_valid_idx = i
            best_step = step


    print("MRR")
    for mrr in mrr_list:
        print(mrr)

    print("H@10")
    for h10 in h10_list:
        print(h10)

    print("Best MRR at step {}".format(best_step))

    save_path = os.path.join(ensemble_path, os.path.basename(path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("copying best results to {}".format(save_path))
    for proc in range(num_proc):
        subprocess.run(["cp", os.path.join(path, "valid_score_{}_{}.pkl".format(proc, best_step)),
                              os.path.join(save_path, "valid_score_{}.pkl".format(proc))])

        subprocess.run(["cp", os.path.join(path, "test_score_{}_{}.pkl".format(proc, best_step)),
                              os.path.join(save_path, "test_score_{}.pkl".format(proc))])




