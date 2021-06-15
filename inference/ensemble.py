import os
import sys
import numpy as np
import ipdb
import torch
import pickle
import dgl.backend as F
from tqdm import tqdm
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator


def main():
    data_path = sys.argv[1]
    submission_path = sys.argv[2]

    with open(os.path.join(data_path, "wikikg90m_kddcup2021/processed/val_t_correct_index.npy"), 'rb') as f:
        t_correct_index = torch.tensor(np.load(f))

    root_path = os.path.join(submission_path, "single_models")
    base_model_dirs = [_ for _ in os.listdir(root_path) if not os.path.isfile(_)]

    print("ensembling the following {} models ...".format(len(base_model_dirs)))
    for dir in base_model_dirs:
        print(dir)

    base_model_dirs = [os.path.join(root_path, _) for _ in base_model_dirs]

    print("loading models ...")
    valid_scores = []
    test_scores = []
    for path in tqdm(base_model_dirs):
        if os.path.isfile(path):
            continue
        valid_score_files = []
        test_score_files = []
        for i in range(4):
            with open(os.path.join(path, 'valid_score_{}.pkl'.format(i)), 'rb') as f:
                valid_score_files.append(torch.load(f))
            with open(os.path.join(path, 'test_score_{}.pkl'.format(i)), 'rb') as f:
                test_score_files.append(torch.load(f))

        valid_score = torch.cat(valid_score_files, 0)
        test_score = torch.cat(test_score_files, 0)
        valid_scores.append(valid_score)
        test_scores.append(test_score)

    print("ensembling ...")
    final_valid_score = torch.stack(valid_scores, 0).mean(0)
    final_test_score = torch.stack(test_scores, 0).mean(0)

    print("generating t_pred_top10 ...")
    valid_argsort = F.argsort(final_valid_score, dim=1, descending=True)
    valid_t_pred_top10 = valid_argsort[:, :10]

    test_argsort = F.argsort(final_test_score, dim=1, descending=True)
    test_t_pred_top10 = test_argsort[:, :10]

    valid_dict = {}
    valid_dict['h,r->t'] = {'t_correct_index': t_correct_index,
                            't_pred_top10': valid_t_pred_top10}

    test_dict = {}
    test_dict['h,r->t'] = {'t_pred_top10': test_t_pred_top10}

    np.save(os.path.join(submission_path, "valid_t_pred_top10.npy"), valid_t_pred_top10.numpy().astype(np.int16))

    evaluator = WikiKG90MEvaluator()

    print("evaluating ensemble results ...")
    metrics = evaluator.eval(valid_dict)

    print("MRR: {}".format(metrics['mrr']))

    print("saving test submission ...")
    save_path = submission_path
    evaluator.save_test_submission(test_dict, save_path)

if __name__ == '__main__':
    main()