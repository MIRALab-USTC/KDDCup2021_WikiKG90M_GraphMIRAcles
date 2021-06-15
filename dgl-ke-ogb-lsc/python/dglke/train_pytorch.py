# -*- coding: utf-8 -*-
#
# train_pytorch.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.optim as optim
import torch as th

from distutils.version import LooseVersion
TH_VERSION = LooseVersion(th.__version__)
if TH_VERSION.version[0] == 1 and TH_VERSION.version[1] < 2:
    raise Exception("DGL-ke has to work with Pytorch version >= 1.2")
from .models.pytorch.tensor_models import thread_wrapped_func
from .models import KEModel
from .utils import save_model, get_compatible_batch_size

import os
import logging
import time
import ipdb
from functools import wraps

import dgl
from dgl.contrib import KVClient
import dgl.backend as F

from .dataloader import EvalDataset
from .dataloader import get_dataset
import pdb
from collections import defaultdict
from ogb.lsc import WikiKG90MDataset, WikiKG90MEvaluator
from tqdm import tqdm
import pickle
from math import ceil

def load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel,
                    ent_feat_dim=ent_feat_dim, rel_feat_dim=rel_feat_dim)
    if ckpt is not None:
        assert False, "We do not support loading model emb for genernal Embedding"
    return model

def load_model_from_checkpoint(args, n_entities, n_relations, ckpt_path, ent_feat_dim, rel_feat_dim):
    model = load_model(args, n_entities, n_relations, ent_feat_dim, rel_feat_dim)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None, client=None):
    logs = []
    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1
    if args.async_update:
        model.create_async_update()
    if args.strict_rel_part or args.soft_rel_part:
        model.prepare_relation(th.device('cuda:' + str(gpu_id)))
    if args.soft_rel_part:
        model.prepare_cross_rels(cross_rels)
    
    if args.encoder_model_name not in ['shallow']:
        model.transform_net = model.transform_net.to(th.device('cuda:' + str(gpu_id)))
        if args.model_name == 'ComplExPriori':
            model.score_func = model.score_func.to(th.device('cuda:' + str(gpu_id)))

        # default
        params = list(model.transform_net.parameters()) + list(model.score_func.parameters())
        optimizer = th.optim.Adam(params, args.mlp_lr, weight_decay=args.weight_decay)

    else:
        optimizer = None
    train_start = start = time.time()
    sample_time = 0
    update_time = 0
    forward_time = 0
    backward_time = 0
    
    for step in range(0, args.max_step):
        # print("step: {}".format(step))
        start1 = time.time()
        pos_g, neg_g = next(train_sampler)
        sample_time += time.time() - start1

        if client is not None:
            model.pull_model(client, pos_g, neg_g)

        start1 = time.time()
        if optimizer is not None:
            optimizer.zero_grad()
        loss, log = model.forward(pos_g, neg_g, gpu_id)
        forward_time += time.time() - start1

        start1 = time.time()
        loss.backward()
        backward_time += time.time() - start1

        start1 = time.time()
        if client is not None:
            model.push_gradient(client)
        else:
            model.update(gpu_id)
        if optimizer is not None:
            optimizer.step()
            # scheduler.step()

        update_time += time.time() - start1
        logs.append(log)

        # force synchronize embedding across processes every X steps
        if args.force_sync_interval > 0 and \
            (step + 1) % args.force_sync_interval == 0:
            if barrier is not None:
                barrier.wait()

        if (step + 1) % args.log_interval == 0:
            if (client is not None) and (client.get_machine_id() != 0):
                pass
            else:
                for k in logs[0].keys():
                    v = sum(l[k] for l in logs) / len(logs)
                    logging.info('[proc {}][Train]({}/{}) average {}: {}'.format(rank, (step + 1), args.max_step, k, v))
                logs = []
                logging.info('[proc {}][Train] {} steps take {:.3f} seconds'.format(rank, args.log_interval,
                                                                time.time() - start))
                logging.info('[proc {}]sample: {:.3f}, forward: {:.3f}, backward: {:.3f}, update: {:.3f}'.format(
                    rank, sample_time, forward_time, backward_time, update_time))

                # logging.info('[proc {}][Train]({}/{}) Parameter w_e: {}'.format(
                #     rank, (step + 1), args.max_step, model.transform_net.w_e)
                # )
                # logging.info('[proc {}][Train]({}/{}) Parameter w_r: {}'.format(
                #     rank, (step + 1), args.max_step, model.transform_net.w_r)
                # )

                sample_time = 0
                update_time = 0
                forward_time = 0
                backward_time = 0
                start = time.time()

        # if True:
        if args.valid and (step + 1) % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            valid_start = time.time()
            if args.strict_rel_part or args.soft_rel_part:
                model.writeback_relation(rank, rel_parts)
            # forced sync for validation
            if barrier is not None:
                barrier.wait()
            logging.info('[proc {}]barrier wait in validation take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            valid_start = time.time()
            if valid_samplers is not None:
                valid_input_dict, valid_scores = test(args, model, valid_samplers, step, rank, mode='Valid')
                th.save(valid_input_dict, os.path.join(args.save_path, "valid_{}_{}.pkl".format(rank, step)))
                th.save(valid_scores, os.path.join(args.save_path, "valid_score_{}_{}.pkl".format(rank, step)))

            if test_samplers is not None:
                test_input_dict, test_scores = test(args, model, test_samplers, step, rank, mode='Test')
                th.save(test_input_dict, os.path.join(args.save_path, "test_{}_{}.pkl".format(rank, step)))
                th.save(test_scores, os.path.join(args.save_path, "test_score_{}_{}.pkl".format(rank, step)))

            logging.info('[proc {}]validation and test take {:.3f} seconds:'.format(rank, time.time() - valid_start))
            if args.soft_rel_part:
                model.prepare_cross_rels(cross_rels)
            if barrier is not None:
                barrier.wait()
    logging.info('get out optimization')
    print('proc {} takes {:.3f} seconds'.format(rank, time.time() - train_start))
    time.sleep(10)
    if rank == 0 and not args.no_save_emb:
        save_model(args, model, None, None)
        print('proc {} model saved'.format(rank))

    if barrier is not None:
        barrier.wait()
    print('proc {} after barrier'.format(rank))
    if args.async_update:
        model.finish_async_update()
    print('proc {} finish async update'.format(rank))
    if args.strict_rel_part or args.soft_rel_part:
        model.writeback_relation(rank, rel_parts)
    print('proc {} return'.format(rank))


def test(args, model, test_samplers, step, rank=0, mode='Test'):
    if len(args.gpu) > 0:
        gpu_id = args.gpu[rank % len(args.gpu)] if args.mix_cpu_gpu and args.num_proc > 1 else args.gpu[0]
    else:
        gpu_id = -1

    if args.strict_rel_part or args.soft_rel_part:
        model.load_relation(th.device('cuda:' + str(gpu_id)))

    test_scores = []
    with th.no_grad():
        logs = defaultdict(list)
        answers = defaultdict(list)
        for sampler in test_samplers:
            print(sampler.num_edges, sampler.batch_size)
            for query, ans, candidate in tqdm(sampler, disable=not args.print_on_screen, total=ceil(sampler.num_edges/sampler.batch_size)):
                log, score = model.forward_test_wikikg(query, ans, candidate, sampler.mode, gpu_id)
                logs[sampler.mode].append(log)
                answers[sampler.mode].append(ans)
                test_scores.append(score)
        print("[{}] finished {} forward".format(rank, mode))

        input_dict = {}
        assert len(answers) == 1
        assert 'h,r->t' in answers
        if 'h,r->t' in answers:
            assert 'h,r->t' in logs, "h,r->t not in logs"
            input_dict['h,r->t'] = {'t_correct_index': th.cat(answers['h,r->t'], 0), 't_pred_top10': th.cat(logs['h,r->t'], 0)}
        # if 't,r->h' in answers:
        #     assert 't,r->h' in logs, "t,r->h not in logs"
        #     input_dict['t,r->h'] = {'h_correct_index': th.cat(answers['t,r->h'], 0), 'h_pred_top10': th.cat(logs['t,r->h'], 0)}
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()
    # test_samplers[0] = test_samplers[0].reset()
    # test_samplers[1] = test_samplers[1].reset()

    return input_dict, th.cat(test_scores, 0)

@thread_wrapped_func
def train_mp(args, model, train_sampler, valid_samplers=None, test_samplers=None, rank=0, rel_parts=None, cross_rels=None, barrier=None):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    train(args, model, train_sampler, valid_samplers, test_samplers, rank, rel_parts, cross_rels, barrier)

@thread_wrapped_func
def test_mp(args, model, test_samplers, rank=0, mode='Test'):
    if args.num_proc > 1:
        th.set_num_threads(args.num_thread)
    test(args, model, test_samplers, 0, rank, mode)
