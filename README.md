# Code of Team GraphMIRAcles in the WikiKG90M-LSC track of KDD Cup 2021
This is the code of Team GraphMIRAcles in the [WikiKG90M-LSC](https://ogb.stanford.edu/kddcup2021/wikikg90m/) track of OGB-LSC @ KDD Cup 2021.

Team Members: Jianyu Cai, Jiajun Chen, Taoxing Pan, Zhanqiu Zhang, Jie Wang.


## Installation requirements
```
ogb>=1.3.0
torch>=1.7.0
dgl==0.4.3
```
In addition, please install the dgl-ke-ogb-lsc by `cd dgl-ke-ogb-lsc/python` and `pip install -e .`

## Performance

| Model              |Valid MRR  | Test MRR|
|:------------------ |:-------------- |:-------------- |
| ComplEx-CMRC (single model)   | 0.926 | - |
| Final submission (ensemble) | 0.978 | 0.9707 |

## Dataset
Please refer to the [official website](https://ogb.stanford.edu/kddcup2021/wikikg90m/) for the information of dataset.

## Reproduce the Results

### Our Infrastructure
Our model is developed on machines with

| Machine Usage | CPU  | Memory | GPU |
|:------------------ |:-------------- |:-------------- |:-------------- |
| For Training  | 2 Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz | 1TB | 8 NVIDIA GEFORCE GTX 2080TI |
| For Rule Mining | 4 Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz | 1TB | - |

### Memory & Time Usage

Training one basic model requires
- **Memory Usage**: Around 300GB memory and 10GB GPU memory.
- **Time**: Around 10.9h 
    - **Training**: around 20s for every 1,000 steps (around 2.6h in total)
    - **Validation/Testing**: around 25min for validation/testing (around 8.3h in total)

Running AMIE 3 to mine rules requires
- **Memory Usage**: Around 400GB memory
- **Time**: Around 50h

### Step 1. Train Basic Models
Use the following script to train 5 basic models with different random seeds.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --data_path /your/data/path/ \
--encoder_model_name concat_mlp_residual_new --inverse_relation \
--instance_id instance_00 --reg None --regularization_coef 0.0 \
--model_name ComplEx --batch_size_eval 100 \
--batch_size 800 --neg_sample_size 100 --num_neg_chunk 1 \
--hidden_dim 300 --feat_dim 300 --gamma 3 --lr 0.1  \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --eval_interval 25000 --max_step 500000 \
--async_update --force_sync_interval 25000 --no_save_emb \
--print_on_screen --save_path /your/save/path/
```
Notice: remember to replace `--data_path` and `--save_path` to your paths.

The random seed is obtained based on the timestamp of training,
please refer to `train.py`.

```python
args.seed = int(time.time()) % 4096
```


### Step 2. Mine Rules

We use [AIME 3](https://github.com/lajus/amie) to mine rules from the knowledge graph. For computational efficiency, we sample five subgraphs from the whole graph and only mine rules  of length no longer than 3.  The sampled subgraphs are as follows, where we use `train_hrt` to represent the NumPy array of the training triples.

| Subgraph ID              | Sampled Triples  | Number of Samples|Number of generated rules|
|:-----------------: |:-------------: |:-------------: |:----------: |
| 0 | train_hrt[0: 200000000] | 200,000,000 | 7179 |
| 1 | train_hrt[200000000: 400000000] | 200,000,000 | 4981 |
| 2 | train_hrt[400000000:], train_hrt[:100000000]| 201,160,482 | 8026 |
| 3 | train_hrt[100000000: 300000000] | 200,000,000 | 3903 |
| 4 | train_hrt[300000000:] | 201,160,482 | 5999 |

After getting the rules from the five subgraphs, we merge all the rules and finally get 11716 rules. 
The code of mining rules is in `rule_miner/rule_miner.ipynb`.


### Step 3. Build Inference Model to Predict Test Labels

#### Rule-based Data Augmentation

We filter the rules in the KG by their confidence, and use high-confident rules to generate new unseen triples. 
The code is in `inference/Rule-based_Data_Augmentation.ipynb`.  

We keep the rules with confidence greater than 0.95 or 0.99. 
For the confidence greater than 0.95, we have 2062 rules and about 100 million newly generated triples. 
For the confidence greater than 0.99, there are 1464 rules and about 200 million newly generated triples. 
We add those triples to the train data, and use the following script to finetune the basic model.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 dglke_train --data_path /your/new_data/path/ \
--encoder_model_name concat_mlp_residual_new --inverse_relation \
--instance_id instance_00 --reg None --regularization_coef 0.0 \
--model_name ComplEx --batch_size_eval 100 \
--batch_size 1600 --neg_sample_size 1600 --num_neg_chunk 1 \
--hidden_dim 200 --feat_dim 200 --gamma 3 --lr 0.05  \
--valid --test -adv --mix_cpu_gpu --num_proc 4 --num_thread 4 \
--gpu 0 1 2 3 --eval_interval 25000 --max_step 500000 \
--async_update --force_sync_interval 25000 --no_save_emb \
--print_on_screen --save_path /your/save/path/
```

Notice: remember to replace `--data_path` and `--save_path` to your paths.

#### Ensemble

Using the above scripts, we train 5 basic models for each group of generated triplets (100 and 200 million).
We evaluate the single models with `inference/evaluate_single.py` by

```python
python evaluate_single.py $SAVE_PATH $NUM_PROC $ENSEMBLE_PATH
```

Then, we apply average bagging to ensemble the 15 single models (5 models on vanilla train dataset, 10 models on the
augmented dataset) with `inference/ensemble.py` by

```python
python ensemble.py $DATA_PATH $ENSEMBLE_PATH
```

#### Knowledge Distillation
We then use knowledge distillation to distill the superior performance of the ensemble model to a single model.
That is, we let single models to learn the output of the ensemble model. After obtaining the newly distilled single 
models, we repeat the Ensemble -> Knowledge Distillation process for two more times. 


### Acknowledgement
Our implementation is based on [ogb](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/wikikg90m).
Thanks for their contributions.