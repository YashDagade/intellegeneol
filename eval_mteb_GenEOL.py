import argparse
import os
from functools import partial

from mteb import MTEB
import torch
from importlib import reload

import time
from peft import PeftModel, PeftConfig

import geneol
reload(geneol)

from geneol import GenEOL

from accelerate import Accelerator, InitProcessGroupKwargs
from datetime import timedelta
import sys
from transformers import AutoTokenizer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import random

import torch
import json

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default="mistralai/Mistral-7B-Instruct-v0.1", type=str)
    parser.add_argument('--gen_model_name_or_path', default=None, type=str)
    parser.add_argument('--method', default="s5", type=str)
    parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
    parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_length', default=1024, type=int)
    parser.add_argument('--dtype', default='bfloat16', type=str)
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--overwrite_results', action='store_true')
    parser.add_argument('--embedding_head', default=None, type=str)
    parser.add_argument('--pooling_method', default='mean', type=str)
    parser.add_argument('--seed', default=42, type=int)
    # used only for transfer tasks
    parser.add_argument('--quick_mode', action='store_true', default=False)
    # used only for STSB
    parser.add_argument('--dev_mode', action='store_true', default=False)
    # gen only removes any need for model.




    parser.add_argument('--suffix', required=True, type=str)    
    parser.add_argument('--EOL', action='store_true', default=False)   
    parser.add_argument('--normalized', action='store_true', default=False)   
    parser.add_argument('--num_gens', default=1, type=int)


    parser.add_argument('--clustering_threshold', default=0.96, type=float)
    parser.add_argument('--gen_only', action='store_true', default=False)
    parser.add_argument('--penultimate_layer', default=-1, type=int)
    parser.add_argument('--compositional', action='store_true', default=False)
    parser.add_argument('--tsep', action='store_true', default=False)
    parser.add_argument('--clus', action='store_true', default=False)
    parser.add_argument('--reeval', action='store_true', default=False)
    parser.add_argument('--select8', action='store_true', default=False)
    parser.add_argument('--select', default=-1, type=int)

    parser.add_argument('--IB', default='e', type=str)
    
    # Add conv_div flag for Conversational Diversity approach
    parser.add_argument('--conv_div', action='store_true', default=False, 
                        help='Use Conversational Diversity approach with hierarchical clustering')


    args = parser.parse_args()

    # if args.gen_model_name_or_path == '-':
    #     args.string_arg = None

    return args

if __name__ == '__main__':
    args = get_args()



    # Set the seed for Python's built-in random module
    random.seed(args.seed)

    # Set the seed for NumPy
    np.random.seed(args.seed)

    # Set the seed for TensorFlow
    torch.manual_seed(args.seed)



    output_folder = args.output_folder if args.output_folder else f"./resultsTF/{args.suffix}_{args.method}_{args.gen_model_name_or_path.split('/')[-1]}_{args.batch_size}_{args.num_gens}/{args.model_name_or_path.split('/')[-1]}"
        
    if (not args.reeval) and  (args.task_names is not None) and (len(args.task_names.split(",")) == 1) and os.path.exists(f"{output_folder}/{args.task_names.split(',')[0]}.json"):
        print(f"Skipping {args.task_names.split(',')[0]}")
        exit()
    

    setattr(args, "output_folder", output_folder)
    setattr(args, "torch_dtype", torch.bfloat16)

    assert args.pooling_method == "mean"

    
    init_proc = InitProcessGroupKwargs(timeout=timedelta(seconds=1500001))
    accelerator = Accelerator(kwargs_handlers=[init_proc])
    model = GenEOL(accelerator=accelerator, args=args)



    kwargs = {"task_langs": ['en']}
    kwargs["tasks"] = args.task_names.split(",")
    tasks = kwargs["tasks"]

    
    # tasks = [(t.metadata.name, t.metadata.type) for t in MTEB(**kwargs).tasks]
    

    if args.max_length is not None:
        model.encode = partial(model.encode, max_length=args.max_length)

    # print(tasks, flush=True)
    for task_name in tasks:
        args.task = task_name
        st = time.time()
        if task_name in ['MSMARCOv2', 'BigPatentClustering']:
            print('Skipping task: ' + task_name)
            continue
            #! need to modify this to account for dual instruction type
        setattr(args, "task", task_name)
        model.encode = partial(model.encode, args=args)
        eval_splits = ["test" if task_name not in ['MSMARCO'] else 'dev']

        if(args.dev_mode):
            # task_name="STSBenchmark"
            eval_splits=["validation"]
        # import ipdb; ipdb.set_trace()
        evaluation = MTEB(tasks=[task_name], task_langs=['en'])
        
        # import ipdb; ipdb.set_trace()

        evaluation.run(model, output_folder=output_folder, eval_splits=eval_splits, overwrite_results=args.reeval)
        print(task_name, flush=True)
        et = time.time()
        print("Total time (hours):", (et-st)/3600)

