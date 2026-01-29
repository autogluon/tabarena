from autogluon.features import TargetAwareFeatureCompressionFeatureGenerator, RandomSubsetTAFC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openml
import seaborn as sns
from sklearn.metrics import roc_auc_score, root_mean_squared_error, log_loss
import os
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
from tabprep.utils.modeling_utils import adjust_target_format
import openml
import time
from autogluon.features import AutoMLPipelineFeatureGenerator

import pickle

from tabarena.icml2026.helpers import run_experiment

data_tid_map = { # https://www.openml.org/search?type=study&study_type=task&id=379&sort=tasks_included
    'balance-scale': 11,
    'jungle_chess_2pcs_raw_endgame_complete_classification': 167119,
    'vehicle': 53,
    'artificial-characters': 14964,
    'electricity': 219,
    'medical_charges': 361294,
    'visualizing_soil': 361094,
    'Bike_sharing_demand': 360085,
    'mfeat-zernike': 22,
    'road-safety': 361285,
    'SpeedDating': 146607,

    'elevators': 3711,
    'ada_agnostic': 3896,
    'phoneme': 9952,
    'nomao': 9977,
}

if __name__ == "__main__":
    max_new_feats = 1500
    save_path = "tabarena/tabarena/tabarena/icml2026/results"
    exp_name = "simulated_highfrequency_test" 
    # exp_name = "simulated_highfrequency_final_noround" 
    
    subsample = None
    rerun = False
    verbosity = 0
    num_bag_folds = 8

    prep_types = [None, "RSTAFC-noround-1order", "RSTAFC-noround"]

    for dataset_name in [
        # 'ada_agnostic', # Experimental, unsure whether it helps
        # 'artificial-characters',
        'electricity',
        ]:

        if os.path.exists(os.path.join(save_path, f'{exp_name}_{dataset_name}_results.pkl')) and not rerun:
            print(f" Skipping {dataset_name} as results already exist.")
            continue

        tid = data_tid_map[dataset_name]
        task = OpenMLTaskWrapper(openml.tasks.get_task(tid))

        fold = 0
        repeat = 0
        sample = None

        X, y, X_test, y_test = task.get_train_test_split(fold=fold, repeat=repeat)
        target_type = task.problem_type


        y = adjust_target_format(y, target_type)
        y_test = adjust_target_format(y_test, target_type)

        prep = AutoMLPipelineFeatureGenerator()
        X = prep.fit_transform(X, y)
        X_test = prep.transform(X_test)

        if subsample is not None and subsample < X.shape[0]:
            X = X.sample(n=subsample, random_state=42)
            y = y.loc[X.index]

        results = {"preds": {}, "performance": {}}
        for model_name in ["LR", "TABM", "PFN", "GBM", "CAT"]:
            results["preds"][model_name] = {}
            results["performance"][model_name] = {}
            print("--"*20)
            order2_max_feats_reached = False
            for prep_type in prep_types:
                if order2_max_feats_reached and prep_type == "3-ARRITHMETIC":
                    print(" Skipping 3-ARRITHMETIC as max new feats reached for 2-ARRITHMETIC")
                    continue
                preds, score, X_used = run_experiment(X, y, X_test, y_test, model_name, prep_type, target_type, verbosity=verbosity, num_bag_folds=num_bag_folds)
                
                if X_used.shape[1] >= 1500:
                    order2_max_feats_reached = True

                results["performance"][model_name][prep_type] = score
                results["preds"][model_name][prep_type] = preds
                print(f"Dataset: {dataset_name}, Model: {model_name}, Prep: {prep_type} (shape=[{X_used.shape}]), Performance: {score:.4f}")


        with open(os.path.join(save_path, f'{exp_name}_{dataset_name}_results.pkl'), 'wb') as f:
            pickle.dump(results, f)