import os

import lightning as L
from lightning.pytorch.callbacks import Timer
import pandas as pd

from configs.icassp import hparams
from datasets.loader.datamodule import EhrDataModule
from datasets.loader.load_los_info import get_los_info
from pipelines import DlPipeline, MlPipeline
from utils.bootstrap import run_bootstrap

if __name__ == "__main__":
    best_hparams = hparams
    performance_table = {'dataset':[], 'task': [], 'model': [], 'fold': [], 'seed': [], 'accuracy': [], 'auroc': [], 'auprc': [], 'f1': [], 'minpse': []}
    for i in range(0, len(best_hparams)):
        print("#########p##################################")
        print("############      [START]         ############")
        print("###########################################")
        
        config = best_hparams[i]
        print(f"Testing... {i}/{len(best_hparams)}")
        print("            [[[", config["model"], config["dataset"], "]]]")

        seeds = [0]
        for seed in seeds:
            config["seed"] = seed
            # for seed in seeds:
            #     config["seed"] = seed
            outs = pd.read_pickle(f'analysis/icassp/{config["dataset"]}-{config["model"]}.pkl')

            y_pred_outcome = outs['preds']
            y_outcome_true = outs['labels'][:,0,:]

            metrics = run_bootstrap(y_pred_outcome, y_outcome_true)
            for k, v in metrics.items():
                mean_var = 100 * v['mean']
                std_var = 100 * v['std']
                meanstd = f"{mean_var:.2f}±{std_var:.2f}"
                print(f"{k:25}: {meanstd}")

        print("###########################################")
        print("############      [END]         ############")
        print("###########################################")
        print("")
        print("")