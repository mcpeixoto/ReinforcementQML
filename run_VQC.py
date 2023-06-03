import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import multiprocessing as mp
import os
from CardPole import CardPole
from utils import NestablePool, GridSearch
from tqdm import tqdm
import sys
import time 

# Ignore warnings 
import warnings
warnings.filterwarnings("ignore")


from config import searchgrid


def worker(HP):
    os.system(f"python CardPole.py --type train --seed {HP['seed']} --reuploading {HP['reuploading']} --cx {HP['cx']} --ladder {HP['ladder']} --n_layers {HP['n_layers']}")
    os.system(f"python CardPole.py --type benchmark --seed {HP['seed']} --reuploading {HP['reuploading']} --cx {HP['cx']} --ladder {HP['ladder']} --n_layers {HP['n_layers']}")
    return

if __name__ == '__main__':

    gridsearch = GridSearch(searchgrid)
    n_processes = 50

    print(f"[INFO] Starting grid search | TOTAL RUNS: {len(gridsearch)} | TOTAL PROCESSES: {n_processes}")

    with NestablePool(processes=n_processes) as pool:
        # Use tqdm
        for _ in tqdm(pool.imap_unordered(worker, gridsearch), total=len(gridsearch), desc="[INFO] Grid Search Progress", dynamic_ncols=True):
            pass





    print("[INFO] Grid search finished")


