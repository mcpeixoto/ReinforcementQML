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


def worker(seed):
    os.system(f"python CardPole_classical.py --type train --seed {seed}")
    os.system(f"python CardPole_classical.py --type benchmark --seed {seed}")
    return

if __name__ == '__main__':

    seeds = list(range(1, 21))
    n_processes = 50

    print(f"[INFO] Starting grid search | TOTAL RUNS: {len(seeds)} | TOTAL PROCESSES: {n_processes}")

    with NestablePool(processes=n_processes) as pool:
        # Use tqdm
        for _ in tqdm(pool.imap_unordered(worker, seeds), total=len(seeds), desc="[INFO] Grid Search Progress", dynamic_ncols=True):
            pass





    print("[INFO] Grid search finished")


