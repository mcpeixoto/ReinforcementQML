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


def worker(HP):
    #algorithm = CardPole(seed=seed, reuploading=reuploading, cx=cx, ladder=ladder, n_layers=n_layers)
    
    # Call python cardpole.py with arguments
    #print("Command:" f"python CardPole.py --seed {HP['seed']} --reuploading {HP['reuploading']} --cx {HP['cx']} --ladder {HP['ladder']} --n_layers {HP['n_layers']}")
    os.system(f"python CardPole.py --seed {HP['seed']} --reuploading {HP['reuploading']} --cx {HP['cx']} --ladder {HP['ladder']} --n_layers {HP['n_layers']}")
    #print("-----"*5)
    return

if __name__ == '__main__':
    searchgrid = {
        "reuploading" : [1, 0],
        "cx" : [1, 0],
        "ladder" : [1, 0],
        "n_layers" : list(range(1, 6)),
        "seed": list(range(1, 11)),
    }

    gridsearch = GridSearch(searchgrid)
    n_processes = 50

    print(f"[INFO] Starting grid search | TOTAL RUNS: {len(gridsearch)} | TOTAL PROCESSES: {n_processes}")

    pool = NestablePool(processes=n_processes)
    
    jobs = []
    for hp in gridsearch:
        jobs.append(pool.apply_async(worker, (hp,)))

    pool.close()

    for job in tqdm(jobs, total=len(jobs), desc="[INFO] Grid search progress"):
        job.get()

    pool.join()

    print("[INFO] Grid search finished")


