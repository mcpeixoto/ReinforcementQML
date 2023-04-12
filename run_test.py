from CardPole import CardPole
from utils import NestablePool
import multiprocessing as mp
import numpy as np



def worker(number):
    algorithm = CardPole(show_game=False, is_classical=False, seed=number)
    algorithm.train()


if __name__ == '__main__':

    n_runs = 10

    # Create a pool of workers
    pool = NestablePool(mp.cpu_count())
   
    # Run the workers
    pool.map(worker, range(n_runs))

    # Close the pool
    pool.close()

    # Wait for the processes to finish
    pool.join()

    print("Done")