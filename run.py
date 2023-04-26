import multiprocessing
import os
import sys
curr_dir = os.getcwd()

# Now asynchronusly
pool = multiprocessing.Pool(processes=9)
for i in range(9):
    pool.apply_async(os.system, args=(f'python CardPole.py {i}',))
pool.close()
pool.join()


