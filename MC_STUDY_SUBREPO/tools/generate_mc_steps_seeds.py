import os
import numpy as np

def generate_mc_steps_seeds(seeds_no):
    seeds = []
    while len(np.unique(seeds)) < seeds_no:
        seeds.append(int.from_bytes(os.urandom(5), 'big'))

    return seeds
