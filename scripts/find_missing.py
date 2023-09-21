import os
import sys
sys.path.append('../src')

import torch.multiprocessing as mp
import multiprocessing as mp2
if __name__ == '__main__':
    mp.set_start_method('spawn')

import zarr
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from itertools import islice

from gen_samples_next3d import WS

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def consumer_proxy(rank, args):
    consumer_fn(rank, **args)
    return

def consumer_fn(rank, **args):
    _s = rank * args['chunk_size']
    _e = _s + args['chunk_size']

    for ii in range(_s, _e):
        img_exists = os.path.exists(os.path.join(args['outdir'], 'images', f'{str(ii).zfill(7)}.png'))
        np_exists = os.path.exists(os.path.join(args['outdir'], 'samples', f'{str(ii).zfill(7)}.npy'))

        if not img_exists or not np_exists:
            args['queue'].put((1, ii))
        if img_exists and not np_exists:
            args['queue'].put((2, ii))

    print(f'Consumer {rank} finished')
    return

if __name__ == '__main__':
    def main():
        try:
            queue = mp.Queue()

            args = {}
            args['queue'] = queue
            args['chunk_size'] = 25000
            args['outdir'] = '/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_cams_ids_0.7_500k_final'

            writers = []
            for rank in range(20):
                writers.append(mp.Process(target=consumer_proxy, args=(rank, args)))
                writers[-1].start()
            print('Writers started')

            print('Waiting for writers to finish')
            # for w in writers:
            #     print(w)
            #     w.join()
            mp2.connection.wait([w.sentinel for w in writers])
            print('writers joined')

            missing_either = []
            missing_np = []

            print('Collecting results')
            for _ in tqdm(range(queue.qsize())):
                _type, _idx = queue.get()
                if _type == 1:
                    missing_either.append(_idx)
                elif _type == 2:
                    missing_np.append(_idx)
                else:
                    raise ValueError('Unknown type')
                
            print(f'Missing either: {len(missing_either)}')
            print(f'Missing np: {len(missing_np)}')

            np.save(os.path.join(args['outdir'], 'missing_either.npy'), np.array(missing_either))
            np.save(os.path.join(args['outdir'], 'missing_np.npy'), np.array(missing_np))

        except Exception as e:
            print(e)
        
        finally:
            queue.close()

        

    main()