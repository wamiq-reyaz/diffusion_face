import os
import sys
sys.path.append('../src')

import torch.multiprocessing as mp
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

def consumer_fn(rank, **args):
    try:
        while True:
            data = args['queue'].get()
            if data is None:
                break

            # save data
            for ii, val in enumerate(data['idxes']):
                np.save(os.path.join(args['outdir'], f'{str(val).zfill(7)}.npy'), data['data'][ii, :, :])
    except Exception as e:
        print(e)
    finally:
        print(f'Consumer {rank} finished')
        return

def producer_proxy(rank, args):
    producer_fn(rank, **args)

def producer_fn(rank, **args):
    group = zarr.group(store=args['store'], overwrite=False)

    num_keys = len(WS)
    num_samples = group[WS[0]]['data'].shape[0]
    samples_per_rank = num_samples // args['world_size']
    _s = rank * samples_per_rank
    _e = _s + samples_per_rank

    if rank == 0:
        _iterator = tqdm(batched(range(_s, _e), args['chunk_size']), total=samples_per_rank//args['chunk_size'])
    else:
        _iterator = batched(range(_s, _e), args['chunk_size'])

    data = np.zeros((args['chunk_size'], num_keys, 512), dtype=np.float32)
    for _chunk in _iterator:
        for ii, n in enumerate(WS):
            idxer = list(_chunk)
            local_idxer = list(range(len(idxer)))
            data[local_idxer, ii, :] = group[n]['data'][idxer, :]

        # put data on the queue
        args['queue'].put({'data':data,
                        'idxes':list(_chunk)})
            
    print(f'Producer {rank} finished')

if __name__ == '__main__':
    def main():
        try:
            store = zarr.LMDBStore('/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_nocams_ids_0.7_150k/samples.lmdb',
                                    readonly=True,
                                    lock=False,)
            queue = mp.Queue()

            args = {}
            args['store'] = store
            args['queue'] = queue
            args['chunk_size'] = 1024
            args['world_size'] = 10
            args['outdir'] = '/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_nocams_ids_0.7_150k/w_plus_img_nocams_ids_0.7_150k/samples'
            os.makedirs(args['outdir'], exist_ok=True)

            writers = []
            for rank in range(8):
                writers.append(mp.Process(target=consumer_proxy, args=(rank, args)))
                writers[-1].start()
            print('Writers started')

            mp.spawn(producer_proxy, args=(args,), nprocs=args['world_size'], join=True)

            print('Producers finished')
            for _ in range(100):
                queue.put(None)

            for w in writers:
                w.join()
            print('writers joined')

            queue.close()
        
        finally:
            queue.close()

        

    main()