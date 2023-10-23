import os
import sys
sys.path.append('../src')
from gen_samples_next3d import WS

import zarr
import numpy as np
from tqdm import tqdm
from itertools import islice
from concurrent.futures import ThreadPoolExecutor

CHUNK_SIZE = 1024
N_ROWS = 73
def create_one_sample(indices, group):
    data = np.zeros((len(indices), N_ROWS, 512), dtype=np.float32)
    for ii, n in enumerate(WS):
        data[:, ii, :] = group[n]['data'][indices, :]

    return data

def task(args):
    store, global_start_idx, outdir = args
    group = zarr.group(store=store, overwrite=False)
    indices = list(range(global_start_idx, global_start_idx + CHUNK_SIZE))
    sample = create_one_sample(indices, group)

    for ii, idx in enumerate(indices):
        np.save(os.path.join(outdir, f'{str(idx).zfill(7)}.npy'), sample[ii, :, :])
    
    return True


if __name__ == '__main__':
    # store = zarr.storage.NestedDirectoryStore('/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_150k_frontal_id_0.9_28/samples.zarr')
    # outdir = '/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_150k_frontal_id_0.9_28/samples2'

    store = zarr.LMDBStore('/ibex/project/c2241/data/diffusion/w_plus_img_cams_ids_0.7_2m_largefov_largestd_final/samples.lmdb',
                        readonly=True, lock=False,)
    outdir = '/ibex/project/c2241/data/diffusion/w_plus_img_cams_ids_0.7_2m_largefov_largestd_final/samples'


    os.makedirs(outdir, exist_ok=True)

    g = zarr.group(store=store, overwrite=False)
    num_samples = g[WS[0]]['data'].shape[0]
    print(f'num_samples: {num_samples}')


    with ThreadPoolExecutor(max_workers=10) as executor:
        global_start_indices = list(range(0, num_samples, CHUNK_SIZE))
        args = [(store, g, outdir) for g in global_start_indices]
        _ = list(tqdm(executor.map(task, args), total=len(args)))

    print('Done!')

    

    