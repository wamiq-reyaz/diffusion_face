import numpy as np
import zarr
import tqdm

import os
import sys
sys.path.append('../src')
from gen_samples_next3d import WS

if __name__ == '__main__':
    idxes = np.load('/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_cams_ids_0.7_500k_final/missing_np.npy')

    store = zarr.LMDBStore('/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_cams_ids_0.7_500k_final/samples.lmdb',
                            readonly=True,
                            lock=False,)
    group = zarr.group(store=store, overwrite=False)

    for ii in tqdm.tqdm(idxes):
        data = np.zeros((len(WS), 512), dtype=np.float32)
        for jj, n in enumerate(WS):
            data[jj, :] = group[n]['data'][ii, :]

        path_to_save = f'/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_cams_ids_0.7_500k_final/samples/{str(ii).zfill(7)}.npy'
        assert not os.path.exists(path_to_save)
        np.save(f'/ibex/ai/home/parawr/Projects/diffusion/data/w_plus_img_cams_ids_0.7_500k_final/samples/{str(ii).zfill(7)}.npy', data)