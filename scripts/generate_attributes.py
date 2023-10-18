import os
import sys
import torch.multiprocessing as mp
import multiprocessing as pmp # python multiprocessing
import torch
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import facer
import tqdm

import logging
import time

# ------------------------------------------------------------------------------
def proxy(rank:int,
        args: dict) -> None:
    kwargs = args
    save_worker(rank, **kwargs)
    return None

def save_worker(rank: int,
                queue: mp.Queue ) -> None:
    try: 
        while True:
            data = queue.get()
            if data is None:
                break
            else:
                seg_map = data['seg_map']
                attrs = data['attrs']
                _dir = data['dir']
                idx = data['idx']

                for ii, n in enumerate(idx):
                    n = int(n.item())
                    seg = seg_map[ii]
                    attr = attrs[ii]
                    
                    seg = Image.fromarray(seg.astype(np.uint8))
                    # seg = seg.resize((512, 512), Image.NEAREST) # the segmentation is done on 448 sized images
                    seg.save(os.path.join(_dir, 'seg', str(n).zfill(7)+'.png'))
                    np.save(os.path.join(_dir, 'attr', str(n).zfill(7)+'.npy'), attr)
    except Exception as e:
        print(e, rank)
    finally:
        print(f"Worker {rank} exiting")
        return None

def get_highest_confidence_face(faces, bs=8):
    # only retain the highest scoring face
    mapping = dict()
    
    for r, p, s, i in zip(faces['rects'], faces['points'], faces['scores'], faces['image_ids']):
        i = int(i.item())
        if i not in mapping:
            mapping[i] = {'rects': r, 'points': p, 'scores': s}
        else:
            prev_score = mapping[i]['scores']
            if s > prev_score:
                mapping[i] = {'rects': r, 'points': p, 'scores': s}
            
    _ret = reconstruct_faces_from_dict(mapping, bs=bs)
    return _ret

def reconstruct_faces_from_dict(_dict, bs=8):
    rects = []
    points = []
    scores = []
    image_ids = []

    valid = []
    for ii in range(bs):
        if ii not in _dict:
            continue
        rects.append(_dict[ii]['rects'])
        points.append(_dict[ii]['points'])
        scores.append(_dict[ii]['scores'])
        image_ids.append(ii)
        valid.append(ii)


    return ({
            'rects': torch.stack(rects, dim=0),
            'points': torch.stack(points, dim=0),
            'scores': torch.tensor(scores),
            'image_ids': torch.tensor(image_ids),
            }, 
            valid
            )
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    mp.set_start_method('spawn')
    from facer.face_parsing.farl import pretrain_settings

    # ------------------------------------------------------------------------------
    log_fname = __name__ + time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(filename=log_fname, filemode='w', level=logging.WARNING)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_fname)
    fh.setLevel(logging.DEBUG)

    # Add a simple formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(fh)
    # ------------------------------------------------------------------------------

    class ALL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            device = 'cuda'

            face_detector = facer.face_detector("retinaface/resnet50", device=device, threshold=0.5)

            for k, v in face_detector.named_parameters():
                v.requires_grad_(False)

            face_parser = facer.face_parser('farl/celebm/448', device=device)

            for k, v in face_parser.named_parameters():
                v.requires_grad_(False)

            face_parser.conf_name = 'lapa/448'

            label_names = pretrain_settings['celebm/448']['label_names']

            face_attr = facer.face_attr("farl/celeba/224", device=device)

            for k, v in face_attr.named_parameters():
                v.requires_grad_(False)


            self.face_detector = face_detector
            self.face_parser = face_parser
            self.face_attr = face_attr

        def forward(self, x):
            # print(x.shape)
            x = x * 255. # the image is normalized within the model
            faces = self.face_detector(x)
            # only retain the highest scoring face
            filtered_faces, valid = get_highest_confidence_face(faces, bs=x.shape[0])

            # reconstruct batch to only include valid faces
            x = x[valid]
            logger.warning(f'valid {valid}')


            val = self.face_parser(x, filtered_faces)
            val['faces'] = faces
            val['attributes'] = self.face_attr(x, filtered_faces)
            seg_logits = val['seg']['logits']
            seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
            seg_map = seg_probs.argmax(dim=1)
            val['seg_map'] = seg_map
            val['valid_idx'] = valid
            return val
        
    model = ALL()
    model = model.cuda()
    model = model.eval()
    # model = torch.nn.DataParallel(model)
    
    sys.path.append('./src')
    from training_diffusion.datasets.latent import WData

    d = WData(
        cfg=None,
        w_path='/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_cams_ids_0.7_2m_final/samples',
        img_path='/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_cams_ids_0.7_2m_final/images',
        stats_path='/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_cams_ids_0.7_150k_frontal_final/stats.pt',
        padding=[0, 0],
        image_size=512,
        normalize_w=True,
        normalize_image=False # the image is normalized within the model
    )

    
    dloader = DataLoader(d, batch_size=16, num_workers=4)
    print('Created dataloader')

    # ------------------------------------------------------------------------------
    # Set up the queue and the workers
    # ------------------------------------------------------------------------------

    # Create a queue for the workers
    queue = mp.Queue(maxsize=100_00)

    # Create the workers
    workers = []
    for rank in range(10):
        p = mp.Process(target=proxy, args=(rank, {'queue': queue}))
        p.start()
        workers.append(p)

    # ------------------------------------------------------------------------------
    # Process the data
    # ------------------------------------------------------------------------------
    _count = 0
    for ii, _data in tqdm.tqdm(enumerate(dloader), total=len(dloader)):
        #20676/124992
        #40880/124992
        #61083
        #81298/124992
        #101498/124992
        # 121702/124992
        if ii < 121701:
            continue
        if _count < -1: 
            _count += 1
            continue
        image = _data['condition']
        image = image.cuda()
        image.requires_grad_(False)

        idx = _data['idx']

        with torch.no_grad():
            with torch.inference_mode():
                retval = model(image)  
                # Log the invalid indices 
                _all_idx = set(range(idx.shape[0]))
                _valid_idx = set(retval['valid_idx'])
                _invalid_idx = _all_idx - _valid_idx
                actual_idx = idx[list(_invalid_idx)]
                logger.warning(f'Batch {_count} Invalid indices: {actual_idx.numpy().ravel().tolist()}')

                # subset the valid indices
                idx = idx[retval['valid_idx']]
                assert idx.shape[0] == retval['seg_map'].shape[0]
                assert idx.shape[0] == retval['attrs'].shape[0]

                queue.put(
                    {
                        'seg_map': retval['seg_map'].clone().detach().cpu().numpy(),
                        'attrs': retval['attrs'].clone().detach().cpu().numpy(),
                        'dir': '/datawaha/cggroup/parawr/Projects/diffusion/data/gen_images/w_plus_img_cams_ids_0.7_2m_final',
                        'idx': idx.clone().detach().cpu().numpy()
                    }
                )

        _count += 1
    print('Finished processing data')

    # ------------------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------------------
    for _ in range(len(workers) * 10):
        queue.put(None)
    
    for p in workers:
        p.join()
    print('Workers joined')

    # empty the queue and close it
    while not queue.empty():
        queue.get()
    queue.close()
    print('Queue closed')

    # ------------------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------------------
    print('Done')
