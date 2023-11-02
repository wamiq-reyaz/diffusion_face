'''
    This file contains functions for generating masks.
    The masks should be in the shape of (1 x H x W ) with values of 0 or 1.
    0 denotes the hole and 1 denotes the reserved region.
    The dtype of the mask should be np.float32.
'''

import numpy as np
from PIL import Image, ImageDraw
import math
import random

ATTRIB_MAP = np.array([['0', '5_o_Clock_Shadow',],
['1', 'Arched_Eyebrows',],
['2', 'Attractive',],
['3', 'Bags_Under_Eyes',],
['4', 'Bald',],
['5', 'Bangs',],
['6', 'Big_Lips',],
['7', 'Big_Nose',],
['8', 'Black_Hair',],
['9', 'Blond_Hair',],
['10', 'Blurry',],
['11', 'Brown_Hair',],
['12', 'Bushy_Eyebrows',],
['13', 'Chubby',],
['14', 'Double_Chin',],
['15', 'Eyeglasses',],
['16', 'Goatee',],
['17', 'Gray_Hair',],
['18', 'Heavy_Makeup',],
['19', 'High_Cheekbones',],
['20', 'Male',],
['21', 'Mouth_Slightly_Open',],
['22', 'Mustache',],
['23', 'Narrow_Eyes',],
['24', 'No_Beard',],
['25', 'Oval_Face',],
['26', 'Pale_Skin',],
['27', 'Pointy_Nose',],
['28', 'Receding_Hairline',],
['29', 'Rosy_Cheeks',],
['30', 'Sideburns',],
['31', 'Smiling',],
['32', 'Straight_Hair',],
['33', 'Wavy_Hair',],
['34', 'Wearing_Earrings',],
['35', 'Wearing_Hat',],
['36', 'Wearing_Lipstick',],
['37', 'Wearing_Necklace',],
['38', 'Wearing_Necktie',],
['39', 'Young']])

VALID_ATTRIB_IDX = [5,8,9,11,12,15,16,17,20,22,24,26,29,32,33,34,35,36,37,38,39]

def get_valid_attrib_idx():
    return (ATTRIB_MAP[VALID_ATTRIB_IDX, 0]).astype(np.int32)

def random_brush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48,
    average_radius = 0):
    H, W = s, s
    if average_radius == 0:
        average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return 1 - mask[np.newaxis, ...].astype(np.float32)

def random_mask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(5 * coef), s // 2)
        MultiFill(int(3 * coef), s)
        # mask = np.logical_and(mask, 1 - random_brush(int(9 * coef), s))  # hole denoted as 0, reserved as 1
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return mask[np.newaxis, ...].astype(np.float32)

def mask_category(max_tries, n_classes, img):
    ''' max_tries: maximum number of tries to generate a mask
        n_classes: number of classes in the image
        img: (H x W x 1) the image based on which the mask is generated
    '''
    base_mask = np.ones_like(img)
    tries = np.random.randint(max_tries)
    for _ in range(tries):
        idx = np.random.randint(n_classes) / float(n_classes)
        mask = np.where(np.abs(img - idx) < 0.01, 0, 1)
        base_mask = np.logical_and(base_mask, mask)
        
    return base_mask.astype(np.float32)

def compute_idxes(idxes_list):
    # go over in steps of two and use those as start, end
    idxes = []
    for i in range(0, len(idxes_list), 2):
        start = idxes_list[i]
        end = idxes_list[i+1]
        idxes.extend(list(range(start, end+1)))
    return idxes

if __name__ == '__main__':
    print(compute_idxes([0, 10, 15, 18]))