import os
import numpy as np
import shutil
import pandas as pd
import random

from ipm import ipm


def read_imgs(img_dir):
    imgs = {}

    classes = [f for f in os.listdir(img_dir) if os.path.isdir(
                os.path.join(img_dir, f))]
    for c in classes:
        path = os.path.join(img_dir, c)
        for i in os.listdir(path):
            img_id = '{}_{}'.format(c, i.split('.')[0])

            imgs[img_id] = os.path.join(path, i)

    return imgs


def random_select(img_dir, out_dir, num, random_seed):
    imgs = read_imgs(img_dir)
    ids = sorted(list(imgs.keys()))

    random.seed(random_seed)
    random.shuffle(ids)
    for img_id in ids[:num]:
        shutil.copy2(imgs[img_id],
                     os.path.join(out_dir, '{}.png'.format(img_id)))


def load_dataset_map(dataset_map):
    mapping = {}
    with open(dataset_map, 'r') as f:
        for line in f:
            parts = line.split(',')
            mapping[parts[0].strip()] = parts[1].strip()

    return mapping


def get_excluded_idxs(img_ids, exclude):
    idxs = []

    excluded = [f.split('.')[0] for f in os.listdir(exclude)]

    for i, img_id in enumerate(img_ids):
        img_id = img_id.split('.')[0]
        if img_id in excluded:
            idxs.append(i)

    return idxs


def ipm_select(img_dir, out_dir, distance_file, num, dataset_map, exclude):
    data = pd.read_csv(distance_file)
    img_ids = data.iloc[:, 0]
    rows, cols = data.shape
    if type(data.iloc[0, cols - 1]) == str:
        # remove last column, if it is a string column
        distances = np.array(data.iloc[:, 1:cols - 1])
    else:
        distances = np.array(data.iloc[:, 1:])

    if exclude is None:
        selected_data = None
    else:
        selected_idxs = get_excluded_idxs(img_ids, exclude)
        selected_data = list(distances[selected_idxs])

    idxs = ipm(list(distances), n=num, selected_data=selected_data)
    if dataset_map is not None:
        id_map = load_dataset_map(dataset_map)

    for idx in idxs:
        img_id = img_ids[idx].split('.')[0]
        if dataset_map is not None:
            img_id = id_map[img_id]
        label, img = img_id.split('_')
        src = os.path.join(img_dir, label, '{}.png'.format(img))
        dest = os.path.join(out_dir, '{}.png'.format(img_id))

        shutil.copy2(src, dest)


def main(img_dir, out_dir, num, distance_file, random_seed, dataset_map,
         exclude):
    if distance_file is not None and random_seed is not None:
        raise ValueError('Either a distance file or a random seed must be '
                         'provided.')
    if distance_file is not None:
        ipm_select(img_dir, out_dir, distance_file, num, dataset_map, exclude)
    elif random_seed is not None:
        random_select(img_dir, out_dir, num, random_seed)
    else:
        raise ValueError('Either a distance file or a random seed must be '
                         'provided.')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Select samples based '
                                     'on the provided distance matrix or '
                                     'randomly.')

    parser.add_argument('img_dir', help='Path to image root folder.')
    parser.add_argument('out_dir', help='Path to folder in which to store the'
                        ' selected samples.')
    parser.add_argument('num', help='Number of samples to select.', type=int)

    distance = parser.add_argument_group(title='Distance based Selection')
    distance.add_argument('--distance_file', help='Path to CSV file with '
                          'precomputed distances.', default=None)
    distance.add_argument('--dataset_map', help='Mapping file, which maps the '
                          'names of the selected files to the names of the '
                          'files to select (In case the file names in the '
                          'distance file do not match the names in the image '
                          'root folder).', default=None)
    distance.add_argument('--exclude', help='Path to folder with images to '
                          'exclude from selection.', default=None)

    rand = parser.add_argument_group(title='Random Selection')
    rand.add_argument('--random_seed', help='Select samples randomly using '
                      'provided seed value.', default=None, type=int)

    args = vars(parser.parse_args())
    main(**args)
