import os
import numpy as np
import pandas as pd
import multiprocessing

from PIL import Image
from sklearn.metrics import pairwise_distances

from lib.image_lib import preprocess
from lib.image_lib import center_of_mass


LEVEL = 'level'
PRETRAIN = 'pre-train'


def find_start_end(profile):
    start = None
    end = None
    for i, val in enumerate(profile):
        if val != 0:
            start = i
            break

    for i, val in enumerate(profile[::-1]):
        if val != 0:
            end = len(profile) - i
            break

    start = start if start is not None else 0
    end = end if end is not None else len(profile)

    return start, end


def extract_word(img):
    x_profile = np.sum(img, axis=0)
    y_profile = np.sum(img, axis=1)

    x_start, x_end = find_start_end(x_profile)
    y_start, y_end = find_start_end(y_profile)

    return img[y_start:y_end, x_start:x_end]


def open_image(img_path, clean):
    img = Image.open(img_path)
    bin_img = preprocess(img)

    return extract_word(bin_img)


def create_level_descriptor(img, levels):
    width = img.shape[1]
    pixel_num = np.sum(img)
    center_x, center_y = center_of_mass(img)

    sums = []
    for level in levels:
        l_width = width // level
        for i in range(level):
            sums.append(np.sum(img[:center_y,
                                   i * l_width:i * l_width + l_width]))
            sums.append(np.sum(img[center_y:,
                                   i * l_width:i * l_width + l_width]))

    sums = np.array(sums)

    # handle special case of emtpy word image
    if pixel_num == 0:
        return sums
    else:
        return sums / float(pixel_num)


def create_level_descriptors(imgs, levels):
    descriptors = []

    for img in imgs:
        descriptors.append(create_level_descriptor(img, levels))

    return np.array(descriptors)


def create_work(img_names, img_paths, clean, thread_num):
    work = []

    img_num = len(img_paths) // thread_num
    for i in range(thread_num):
        work.append([img_names[i * img_num:i * img_num + img_num],
                     img_paths[i * img_num:i * img_num + img_num], clean])

    # add remaining paths to last worker
    work[-1][0] += img_names[img_num * thread_num:len(img_paths)]
    work[-1][1] += img_paths[img_num * thread_num:len(img_paths)]

    return work


def load(data):
    img_names, img_paths, clean = data
    imgs = []
    names = []

    for name, img_path in zip(img_names, img_paths):
        img = open_image(img_path, clean)
        # ignore empty images
        if np.sum(img) > 0:
            imgs.append(img)
            names.append(name)
        else:
            print('WARN: Skip image: {}'.format(name))

    return imgs, names


def load_imgs(img_dir, clean, thread_num):
    img_names = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    img_paths = []
    if len(img_names) == 0:
        labels = [f for f in os.listdir(img_dir)
                  if os.path.isdir(os.path.join(img_dir, f))]
        for label in labels:
            path = os.path.join(img_dir, label)
            for img in os.listdir(path):
                img_names.append('{}_{}'.format(label, img))
                img_paths.append(os.path.join(path, img))
    else:
        img_paths = [os.path.join(img_dir, f) for f in img_names]

    imgs = []
    names = []

    work = create_work(img_names, img_paths, clean, thread_num)
    pool = multiprocessing.Pool(processes=thread_num)
    results = pool.map(load, work)

    for result in results:
        imgs += result[0]
        names += result[1]

    return imgs, names


def write_distances(distances, names, out_file):
    df = pd.DataFrame(data=distances, index=names, columns=names)
    df.to_csv(out_file)


def load_phocs(phoc_file):
    data = np.load(phoc_file)
    descriptors = data['output']
    img_names = ['{}.png'.format(f.decode('utf-8')) for f in data['img_ids']]

    return descriptors, img_names


def main(img_dir, strategy, out_file, clean, levels, metric, thread_num,
         phoc_file):

    if img_dir is not None:
        imgs, img_names = load_imgs(img_dir, clean, thread_num)
    elif strategy == LEVEL:
        raise ValueError('--img_dir required for strategy: {}'
                         .format(strategy))

    if strategy == PRETRAIN and phoc_file is None:
        raise ValueError('--phoc_file required for strategy: {}'
                         .format(strategy))

    if strategy == LEVEL:
        descriptors = create_level_descriptors(imgs, levels)
    elif strategy == PRETRAIN:
        descriptors, img_names = load_phocs(phoc_file)
    else:
        raise ValueError('Strategy "{}" not supported'.format(strategy))

    distances = pairwise_distances(descriptors, metric=metric,
                                   n_jobs=thread_num)
    write_distances(distances, img_names, out_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute distances for '
                                     'given files.')
    parser.add_argument('strategy', choices=[LEVEL, PRETRAIN])
    parser.add_argument('out_file', help='Path to output file.')
    parser.add_argument('--img_dir', help='Path to folder with images to '
                        'compute distances from.', default=None)
    parser.add_argument('--phoc_file', help='Path to NPZ file with predicted '
                        'PHOC descriptors.', default=None)
    parser.add_argument('--clean', help='Remove border noise before computing '
                        'representations.', action='store_true')
    parser.add_argument('--levels', '-l', action='store',
                        type=lambda x: [int(elem) for elem in x.split(',')],
                        default='2,3,4,5', help='Comma seperated list of PHOC '
                        'unigram levels to be used when computing PHOCs. '
                        'Default: 2,3,4,5')
    parser.add_argument('--metric', help='Distance metric to use (must be'
                        'supported by SciPy).', default='l2')
    parser.add_argument('--thread_num', type=int, default=20, help='Number '
                        'of threads to use for computations (Default: 20).')

    args = vars(parser.parse_args())
    main(**args)
