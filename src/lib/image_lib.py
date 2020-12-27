import PIL
import numpy as np
import skimage.filters as filters


def binarize(image, name):
    """Binarize PIL image using Otsu's binarization algorithm."""
    a = np.array(image)
    new = np.zeros(a.shape)
    try:
        thresh = filters.threshold_otsu(a)
        idxs = a > thresh
        new[idxs] = 1
    except ValueError:
        print('WARN: Empty word image: {}'.format(name))

    return new


def preprocess(image, seam_carve=True):
    """Binarize image and remove border noise if seam_carve is True."""
    i = image.convert('L')
    i = PIL.ImageOps.invert(i)

    new = binarize(i, image.filename)
#        new = self._pp_extract(new)

    # only run seam carving, if there is something in the image
    if seam_carve and np.sum(new) > 0:
        carved = seam_carve_extract(new.copy())
        # if less than 5% of the original pixels remain after seam carving,
        # the result is probably not very good, so we rather use the
        # original image
        if np.sum(new) * 0.05 > np.sum(carved):
            processed = new
        else:
            processed = carved
    else:
        processed = new

    return processed


def center_of_mass(segment):
    """Compute the center of mass for the given Numpy array."""
    n = np.sum(segment)

    # handle special case if there is no foreground pixel in segment
    if n == 0:
        return segment.shape[1]//2, segment.shape[0]//2
    # set index to start with 1 to avoid multiplication with 0
    x = np.sum(segment, axis=0) * range(1, segment.shape[1]+1)
    x = np.round(np.sum(x)/n).astype(int)
    y = np.sum(segment, axis=1) * range(1, segment.shape[0]+1)
    y = np.round(np.sum(y)/n).astype(int)
    # reset index to Python indexing scheme
    return x-1, y-1


def seam_carve_extract(image):
    """Remove boarder noise from given PIL image using seam carving."""
    r, c = image.shape
    c_mid, r_mid = center_of_mass(image)
    r_min_gap = r * 0.01
    c_min_gap = c * 0.01

    left_cost, left_back = _minimum_seam(image[:, :c_mid], c_min_gap,
                                         True, True)
    right_cost, right_back = _minimum_seam(image[:, c_mid:], c_min_gap,
                                           False, True)
    top_cost, top_back = _minimum_seam(image[:r_mid, :], r_min_gap,
                                       True, False)
    bottom_cost, bottom_back = _minimum_seam(image[r_mid:, :], r_min_gap,
                                             False, False)

    total_pixels = np.sum(image)
    image[:, :c_mid] = _clear_seam(image[:, :c_mid], left_cost, left_back,
                                   True, True, total_pixels)
    image[:, c_mid:] = _clear_seam(image[:, c_mid:], right_cost, right_back,
                                   False, True, total_pixels)
    image[:r_mid, :] = _clear_seam(image[:r_mid, :], top_cost, top_back,
                                   True, False, total_pixels)
    image[r_mid:, :] = _clear_seam(image[r_mid:, :], bottom_cost, bottom_back,
                                   False, False, total_pixels)

    return image


def _init_costs(M, min_gap):
    r, c = M.shape
    border = -1 * r * c
    M *= border   # cost to cross a foreground patch
    for i in range(r):
        for j in range(c):
            if M[i, j] >= 0:
                if j == 0 or M[i, j-1] < 0:
                    M[i, j] = 1
                else:
                    M[i, j] = M[i, j-1] + 1
        for j in reversed(range(c)):
            if M[i, j] >= 0:
                if j == c-1 or M[i, j+1] < 0:
                    if M[i, j] >= min_gap:
                        even = (M[i, j] % 2) == 0
                        M[i, j] = 1
                        block = False
                    else:
                        M[i, j] = border
                        block = True
                else:
                    if block:
                        M[i, j] = border
                    else:
                        tmp = M[i, j+1]
                        if tmp <= M[i, j-1]:
                            # prevent pull to the image center
                            M[i, j] = tmp+1 if tmp+1 <= 2*min_gap else tmp
                        elif even:
                            M[i, j] = tmp
                            even = False
                        else:
                            M[i, j] = tmp-1


def _minimum_seam(img, min_gap, start=True, column=True):
    M = img.copy()

    if not column:
        M = M.transpose()

    r, c = M.shape
    _init_costs(M, min_gap)

    backtrack = np.zeros_like(M, dtype=np.int)
    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't
            # index -1
            if j == 0:
                # set preference for seam to choose - for start prefer left
                # for end prefer right
                if start:
                    idx = np.argmax(M[i - 1, j:j + 2])
                else:
                    idx = np.argmax(M[i-1, j:j + 2][::-1])
                    idx = 2 - idx - 1
                backtrack[i, j] = idx + j
                max_energy = M[i - 1, idx + j]
            # Handle the right edge of the image
            elif j == c-1:
                if start:
                    idx = np.argmax(M[i - 1, j - 1:j + 1])
                else:
                    idx = np.argmax(M[i - 1, j - 1:j + 1][::-1])
                    idx = 2 - idx - 1
                backtrack[i, j] = idx + j - 1
                max_energy = M[i - 1, idx + j - 1]
            else:
                if start:
                    idx = np.argmax(M[i - 1, j - 1:j + 2])
                else:
                    idx = np.argmax(M[i - 1, j - 1:j + 2][::-1])
                    idx = 3 - idx - 1
                backtrack[i, j] = idx + j - 1
                max_energy = M[i - 1, idx + j - 1]

            M[i, j] += max_energy

    return M, backtrack


def _clear_seam(img, cost, back, start, column, total_pixels):
    # handle corner cases where the cost function has zero width
    if len(cost[-1]) == 0:
        return img
    # find starting point, which maximizes margins between non zero seams
    # idx, length = self._longest_zeros(cost[-1])
    if start:
        j = np.argmax(cost[-1])
    else:
        j = np.argmax(cost[-1][::-1])
        j = len(cost[-1]) - 1 - j

    # clear only if seam with zero cost is present
    if cost[-1, j] > 0:
        r, c = cost.shape
        for i in reversed(range(r)):
            _erase_pixels(img, i, j, start, column)
            j = back[i, j]
    return img


def _erase_pixels(img, i, j, start, column):
    if start and column:
        img[i, :j] = 0
    elif not start and column:
        img[i, j:] = 0
    elif start and not column:
        img[:j, i] = 0
    elif not start and not column:
        img[j:, i] = 0
