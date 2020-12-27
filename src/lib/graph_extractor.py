import numpy as np

from PIL import Image

import skimage.morphology as morph
import networkx as nx
import lib.gxl_graph as gxl

from lib.image_lib import preprocess


class GraphExtractor:

    def __init__(self, d_v=40, d_h=40, debug=False, seam_carve=True):
        self.d_v = d_v
        self.d_h = d_h
        self.debug = debug
        self.seam_carve = seam_carve

    def extract_graph(self, img_path, name):
        org_img = Image.open(img_path)
        img = preprocess(org_img, self.seam_carve)

        # do not extract a graph if the image is empty
        if np.sum(img) == 0:
            return None

        vpp = self._vertical_pp(img)
        segments = self._profile_segment(img, vpp, True)
        v_sgmts = []
        for sgmt in segments:
            v_sgmts += self._segment_equidistant(sgmt[2], self.d_v, True,
                                                 sgmt[0], sgmt[1])
        final_sgmts = []
        for sgmt in v_sgmts:
            hpp = self._horizontal_pp(sgmt[2])
            segments = self._profile_segment(sgmt[2], hpp, False, sgmt[0],
                                             sgmt[1])
            for s in segments:
                final_sgmts += self._segment_equidistant(s[2], self.d_h,
                                                         False, s[0], s[1])
        nodes = []
        locations = []
        for sgmt in final_sgmts:
            x, y = self._center_of_mass(sgmt[2])
            node = np.array([x+sgmt[0], y+sgmt[1]])
            nodes.append(node)
            locations.append((node, sgmt))
        thin = morph.thin(img)
        edges = []
        for loc in locations:
            neighbors = self._find_neighbors(loc, locations, thin)
            for n in neighbors:
                if loc[0][0] <= n[0][0]:
                    edges.append((loc[0], n[0]))
        assert len(nodes) > 0, 'no node found in {}'.format(org_img.filename)
        return self._build_gxl(nodes, edges, name)

    def _longest_zeros(self, array):
        idx = None
        tmp = None
        cnt = 0
        for i, v in enumerate(array):
            if v == 0:
                if tmp:
                    tmp += 1
                else:
                    tmp = 1
            else:
                if tmp:
                    if tmp > cnt:
                        cnt = tmp
                        idx = i - tmp
                    tmp = None
        if tmp:
            if tmp > cnt:
                cnt = tmp
                idx = len(array)-tmp
        return idx, cnt

    def _find_gaps(self, pp):
        mid = len(pp) // 2
        start = 0
        end = len(pp) - 1

        idx, cnt = self._longest_zeros(pp[:mid])
        if idx:
            start = idx + (cnt // 2)
        idx, cnt = self._longest_zeros(pp[mid:])
        if idx:
            end = mid + idx + (cnt // 2)
        return start, end

    def _vertical_pp(self, a):
        return np.sum(a, axis=0).reshape(-1)

    def _horizontal_pp(self, a):
        return np.sum(a, axis=1).reshape(-1)

    def _profile_segment(self, array, pp, vertical, s_x=0, s_y=0):
        # locate change between 0 and 1
        tmp = pp != 0
        idxs = np.where(tmp[:-1] != tmp[1:])[0]
        if len(idxs) == 0:
            return [(s_x, s_y, array)]
        skip = False
        segments = []
        first = None if pp[0] == 0 else 0
        for i in idxs:
            if first is None:
                first = i + 1
                continue
            elif not skip:
                x = first if vertical else 0
                y = 0 if vertical else first
                if vertical:
                    segments.append((x+s_x, y+s_y, array[:, first:i+1]))
                else:
                    segments.append((x+s_x, y+s_y, array[first:i+1, :]))
            first = i+1
            skip = not skip

        if not skip:
            x = first if vertical else 0
            y = 0 if vertical else first
            if vertical:
                segments.append((x+s_x, y+s_y, array[:, first:]))
            else:
                segments.append((x+s_x, y+s_y, array[first:, :]))

        return segments

    def _segment_equidistant(self, array, distance, vertical, s_x=0, s_y=0):
        side = array.shape[1] if vertical else array.shape[0]
        first = 0
        segments = []
        for i in range(distance, side, distance):
            x = first if vertical else 0
            y = 0 if vertical else first
            if vertical:
                segments.append((x+s_x, y+s_y, array[:, first:i]))
            else:
                segments.append((x+s_x, y+s_y, array[first:i, :]))
            first = i
        if len(segments) == 0:
            segments.append((s_x, s_y, array))
        elif i != side:
            x = first if vertical else 0
            y = 0 if vertical else first
            if vertical:
                segments.append((x+s_x, y+s_y, array[:, first:side]))
            else:
                segments.append((x+s_x, y+s_y, array[first:side, :]))
        return segments

    def _get_coordinates(self, sgmt):
        x = sgmt[0]
        x2 = x + sgmt[2].shape[1]
        y = sgmt[1]
        y2 = y + sgmt[2].shape[0]
        return x, x2, y, y2

    def _between(self, u1, u2, v1, v2):
        if v1 >= u1 and v1 <= u2:
            return True
        elif v2 >= u1 and v2 <= u2:
            return True
        elif u1 > v1 and u2 < v2:
            return True
        else:
            return False

    def _connecting_line(self, slice1, slice2, id1, id2):
        idx1 = np.where(slice1 > 0)
        idx2 = np.where(slice2 > 0)
        if len(idx1[0]) == 0 or len(idx2[0]) == 0:
            return False
        connecting = False
        for i1 in idx1[0]:
            for i2 in idx2[0]:
                pos1 = i1 + id1
                pos2 = i2 + id2
                connecting = connecting or np.abs(pos1 - pos2) <= 2
        return connecting

    def _connecting_line_x(self, y11, y12, y21, y22, x1, x2, thin):
        slice1 = thin[y11:y12, x1].reshape(-1)
        slice2 = thin[y21:y22, x2].reshape(-1)
        return self._connecting_line(slice1, slice2, y11, y21)

    def _connecting_line_y(self, x11, x12, x21, x22, y1, y2, thin):
        slice1 = thin[y1, x11:x12].reshape(-1)
        slice2 = thin[y2, x21:x22].reshape(-1)
        return self._connecting_line(slice1, slice2, x11, x21)

    def _left_neighbor(self, coord1, coord2, thin):
        # one y value of coord2 is between the y values of coord1 and they
        # border on the left
        neighbor = False
        if (self._between(coord1[2], coord1[3], coord2[2], coord2[3])
                and coord1[0] == coord2[1]):
            neighbor = self._connecting_line_x(coord1[2], coord1[3],
                                               coord2[2], coord2[3],
                                               coord1[0], coord2[1] - 1, thin)
        return neighbor

    def _right_neighbor(self, coord1, coord2, thin):
        # one y value of coord2 is between the y values of coord1 and they
        # border on the right
        neighbor = False
        if (self._between(coord1[2], coord1[3], coord2[2], coord2[3])
                and coord1[1] == coord2[0]):
            neighbor = self._connecting_line_x(coord1[2], coord1[3],
                                               coord2[2], coord2[3],
                                               coord1[1] - 1, coord2[0], thin)
        return neighbor

    def _top_neighbor(self, coord1, coord2, thin):
        neighbor = False
        if (self._between(coord1[0], coord1[1], coord2[0], coord2[1])
                and coord1[3] == coord2[2]):
            neighbor = self._connecting_line_y(coord1[0], coord1[1],
                                               coord2[0], coord2[1],
                                               coord1[3] - 1, coord2[2], thin)
        return neighbor

    def _bottom_neighbor(self, coord1, coord2, thin):
        neighbor = False
        if (self._between(coord1[0], coord1[1], coord2[0], coord2[1])
                and coord1[2] == coord2[3]):
            neighbor = self._connecting_line_y(coord1[0], coord1[1],
                                               coord2[0], coord2[1],
                                               coord1[2], coord2[3] - 1, thin)
        return neighbor

    def _find_neighbors(self, instance, locations, thin):
        sgmt = instance[1]
        i_coords = self._get_coordinates(sgmt)
        neighbors = []
        for loc in locations:
            if loc is instance:
                continue
            else:
                coords = self._get_coordinates(loc[1])
                if self._top_neighbor(i_coords, coords, thin):
                    neighbors.append(loc)
                elif self._bottom_neighbor(i_coords, coords, thin):
                    neighbors.append(loc)
                elif self._left_neighbor(i_coords, coords, thin):
                    neighbors.append(loc)
                elif self._right_neighbor(i_coords, coords, thin):
                    neighbors.append(loc)
        return neighbors

    def _standardize(self, node, mean, std):
        # avoid division by zero
        std[std == 0] = 1
        return (node - mean) / std

    def _get_mean_std(self, nodes):
        vertices = np.array(nodes)
        mean = np.mean(vertices, axis=0)
        std = np.std(vertices, axis=0)

        return mean, std

    def _build_graph(self, nodes, edges):
        mean, std = self._get_mean_std(nodes)

        graph = nx.Graph(meano=mean, stdo=std)
        for edge in edges:
            n1 = self._standardize(edge[0], mean, std)
            n2 = self._standardize(edge[1], mean, std)
            graph.add_edge((n1[0], n1[1]), (n2[0], n2[1]))
        n = np.array(graph.nodes)
        graph.graph['std'] = np.std(n, axis=0)
        return graph

    def _build_gxl(self, nodes, edges, name):
        mean, std = self._get_mean_std(nodes)
        graph = gxl.GXLGraph(name, mean, std)

        for edge in edges:
            n1 = self._standardize(edge[0], mean, std)
            n2 = self._standardize(edge[1], mean, std)
            graph.add_edge((n1[0], n1[1]), (n2[0], n2[1]))

        return graph.to_gxl()


def main(img_path, out_file, d_v, d_h, debug):
    path = img_path.split('/')
    name = '{}/{}'.format(path[-2], path[-1].split('.')[0])

    extractor = GraphExtractor(d_v, d_h, debug)
    g = extractor.extract_graph(img_path, name)

    with open(out_file, 'w') as out:
        out.write(g)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract graphs from given '
                                     'image')

    parser.add_argument('img_path', help='Path to image.')
    parser.add_argument('out_file', help='Path to output GXL file.')
    parser.add_argument('--d_v', help='Vertical projection profile parameter '
                        '(default: 16).', type=int, default=16)
    parser.add_argument('--d_h', help='Horizontal projection profile parameter'
                        ' (default: 10).', type=int, default=10)
    parser.add_argument('--debug', help='Add debug output',
                        action='store_true')

    args = vars(parser.parse_args())

    main(**args)
