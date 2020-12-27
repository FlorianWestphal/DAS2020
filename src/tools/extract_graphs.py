from lib.graph_extractor import GraphExtractor
from lib.gxl_graph import GraphCollection

import multiprocessing
import os


def create_work(src_path, dest_path, d_v, d_h, thread_num, seam_carve):
    '''
        Create list of work items for <thread_num> many threads.
        One work item consists of a list of images from which graphs should be
        extracted and the graph extraction parameters.

        Args:
            src_path - folder structure containing word images
            dest_path - target folder in which graphs should be stored
            d_v - Vertical distance threshold
            d_h - Horizontal distance threshold
            thread_num - number of threads

        Returns:
            work - list of work items
    '''
    work = []
    folders = [os.path.join(src_path, f)
               for f in os.listdir(src_path)
               if os.path.isdir(os.path.join(src_path, f))]
    input_files = []
    for folder in folders:
        input_files += [os.path.join(folder, f)
                        for f in os.listdir(folder)
                        if f.endswith('.png')]

    per_thread, rest = divmod(len(input_files), thread_num)
    last = 0
    for i in range(thread_num):
        tmp_list = input_files[last:last+per_thread]
        last = last+per_thread
        # distribute remaining files evenly among threads
        if rest != 0:
            tmp_list.append(input_files[last])
            last += 1
            rest -= 1
        work.append((tmp_list, dest_path, d_v, d_h, seam_carve))
    return work


def extract_graphs(data):
    '''
        Thread function, which extracts the graphs from the provided images.

        Args:
            data - tuple containing (<list of image files>, <destination path>,
                                    <Vertical distance>, <Horizontal distance>,
                                    <use seam carving>)
        Returns:
            processed - list of tuples containing (<graph file name>,
                                                    <word class>)
    '''
    files, dest_path, d_v, d_h, seam_carve = data
    extractor = GraphExtractor(d_v=d_v, d_h=d_h, seam_carve=seam_carve)

    processed = []
    for f in files:
        path = f.split('/')
        name = '{}_{}'.format(path[-2], path[-1].split('.')[0])

        g = extractor.extract_graph(f, name)
        if g is not None:
            out_file = '{}.gxl'.format(name)
            processed.append((out_file, path[-2]))
            with open(os.path.join(dest_path, out_file), 'w') as out:
                out.write(g)
        else:
            print('WARN: skipping graph: {}'.format(name))

    return processed


def assemble(results):
    '''
        Convert lists of extracted graphs into XML format.

        Args:
            retults - list of graph lists

        Returns:
            xml - XML string representing converted graph files
    '''

    result = []
    for r in results:
        result += r

    graphs = GraphCollection(result)
    return graphs.to_xml()


def main(src_path, dest_path, d_v, d_h, thread_num, no_seam_carve):

    work = create_work(src_path, dest_path, d_v, d_h, thread_num,
                       not no_seam_carve)

    pool = multiprocessing.Pool(processes=thread_num)
    results = pool.map(extract_graphs, work)

    result = assemble(results)

    xml_name = 'graphs_dv{}_dh{}.xml'.format(d_v, d_h)
    with open(os.path.join(dest_path, xml_name), 'w') as out:
        out.write(result)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract graphs from given '
                                     'images using projection profiles with '
                                     'provided distance thresholds.')

    parser.add_argument('src_path', help='Path to images root folder')
    parser.add_argument('dest_path', help='Path to target location')
    parser.add_argument('d_v', help='Vertical distance threshold', type=int)
    parser.add_argument('d_h', help='Horizontal distance threshold', type=int)
    parser.add_argument('--thread_num', help='Number of threads to use '
                        '(default: 20)', type=int, default=20)
    parser.add_argument('--no_seam_carve', action='store_true',
                        help='Deactivate seam carving.')

    args = vars(parser.parse_args())

    main(**args)
