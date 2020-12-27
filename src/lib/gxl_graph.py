import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np


class GXLGraph(object):

    def __init__(self, name, mean, std, label=None):
        self.name = name
        self.org_mean = mean
        self.org_std = std
        self.nodes = {}
        self.edges = []
        self.label = label

    @staticmethod
    def read_gxl_header(graph):
        mean = np.array([float(graph.attrib['org-mean-x']),
                        float(graph.attrib['org-mean-y'])])
        std = np.array([float(graph.attrib['org-std-x']),
                        float(graph.attrib['org-std-y'])])

        name = graph.attrib['id']
        if 'label' in graph.attrib:
            label = graph.attrib['label']
        else:
            label = None
        return name, mean, std, label

    @staticmethod
    def read_gxl_nodes(graph):
        nodes = {}
        for node in graph.findall('node'):
            key = node.attrib['id']
            x = float(node[0][0].text)
            y = float(node[1][0].text)
            nodes[(x, y)] = GXLGraph.key_to_int(key)
        return nodes

    @staticmethod
    def read_gxl_edges(graph):
        edges = []
        for edge in graph.findall('edge'):
            key1 = edge.attrib['from']
            key2 = edge.attrib['to']
            key1 = GXLGraph.key_to_int(key1)
            key2 = GXLGraph.key_to_int(key2)
            edges.append((key1, key2))
        return edges

    @staticmethod
    def key_to_int(key):
        return int(key.split('_')[1])

    @staticmethod
    def int_to_key(key):
        return '_{}'.format(key)

    @classmethod
    def from_gxl(cls, gxl_path):
        tree = ET.parse(gxl_path)
        root = tree.getroot()
        graph = root[0]

        name, mean, std, label = GXLGraph.read_gxl_header(graph)
        self = cls(name, mean, std, label)

        self.nodes = GXLGraph.read_gxl_nodes(graph)
        self.edges = GXLGraph.read_gxl_edges(graph)

        return self

    def get_unnormalized_nodes(self):
        mean = self.org_mean
        std = self.org_std
        nodes = [None for n in self.nodes]
        for n in self.nodes:
            nodes[self.nodes[n]] = n
        nodes = np.array(nodes)
        nodes = np.round(nodes*std + mean).astype(int)

        return nodes

    def update_node_locations(self, node_locations):
        locations = (node_locations - self.org_mean) / self.org_std

        # replace node locations
        self.nodes.clear()
        for i, n in enumerate(locations):
            self.nodes[(n[0], n[1])] = i

    def add_edge(self, n1, n2):
        n1 = tuple(n1)
        n2 = tuple(n2)
        if n1 not in self.nodes:
            self.nodes[n1] = len(self.nodes)
        if n2 not in self.nodes:
            self.nodes[n2] = len(self.nodes)

        self.edges.append((self.nodes[n1], self.nodes[n2]))

    def _to_string(self, root):
        # add document type and pretty print XML
        rough_string = ET.tostring(root, 'utf-8')
        reparsed = minidom.parseString(rough_string)

        imp = minidom.getDOMImplementation('')
        dt = imp.createDocumentType('gxl', None,
                                    'http://www.gupro.de/GXL/gxl-1.0.dtd')
        reparsed.insertBefore(dt, reparsed.documentElement)

        return reparsed.toprettyxml(encoding="utf-8",
                                    indent="  ").decode('utf-8')

    def _add_attr(self, root, name, value):
        attr = ET.SubElement(root, 'attr')
        attr.set('name', name)
        val = ET.SubElement(attr, 'float')
        val.text = str(value)

    def _setup_graph(self, root):
        graph = ET.SubElement(root, 'graph')
        graph.set('id', self.name)
        graph.set('edgeids', 'false')
        graph.set('edgemode', 'undirected')
        graph.set('org-mean-x', str(self.org_mean[0]))
        graph.set('org-mean-y', str(self.org_mean[1]))
        graph.set('org-std-x', str(self.org_std[0]))
        graph.set('org-std-y', str(self.org_std[1]))
        if self.label:
            graph.set('label', self.label)

        return graph

    def _add_nodes(self, root, node_std):
        for node in self.nodes:
            n = ET.SubElement(root, 'node')
            n.set('id', GXLGraph.int_to_key(self.nodes[node]))

            self._add_attr(n, 'x', node[0])
            self._add_attr(n, 'y', node[1])
            self._add_attr(n, 'std_x', node_std[0])
            self._add_attr(n, 'std_y', node_std[1])

    def _add_edges(self, root):
        for edge in self.edges:
            e = ET.SubElement(root, 'edge')
            e.set('from', GXLGraph.int_to_key(edge[0]))
            e.set('to', GXLGraph.int_to_key(edge[1]))

    def to_gxl(self):
        n = np.array(list(self.nodes))
        node_std = np.std(n, axis=0)

        root = ET.Element('gxl')
        root.set('xmlns:xlink', 'http://www.w3.org/1999/xlink')

        graph = self._setup_graph(root)
        self._add_nodes(graph, node_std)
        self._add_edges(graph)

        return self._to_string(root)


class GraphCollection(object):

    def __init__(self, files):
        '''
            Class for generating the GraphCollection XML required for GEDLIB

            Args:
                files - list of tuples (<file name>, <class>)

        '''
        self._root = ET.Element('GraphCollection')

        for f in files:
            sub = ET.SubElement(self._root, 'graph')
            sub.set('file', f[0])
            sub.set('class', f[1])

    def to_xml(self):
        '''
            Convert graph collection into XML

            Returns:
                XML string
        '''
        # add document type and pretty print XML
        rough_string = ET.tostring(self._root, 'utf-8')
        reparsed = minidom.parseString(rough_string)

        imp = minidom.getDOMImplementation('')
        dt = imp.createDocumentType('GraphCollection', None,
                                    'http://www.inf.unibz.it/~blumenthal/dtd'
                                    '/GraphCollection.dtd')
        reparsed.insertBefore(dt, reparsed.documentElement)

        return reparsed.toprettyxml(encoding="utf-8",
                                    indent="  ").decode('utf-8')
