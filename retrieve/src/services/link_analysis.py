import hazm
import tqdm
import re
import networkx as nx
import pandas as pd
import numpy as np


class LinkDocumentsAnalyzer:

    def __init__(self, document, elements, threshold, window_size):

        positions, filtered_elements = self.create_position_matrix(document, elements, threshold)

        self.id2element = dict(enumerate(filtered_elements))

        adjacency_matrix = self.create_adjacency_matrix(positions, window_size)

        self.graph = nx.from_numpy_matrix(adjacency_matrix)

        self.pagerank = pd.DataFrame([
            {'element': self.id2element[k], 'rank': v} for k, v in nx.pagerank_numpy(self.graph, alpha=0.9).items()
        ])
        self.pagerank = self.pagerank.sort_values('rank', ascending=False)

        hubs, authorities = nx.hits(self.graph, max_iter=1e3)

        self.hitsrank = pd.DataFrame([
            {'element': self.id2element[k], 'hubs': vh, 'authorities': va}
            for (k, vh), (_, va) in zip(hubs.items(), authorities.items())
        ])
        self.hitsrank = self.hitsrank.sort_values('hubs', ascending=False)

    @staticmethod
    def create_position_matrix(documents, elements, threshold):

        positions = np.zeros((len(elements), len(documents)), dtype=bool)
        regex = [re.compile(ele) for ele in elements]

        for row, rgx in enumerate(tqdm.tqdm(regex)):
            for col, document in enumerate(documents): positions[row, col] = bool(rgx.search(document))

        counts = np.sum(positions, axis=1)
        elements, positions = elements[counts >= threshold], positions[counts >= threshold]

        return positions, elements

    @staticmethod
    def create_adjacency_matrix(positions, window_size):

        padding = window_size - positions.shape[1] % window_size
        element_sh = positions.shape[0]

        element_pp_1 = np.pad(
            positions, pad_width=((0, 0), (0, padding))).reshape((element_sh, -1, window_size))

        element_pp_1 = np.any(element_pp_1, axis=2).astype(int)

        shift = window_size // 2
        padding = window_size - (positions.shape[1] + shift) % window_size

        element_pp_2 = np.pad(
            positions, pad_width=((0, 0), (shift, padding))).reshape((element_sh, -1, window_size))

        element_pp_2 = np.any(element_pp_2, axis=2).astype(int)

        adjacency = np.dot(element_pp_1, element_pp_1.T) + np.dot(element_pp_2, element_pp_2.T)
        adjacency = adjacency.astype(bool).astype(int)

        return adjacency

    def get_pagerank(self, element):
        return self.pagerank[self.pagerank.element.str.contains(element)]

    def get_hitsrank(self, element):
        return self.hitsrank[self.hitsrank.element.str.contains(element)]

    def get_pageranks(self, n):
        return self.pagerank[:n]

    def get_hitsranks(self, n):
        return self.hitsrank[:n]


if __name__ == '__main__':
    normalizer = hazm.Normalizer(token_based=True)

    poems = pd.read_csv('../../resources/shahnameh-dataset.csv')['text']
    poems = poems.apply(normalizer.normalize)

    chars = pd.read_csv('../../resources/shahnameh_characters.csv')['regex']
    chars = chars.apply(normalizer.normalize)

    analyzer = LinkDocumentsAnalyzer(poems, chars, 1, 5)

    print(analyzer.get_pagerank('رستم'), analyzer.get_hitsrank('رستم'))
