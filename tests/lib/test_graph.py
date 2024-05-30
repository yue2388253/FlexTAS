import unittest

from src.lib.graph import neighbors_within_distance
from src.network.net import generate_cev


class TestGraphNeighbor(unittest.TestCase):
    def test_neighbors(self):
        g = generate_cev()
        neighbors = neighbors_within_distance(g, 'SW11', 2)
        self.assertEqual(len(neighbors), 14)
