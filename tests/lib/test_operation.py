import unittest

from src.lib.operation import Operation


class TestOperation(unittest.TestCase):
    def test_earliest_time(self):
        o1 = Operation(0, 10, 10, 20)
        self.assertEqual(o1.earliest_time, 10)

        with self.assertRaises(AssertionError):
            Operation(0, 10, 5, 20)

        with self.assertRaises(AssertionError):
            Operation(0, 10, 15, 20)

        o2 = Operation(0, None, 10, 20)
        self.assertEqual(o2.earliest_time, 0)

        o3 = o1.add(3)
        self.assertEqual(o3.earliest_time, 13)
        self.assertEqual(o1.earliest_time, 13)
        self.assertEqual(id(o3), id(o1))

        o4 = o2.add(5)
        self.assertEqual(o4.earliest_time, 5)
