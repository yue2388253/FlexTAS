import unittest

from src.lib.operation import Operation, check_operation_isolation


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

    def test_isolation(self):
        o1 = Operation(0, None, 10, 20)
        o2 = Operation(100, None, 110, 120)
        safe_distance = 1

        self.assertEqual(
            check_operation_isolation((o1, 100), (o2, 1000), safe_distance),
            21
        )

        self.assertEqual(
            check_operation_isolation((o1, 200), (o2, 300), safe_distance),
            21
        )

        self.assertEqual(
            check_operation_isolation((o1, 20), (o2, 120), safe_distance),
            41
        )

        self.assertEqual(
            check_operation_isolation((o1, 200), (o2, 200), safe_distance),
            None
        )

        with self.assertRaises(AssertionError):
            check_operation_isolation((o1, 200), (o2, 100), safe_distance)

        # The original value should not change
        self.assertEqual(o1.start_time, 0)
        self.assertEqual(o2.start_time, 100)
