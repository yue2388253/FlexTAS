import logging
import os.path
import unittest
from app.scheduler import schedule
from definitions import ROOT_DIR


class TestSchedule(unittest.TestCase):
    def setUp(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.filename = os.path.join(ROOT_DIR, 'data/input/smt_output.json')

    def test_schedule(self):
        schedule(self.filename)

    def test_vec_env(self):
        schedule(self.filename, 4)

