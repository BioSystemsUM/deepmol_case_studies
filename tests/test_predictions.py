import os
from unittest import TestCase

from deepmol_models.models.tdc_models import BBB
from tests import TEST_DIR


class TestPredictions(TestCase):

    def setUp(self):
        self.dataset = os.path.join(TEST_DIR, 'data')
        self.dataset = os.path.join(self.dataset, "test.csv")

    def test_bbb(self):
        BBB().predict_from_csv(self.dataset, "smiles", "ID")
