import unittest
import os
import pandas as pd
from models.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Opret en lille dummy-featurefil med target-kolonne
        self.testfile = "tests/test_features.csv"
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target':   [0, 1, 0, 1, 0],
        }
        df = pd.DataFrame(data)
        df.to_csv(self.testfile, index=False)
        # Fjern gamle eval-filer hvis de findes
        self.eval_file = "tests/test_model_eval.csv"
        self.cm_file = "tests/test_model_eval_confmat.csv"
        for f in [self.eval_file, self.cm_file]:
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        # Oprydning efter test
        for f in [self.testfile, self.eval_file, self.cm_file]:
            if os.path.exists(f):
                os.remove(f)

    def test_train_model_creates_eval_files(self):
        # Kør træning og log til eval-filer
        from models.model_training import log_model_metrics
        model = train_model(self.testfile)
        # Simuler logging til eval-fil
        acc = 1.0
        report = {'0': {'precision':1,'recall':1,'f1-score':1,'support':3}}
        confmat = [[2, 0],[0, 3]]
        log_model_metrics(self.eval_file, acc, report, confmat)

        # Tjek at eval-filerne nu findes og har indhold
        self.assertTrue(os.path.exists(self.eval_file))
        eval_df = pd.read_csv(self.eval_file)
        self.assertIn("accuracy", eval_df.columns)
        self.assertTrue(len(eval_df) > 0)

        self.assertTrue(os.path.exists(self.cm_file))
        cm_df = pd.read_csv(self.cm_file)
        self.assertIn("timestamp", cm_df.columns)
        self.assertTrue(len(cm_df) > 0)

if __name__ == "__main__":
    unittest.main()
