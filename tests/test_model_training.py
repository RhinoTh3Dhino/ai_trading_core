import unittest
import os
import pandas as pd
from models.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Opret en dummy feature-fil med 100 rækker og target
        self.testfile = "tests/test_features.csv"
        data = {
            'feature1': list(range(100)),
            'feature2': list(range(100, 200)),
            'target':   [0, 1] * 50
        }
        df = pd.DataFrame(data)
        df.to_csv(self.testfile, index=False)

        # Paths brugt i model_training.py
        self.eval_file = "data/model_eval.csv"
        self.cm_file = "data/model_eval_confmat.csv"
        self.model_path = "models/best_model.pkl"

        # Slet gamle filer før test
        for f in [self.eval_file, self.cm_file, self.model_path]:
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        # Slet testdata og outputfiler efter test
        for f in [self.testfile, self.eval_file, self.cm_file, self.model_path]:
            if os.path.exists(f):
                os.remove(f)

    def test_train_model_creates_outputs(self):
        # Kør træning
        train_model(self.testfile)

        # Eval-fil skal være oprettet og indeholde accuracy
        self.assertTrue(os.path.exists(self.eval_file))
        eval_df = pd.read_csv(self.eval_file)
        self.assertIn("accuracy", eval_df.columns)
        self.assertTrue(len(eval_df) > 0)

        # Confusion matrix-fil skal eksistere og have timestamp
        self.assertTrue(os.path.exists(self.cm_file))
        cm_df = pd.read_csv(self.cm_file)
        self.assertIn("timestamp", cm_df.columns)
        self.assertTrue(len(cm_df) > 0)

        # Model skal være gemt
        self.assertTrue(os.path.exists(self.model_path))

if __name__ == "__main__":
    unittest.main()
