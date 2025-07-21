import unittest

import glob
import pandas as pd




from models.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def setUp(self):
        # Opret testdata
        self.testfile = "tests/test_features.csv"
        data = {
            'feature1': list(range(100)),
            'feature2': list(range(100, 200)),
            'target':   [0, 1] * 50
        }
        os.makedirs("tests", exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(self.testfile, index=False)

        # Paths brugt i model_training.py
        self.eval_file = "data/model_eval.csv"
        self.model_path = "models/best_model.pkl"
        self.meta_path = "models/best_model_meta.json"
        # Alle .csv/.png confusion matrix-filer (de får timestamp på nu)
        self.cm_files = glob.glob("data/model_eval_confmat*.csv")
        self.cm_pngs = glob.glob("data/model_eval_confmat*.png")

        # Slet gamle outputfiler
        for f in [self.eval_file, self.model_path, self.meta_path] + self.cm_files + self.cm_pngs:
            if os.path.exists(f):
                os.remove(f)

    def tearDown(self):
        # Slet alt testoutput efter test
        files = [self.testfile, self.eval_file, self.model_path, self.meta_path]
        files += glob.glob("data/model_eval_confmat*.csv")
        files += glob.glob("data/model_eval_confmat*.png")
        for f in files:
            if os.path.exists(f):
                os.remove(f)

    def test_train_model_creates_outputs(self):
        # Kør træning
        model, model_path, feature_cols = train_model(self.testfile)

        # Eval-fil skal være oprettet og indeholde vigtige kolonner
        self.assertTrue(os.path.exists(self.eval_file))
        eval_df = pd.read_csv(self.eval_file)
        for col in ["accuracy", "version", "git_hash"]:
            self.assertIn(col, eval_df.columns)
        self.assertTrue(len(eval_df) > 0)

        # Confusion matrix-fil skal eksistere (de får nu timestamp)
        cm_files = glob.glob("data/model_eval_confmat*.csv")
        self.assertTrue(len(cm_files) > 0)
        cm_df = pd.read_csv(cm_files[0])
        self.assertIn("timestamp", cm_df.columns)

        # Confusion matrix-billede skal eksistere
        cm_pngs = glob.glob("data/model_eval_confmat*.png")
        self.assertTrue(len(cm_pngs) > 0)

        # Model og meta skal være gemt
        self.assertTrue(os.path.exists(self.model_path))
        self.assertTrue(os.path.exists(self.meta_path))

        # Feature_cols skal være rigtig
        self.assertIn("feature1", feature_cols)
        self.assertIn("feature2", feature_cols)

if __name__ == "__main__":
    unittest.main()
