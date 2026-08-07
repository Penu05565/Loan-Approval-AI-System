import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from data_pipeline import preprocess_data

def test_preprocess_runs():
    df = pd.DataFrame({
        "Gender": ["Male"],
        "Married": ["Yes"],
        "ApplicantIncome": [5000]
    })

    result = preprocess_data(df)

    assert result is not None
    assert len(result) == 1