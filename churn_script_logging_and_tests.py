"""
churn_script_logging_and_tests.py

Tests and logging for churn project.

Author: Fernando"""

import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_import():
    """
    test data import
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("test_import: SUCCESS")
    except Exception as err:
        logging.error("test_import: FAILED")
        raise err


def test_eda():
    """
    test eda
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)
        logging.info("test_eda: SUCCESS")
    except Exception as err:
        logging.error("test_eda: FAILED")
        raise err


def test_feature_engineering():
    """
    test feature engineering
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0

        logging.info("test_feature_engineering: SUCCESS")
    except Exception as err:
        logging.error("test_feature_engineering: FAILED")
        raise err


def test_train_models():
    """
    test model training
    """
    try:
        df = cls.import_data("./data/bank_data.csv")
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, "Churn")

        cls.train_models(X_train, X_test, y_train, y_test)

        logging.info("test_train_models: SUCCESS")
    except Exception as err:
        logging.error("test_train_models: FAILED")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda()
    test_feature_engineering()
    test_train_models()