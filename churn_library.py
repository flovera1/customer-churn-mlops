"""
churn_library.py
Library of functions for customer churn prediction.
Fernando
"""

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve, auc
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def import_data(pth):
    """
    returns dataframe for the csv found at pth
    """
    try:
        df = pd.read_csv(pth)
        logging.info("Data imported successfully")
        return df
    except FileNotFoundError as err:
        logging.error("File not found")
        raise err


def perform_eda(df):
    """
    perform eda on df and save figures
    """
    os.makedirs('./images/eda', exist_ok=True)

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # churn distribution
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_distribution.png')
    plt.close()

    # age distribution
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.close()

    # heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=False, cmap='Dark2_r')
    plt.savefig('./images/eda/heatmap.png')
    plt.close()

    logging.info("EDA completed")


def encoder_helper(df, category_lst, response):
    """
    encode categorical columns
    """
    for cat in category_lst:
        means = df.groupby(cat)[response].mean()
        df[cat + '_' + response] = df[cat].map(means)

    logging.info("Encoding completed")
    return df

def perform_feature_engineering(df, response):
    """
    feature engineering and split
    """

    # create response variable
    df[response] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )

    # categorical columns
    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # encode
    df = encoder_helper(df, cat_cols, response)

    # KEEP ONLY NUMERIC + ENCODED COLUMNS
    encoded_cols = [col + "_" + response for col in cat_cols]

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # remove response if present
    if response in num_cols:
        num_cols.remove(response)

    X = df[num_cols + encoded_cols]
    y = df[response]

    return train_test_split(X, y, test_size=0.3, random_state=42)

def classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf):
    """
    save classification reports
    """
    os.makedirs('./images/results', exist_ok=True)

    plt.rc('figure', figsize=(8, 6))

    for model, y_tr, y_te, name in [
        ("Logistic Regression", y_train_preds_lr, y_test_preds_lr, "logistic"),
        ("Random Forest", y_train_preds_rf, y_test_preds_rf, "rf")
    ]:
        plt.text(
            0.01,
            1.25,
            f'{model} Train\n{classification_report(y_train, y_tr)}'
        )
        plt.text(
            0.01,
            0.05,
            f'{model} Test\n{classification_report(y_test, y_te)}'
        )
        plt.axis('off')
        plt.savefig(f'./images/results/{name}_results.png')
        plt.close()

    logging.info("Classification reports saved")


def feature_importance_plot(model, X_data, output_pth):
    """
    feature importance plot
    """
    os.makedirs('./images/results', exist_ok=True)

    importances = model.feature_importances_
    indices = importances.argsort()[::-1]

    plt.figure(figsize=(10, 5))
    plt.title("Feature Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), X_data.columns[indices], rotation=90)

    plt.savefig(output_pth)
    plt.close()

    logging.info("Feature importance saved")

    

def train_models(X_train, X_test, y_train, y_test):
    """
    train models and save outputs
    """
    os.makedirs('./models', exist_ok=True)

    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=3000)

    rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = rfc.predict(X_train)
    y_test_preds_rf = rfc.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
    )

    feature_importance_plot(
        rfc,
        X_test,
        './images/results/feature_importances.png'
    )

    pd.to_pickle(rfc, './models/rfc_model.pkl')
    pd.to_pickle(lrc, './models/logistic_model.pkl')

    logging.info("Models trained and saved")

    plot_roc_curve(
    {
        "Logistic Regression": lrc,
        "Random Forest": rfc
    },
    X_test,
    y_test
    )


def plot_roc_curve(models, X_test, y_test):
    """
    plots ROC curves for multiple models

    input:
        models: dict of trained models
        X_test: test features
        y_test: test labels
    output:
        None (saves plot)
    """
    import os
    os.makedirs('./images/results', exist_ok=True)

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        y_probs = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")

    plt.savefig('./images/results/roc_curve.png')
    plt.close()

if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    perform_eda(df)
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(X_train, X_test, y_train, y_test)