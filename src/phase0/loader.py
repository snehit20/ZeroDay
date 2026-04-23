import pandas as pd

def load_data():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race",
        "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    train = pd.read_csv(
        r"C:\Users\snehit\Downloads\adult\adult.data",
        names=columns,
        sep=",",
        skipinitialspace=True
    )

    test = pd.read_csv(
        r"C:\Users\snehit\Downloads\adult\adult.test",
        names=columns,
        sep=",",
        skiprows=1,
        skipinitialspace=True
    )

    test["income"] = test["income"].str.replace(".", "", regex=False)

    df = pd.concat([train, test], ignore_index=True)

    return df