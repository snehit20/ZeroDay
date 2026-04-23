from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class ModelHandler:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def train_default_model(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        # Convert categorical to numeric
        X = X.apply(lambda col: col.astype('category').cat.codes)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        return model, X_test

    def predict(self, model, X):
        return model.predict(X)