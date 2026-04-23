import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

class BiasInspector:
    def __init__(self, df, sensitive_features, target):
        self.df = df
        self.sensitive_features = sensitive_features
        self.target = target
        self.distribution_data = {}
        self.disparity_data = {}

    # STEP 2 → Distribution Bias
    def distribution_bias(self):
        print("\n=== Distribution Bias ===")
        for col in self.sensitive_features:
            dist = self.df[col].value_counts(normalize=True) * 100
            print(f"\n{col}:\n{dist}")
            
            # Store distribution data for JSON output
            self.distribution_data[col] = dist.to_dict()

            self.df[col].value_counts().plot(kind='bar')
            plt.title(f"{col} Distribution")
            plt.show()

    # STEP 3 → Outcome Disparity
    def outcome_disparity(self):
        print("\n=== Outcome Disparity ===")
        for col in self.sensitive_features:
            table = pd.crosstab(
                self.df[col],
                self.df[self.target],
                normalize='index'
            ) * 100

            print(f"\n{col}:\n{table}")
            
            # Store disparity data for JSON output
            self.disparity_data[col] = table.to_dict()

    # STEP 4 → Fairness Metrics
    def fairness_metrics(self):
        print("\n=== Fairness Metrics ===")

        metrics = {}

        for col in self.sensitive_features:
            groups = self.df[col].unique()

            if len(groups) < 2:
                continue

            g1, g2 = groups[0], groups[1]

            p1 = self.df[self.df[col] == g1][self.target].mean()
            p2 = self.df[self.df[col] == g2][self.target].mean()

            dp = abs(p1 - p2)
            di = min(p1, p2) / max(p1, p2) if max(p1, p2) > 0 else 0

            metrics[col] = {
                "Demographic Parity": float(dp),
                "Disparate Impact": float(di)
            }

            print(f"\n{col}")
            print("DP:", dp)
            print("DI:", di)

        return metrics
    
    def get_distribution_data(self):
        """Return collected distribution bias data"""
        return self.distribution_data
    
    def get_disparity_data(self):
        """Return collected outcome disparity data"""
        return self.disparity_data