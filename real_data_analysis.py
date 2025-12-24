import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class RealDataAnalyzer:
    """
    Extracts and visualizes India data from Global FINDEX wide dataset.
    Handles gender comparisons for 2021 and 2024.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.summary = None

    # ----------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded dataset: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    # ----------------------------------------------------
    # BUILD SUMMARY (INDIA)
    # ----------------------------------------------------
    def build_indicator_summary(self):
        """Pull India rows and extract 2021–2024 values for important indicators."""

        if self.df is None:
            raise ValueError("Dataset not loaded. Run load_data().")

        # INDIA = "IND"
        india = self.df[self.df["REF_AREA"] == "IND"].copy()

        # Indicators YOU actually have (verified from your file)
        metrics = {
            "WB_FINDEX_FIACCOUNT_T_D": "Account Ownership (%)",
            "WB_FINDEX_MOBILEACCOUNT_T_D": "Mobile Money Account (%)",
            "WB_FINDEX_FIN3": "Borrowed Any (%)",
            "WB_FINDEX_FING2P_FIN": "Digital G2P (%)"
        }

        rows = []

        for ind_code, metric_name in metrics.items():
            sub = india[india["INDICATOR"] == ind_code]

            for sex in ["M", "F"]:
                row = {
                    "metric": metric_name,
                    "sex": sex,
                    "2021": sub[sub["SEX"] == sex]["2021"].mean(),
                    "2024": sub[sub["SEX"] == sex]["2024"].mean(),
                }
                rows.append(row)

        self.summary = pd.DataFrame(rows)
        print("\nSummary table built:")
        print(self.summary)

    # ----------------------------------------------------
    # TABLE FOR 2024
    # ----------------------------------------------------
    def get_2024_table(self):
        if self.summary is None:
            self.build_indicator_summary()
        return self.summary[["metric", "sex", "2024"]]

    # ----------------------------------------------------
    # TABLE FOR 2021
    # ----------------------------------------------------
    def get_2021_table(self):
        if self.summary is None:
            self.build_indicator_summary()
        return self.summary[["metric", "sex", "2021"]]

    # ----------------------------------------------------
    # PLOT 2024
    # ----------------------------------------------------
    def plot_gender_2024(self):
        if self.summary is None:
            self.build_indicator_summary()

        df = self.summary.copy()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="metric",
            y="2024",
            hue="sex"
        )
        plt.title("India 2024 – Gender Comparison of Financial Inclusion Indicators",
                  fontsize=14, fontweight="bold")
        plt.ylabel("Percentage (%)")
        plt.xlabel("")
        plt.xticks(rotation=15)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------
    # PLOT 2021
    # ----------------------------------------------------
    def plot_gender_2021(self):
        if self.summary is None:
            self.build_indicator_summary()

        df = self.summary.copy()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df,
            x="metric",
            y="2021",
            hue="sex"
        )
        plt.title("India 2021 – Gender Comparison of Financial Inclusion Indicators",
                  fontsize=14, fontweight="bold")
        plt.ylabel("Percentage (%)")
        plt.xlabel("")
        plt.xticks(rotation=15)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ----------------------------------------------------
    # EXPORT SUMMARY
    # ----------------------------------------------------
    def export_summary(self, path):
        if self.summary is not None:
            self.summary.to_csv(path, index=False)
            print(f"Saved summary to {path}")
