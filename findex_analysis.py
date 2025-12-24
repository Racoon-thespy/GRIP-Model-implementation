import pandas as pd
import matplotlib.pyplot as plt

# Indicators we care about
FINDEX_INDICATORS = {
    "Account Ownership (%)": "WB_FINDEX_FIACCOUNT_T_D",
    "Mobile Money Account (%)": "WB_FINDEX_MOBILEACCOUNT_T_D",
    "Borrowed Any (%)": "WB_FINDEX_FIN37_39_ACC",
    "Digital G2P (%)": "WB_FINDEX_FING2P_FIN",
}

YEARS = ["2011", "2014", "2017", "2021", "2024"]


# ------------------------------------------------
# 1. LOAD ONLY INDIA + NEEDED INDICATORS & GENDER
# ------------------------------------------------
def load_findex(file_path="WB_FINDEX_WIDEF.csv"):
    chunksize = 5000  # read small chunks
    filtered_rows = []

    needed_indicators = set(FINDEX_INDICATORS.values())
    needed_sex = {"M", "F"}

    for chunk in pd.read_csv(file_path, chunksize=chunksize, low_memory=False):
        chunk_filtered = chunk[
            (chunk["REF_AREA"] == "IND")
            & (chunk["INDICATOR"].isin(needed_indicators))
            & (chunk["SEX"].isin(needed_sex))
        ]
        if len(chunk_filtered):
            filtered_rows.append(chunk_filtered)

    df = pd.concat(filtered_rows, ignore_index=True)
    return df


# ------------------------------------------------
# 2. PRINT SNAPSHOT FOR A SINGLE YEAR
# ------------------------------------------------
def show_year_snapshot(df, year):
    print(f"\n===== FINDEX {year} SNAPSHOT (India) =====")

    for pretty_name, indicator in FINDEX_INDICATORS.items():
        row_m = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "M")]
        row_f = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "F")]

        male_val = row_m[year].iloc[0] if len(row_m) else None
        female_val = row_f[year].iloc[0] if len(row_f) else None

        print(f"{pretty_name}: Male={male_val}, Female={female_val}")

    print("====================================\n")


# ------------------------------------------------
# 3. BAR CHART FOR YEAR
# ------------------------------------------------
def plot_gender_comparison(df, year):
    labels = []
    males = []
    females = []

    for pretty_name, indicator in FINDEX_INDICATORS.items():
        labels.append(pretty_name)

        row_m = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "M")]
        row_f = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "F")]

        males.append(row_m[year].iloc[0] if len(row_m) else 0)
        females.append(row_f[year].iloc[0] if len(row_f) else 0)

    x = range(len(labels))
    plt.figure(figsize=(10, 6))

    plt.bar([p - 0.2 for p in x], males, width=0.4, label="Male", color="steelblue")
    plt.bar([p + 0.2 for p in x], females, width=0.4, label="Female", color="lightcoral")

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Percentage (%)")
    plt.title(f"India – Gender Comparison ({year})")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------
# 4. LONG-TERM TREND (2011 → 2024)
# ------------------------------------------------
def gender_gap_trend(df):
    for pretty_name, indicator in FINDEX_INDICATORS.items():
        plt.figure(figsize=(10, 6))

        row_m = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "M")]
        row_f = df[(df["INDICATOR"] == indicator) & (df["SEX"] == "F")]

        male_vals = [row_m[year].iloc[0] if len(row_m) else None for year in YEARS]
        female_vals = [row_f[year].iloc[0] if len(row_f) else None for year in YEARS]

        plt.plot(YEARS, male_vals, marker="o", label="Male", linewidth=2)
        plt.plot(YEARS, female_vals, marker="o", label="Female", linewidth=2)

        plt.title(f"India – {pretty_name} Trend (2011 → 2024)")
        plt.xlabel("Year")
        plt.ylabel("Percentage (%)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
