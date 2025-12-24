import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")
sns.set_palette("husl")


@dataclass
class GRIPParameters:
    """Parameters for the GRIP model"""
    delta_min: float = 0.05
    delta_max: float = 0.25
    delta_peak: float = 0.16
    w_L: float = 0.10
    w_R: float = 0.06
    rho: float = 1.5
    sv_max: float = 1.0
    k_adopt: float = 20
    delta_c: float = 0.10
    alpha_impact: float = 3.0
    L_avg: float = 75000


class GRIPModel:
    """Full GRIP Model implementation"""

    def __init__(self, params: GRIPParameters = None):
        self.params = params if params else GRIPParameters()

    # ---------------------------------------------------------
    # 1. SOCIAL VALUE FUNCTION
    # ---------------------------------------------------------
    def social_value_function(self, delta):
        delta = np.atleast_1d(delta)
        sv = np.zeros_like(delta)

        left_mask = delta <= self.params.delta_peak
        sv[left_mask] = self.params.sv_max * np.exp(
            -((delta[left_mask] - self.params.delta_peak) / self.params.w_L) ** 2
        )

        right_mask = delta > self.params.delta_peak
        sv[right_mask] = self.params.sv_max * np.exp(
            -((delta[right_mask] - self.params.delta_peak) / self.params.w_R) ** self.params.rho
        )
        return sv

    # ---------------------------------------------------------
    # 2. IMPACT FUNCTION
    # ---------------------------------------------------------
    def impact_function(self, delta):
        delta = np.atleast_1d(delta)
        return 1 - np.exp(-self.params.alpha_impact * delta)

    # ---------------------------------------------------------
    # 3. ADOPTION PROBABILITY
    # ---------------------------------------------------------
    def adoption_probability(self, delta):
        delta = np.atleast_1d(delta)
        return 1 / (1 + np.exp(-self.params.k_adopt * (delta - self.params.delta_c)))

    # ---------------------------------------------------------
    # 4. FIND OPTIMAL DISCOUNT
    # ---------------------------------------------------------
    def find_optimal_discount(self):
        obj = lambda d: -self.social_value_function(d)[0]

        result = minimize_scalar(
            obj,
            bounds=(self.params.delta_min, self.params.delta_max),
            method="bounded"
        )

        return {
            "optimal_delta": float(result.x),
            "optimal_social_value": float(-result.fun),
            "optimization_success": result.success
        }

    # ---------------------------------------------------------
    # 5. ORIGINAL SYNTHETIC SIMULATION
    # ---------------------------------------------------------
    def simulate_lending_scenarios(self, num_borrowers=1000):
        np.random.seed(42)
        scenarios = []

        optimal = self.find_optimal_discount()
        optimal_delta = optimal["optimal_delta"]

        for i in range(num_borrowers):
            is_rural = np.random.choice([True, False], p=[0.7, 0.3])
            is_shg_member = (
                np.random.choice([True, False], p=[0.65, 0.35]) if is_rural
                else np.random.choice([True, False], p=[0.2, 0.8])
            )

            baseline_rate = 0.12 + np.random.normal(0, 0.02)
            discount_rate = optimal_delta * (1.5 if is_shg_member else 1.0)
            final_rate = max(0.06, baseline_rate - discount_rate)

            impact_score = float(self.impact_function(discount_rate)[0])
            adoption_prob = float(self.adoption_probability(discount_rate)[0])

            loan_amount = self.params.L_avg * (0.8 if is_rural else 1.2) * np.random.uniform(0.7, 1.3)

            scenarios.append({
                "borrower_id": i,
                "is_rural": is_rural,
                "is_shg_member": is_shg_member,
                "baseline_rate": baseline_rate,
                "discount_applied": discount_rate,
                "final_interest_rate": final_rate,
                "loan_amount": loan_amount,
                "impact_score": impact_score,
                "adoption_probability": adoption_prob,
                "savings_inr": loan_amount * discount_rate
            })

        return pd.DataFrame(scenarios)

    # ---------------------------------------------------------
    # 6. **FINDEX CALIBRATED SIMULATION**
    # ---------------------------------------------------------
    def simulate_lending_scenarios_from_findex(
        self,
        num_borrowers,
        findex_csv_path,
        year="2024"
    ):
        """
        New method:
        Borrower behavior is calibrated using real FINDEX borrowing rates
        (Male/Female × Urban/Rural).
        """

        df = pd.read_csv(findex_csv_path)

        india = df[df.get("REF_AREA_LABEL", "") == "India"].copy()
        year_col = str(year)

        if india.empty:
            raise ValueError("India not found in FINDEX file.")

        if year_col not in india.columns:
            raise ValueError(f"Year '{year_col}' not available.")

        borrowed = india[india["INDICATOR_LABEL"].str.contains("Borrow", case=False, na=False)].copy()
        borrowed = borrowed[borrowed["SEX_LABEL"].isin(["Male", "Female"])]
        borrowed = borrowed.dropna(subset=[year_col])

        if "URBANISATION_LABEL" in borrowed.columns:
            borrowed["URB"] = borrowed["URBANISATION_LABEL"].fillna("All")
        else:
            borrowed["URB"] = "All"

        borrowed["segment_key"] = borrowed["URB"] + "_" + borrowed["SEX_LABEL"]
        borrowed["prob"] = borrowed[year_col] / 100

        seg_probs = borrowed.groupby("segment_key")["prob"].mean().to_dict()

        national_avg_prob = max(0.01, borrowed["prob"].mean())

        # Run simulation
        np.random.seed(42)
        results = []

        sexes = ["Male", "Female"]

        optimal = self.find_optimal_discount()
        base_delta = optimal["optimal_delta"]

        for i in range(num_borrowers):

            is_rural = np.random.choice([True, False], p=[0.7, 0.3])
            sex = np.random.choice(sexes)

            urb = "Rural" if is_rural else "Urban"
            seg_key = f"{urb}_{sex}"

            seg_prob = seg_probs.get(seg_key, national_avg_prob)

            is_shg_member = (
                np.random.choice([True, False], p=[0.65, 0.35]) if is_rural
                else np.random.choice([True, False], p=[0.2, 0.8])
            )

            baseline_rate = 0.12 + np.random.normal(0, 0.02)

            discount_rate = base_delta * (1.5 if is_shg_member else 1.0)
            final_rate = max(0.06, baseline_rate - discount_rate)

            base_adopt = float(self.adoption_probability(discount_rate)[0])
            scaled_adopt = base_adopt * (seg_prob / national_avg_prob)
            scaled_adopt = max(0, min(1, scaled_adopt))

            impact = float(self.impact_function(discount_rate)[0])

            loan_amount = self.params.L_avg * (0.8 if is_rural else 1.2) * np.random.uniform(0.7, 1.3)

            results.append({
                "borrower_id": i,
                "sex": sex,
                "segment_key": seg_key,
                "findex_borrow_prob": seg_prob,

                "is_rural": is_rural,
                "is_shg_member": is_shg_member,

                "baseline_rate": baseline_rate,
                "discount_applied": discount_rate,
                "final_interest_rate": final_rate,

                "loan_amount": loan_amount,
                "impact_score": impact,
                "adoption_probability": scaled_adopt,
                "savings_inr": loan_amount * discount_rate
            })

        return pd.DataFrame(results)


# -------------------------------------------------------------------
# ANALYZER
# -------------------------------------------------------------------
class GRIPAnalyzer:
    def __init__(self, model: GRIPModel):
        self.model = model

    def plot_social_value_curve(self):
        params = self.model.params
        delta_range = np.linspace(params.delta_min, params.delta_max, 200)
        sv = self.model.social_value_function(delta_range)
        opt = self.model.find_optimal_discount()

        plt.figure(figsize=(10, 6))
        plt.plot(delta_range, sv, label="Social Value Curve", linewidth=2)
        plt.axvline(opt["optimal_delta"], color="red", linestyle="--", label=f"Optimal δ* = {opt['optimal_delta']:.3f}")
        plt.xlabel("Discount Sensitivity (δ)")
        plt.ylabel("Social Value")
        plt.title("GRIP Social Value Optimization")
        plt.grid(True)
        plt.legend()
        plt.show()

    def analyze_lending_scenarios(self, df):
        plt.figure(figsize=(10, 6))
        sns.histplot(df["final_interest_rate"], kde=True)
        plt.title("Distribution of Final Interest Rates")
        plt.xlabel("Final Rate")
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(df["adoption_probability"], kde=True)
        plt.title("Adoption Probability Distribution")
        plt.xlabel("Adoption Probability")
        plt.show()

