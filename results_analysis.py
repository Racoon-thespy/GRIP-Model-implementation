import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class GRIPResultsAnalyzer:
    """
    Advanced analysis for GRIP model results specifically for presentation purposes
    """

    def __init__(self, scenarios_df):
        self.scenarios = scenarios_df
        self.prepare_analysis_data()

    # ----------------------------------------------------------
    # PREPARE ADDITIONAL FIELDS
    # ----------------------------------------------------------
    def prepare_analysis_data(self):
        """Prepare additional calculated fields for analysis"""
        
        self.scenarios['demographic'] = self.scenarios.apply(
            lambda row: 'Rural SHG' if row['is_rural'] and row['is_shg_member']
                       else 'Rural Non-SHG' if row['is_rural'] and not row['is_shg_member']
                       else 'Urban SHG' if not row['is_rural'] and row['is_shg_member']
                       else 'Urban Non-SHG',
            axis=1
        )
        
        self.scenarios['interest_savings_percent'] = (
            self.scenarios['discount_applied'] / self.scenarios['baseline_rate'] * 100
        )
        
        self.scenarios['risk_adjusted_return'] = (
            self.scenarios['impact_score'] * self.scenarios['adoption_probability']
        )

    # ----------------------------------------------------------
    # KPIs
    # ----------------------------------------------------------
    def generate_key_performance_indicators(self):
        kpis = {}

        kpis['total_borrowers'] = len(self.scenarios)
        kpis['total_loan_amount'] = self.scenarios['loan_amount'].sum()
        kpis['total_annual_savings'] = self.scenarios['savings_inr'].sum()

        kpis['avg_interest_reduction'] = self.scenarios['discount_applied'].mean()
        kpis['avg_final_rate'] = self.scenarios['final_interest_rate'].mean()

        kpis['rural_representation'] = self.scenarios['is_rural'].mean() * 100
        kpis['shg_representation'] = self.scenarios['is_shg_member'].mean() * 100

        kpis['rural_avg_savings'] = self.scenarios[self.scenarios['is_rural']]['savings_inr'].mean()
        kpis['urban_avg_savings'] = self.scenarios[~self.scenarios['is_rural']]['savings_inr'].mean()

        kpis['avg_impact_score'] = self.scenarios['impact_score'].mean()
        kpis['avg_adoption_rate'] = self.scenarios['adoption_probability'].mean()
        kpis['high_impact_borrowers'] = (self.scenarios['impact_score'] > 0.8).mean() * 100

        return kpis

    # ----------------------------------------------------------
    # COMPARISON VS TRADITIONAL
    # ----------------------------------------------------------
    def comparative_analysis(self):

        traditional_rate = 0.12
        df = self.scenarios.copy()

        df['traditional_rate'] = traditional_rate
        df['traditional_cost'] = df['loan_amount'] * traditional_rate
        df['grip_cost'] = df['loan_amount'] * df['final_interest_rate']
        df['cost_savings'] = df['traditional_cost'] - df['grip_cost']

        comparison = {
            'Traditional Lending': {
                'avg_rate': traditional_rate,
                'total_cost': df['traditional_cost'].sum(),
                'avg_adoption': 0.3,
                'rural_access': 0.4
            },
            'GRIP Model': {
                'avg_rate': df['final_interest_rate'].mean(),
                'total_cost': df['grip_cost'].sum(),
                'avg_adoption': df['adoption_probability'].mean(),
                'rural_access': df['is_rural'].mean()
            }
        }

        comparison['improvement'] = {
            'cost_reduction':
                (comparison['Traditional Lending']['total_cost'] - comparison['GRIP Model']['total_cost'])
                / comparison['Traditional Lending']['total_cost'] * 100,

            'adoption_increase':
                (comparison['GRIP Model']['avg_adoption'] - comparison['Traditional Lending']['avg_adoption'])
                / comparison['Traditional Lending']['avg_adoption'] * 100,

            'rural_access_increase':
                (comparison['GRIP Model']['rural_access'] - comparison['Traditional Lending']['rural_access'])
                / comparison['Traditional Lending']['rural_access'] * 100
        }

        return comparison

    # ----------------------------------------------------------
    # STATISTICS
    # ----------------------------------------------------------
    def statistical_significance_tests(self):
        results = {}

        shg = self.scenarios[self.scenarios['is_shg_member']]['savings_inr']
        non_shg = self.scenarios[~self.scenarios['is_shg_member']]['savings_inr']

        t, p = stats.ttest_ind(shg, non_shg)
        results['shg_savings_test'] = {
            't_statistic': t,
            'p_value': p,
            'significant': p < 0.05
        }

        rural = self.scenarios[self.scenarios['is_rural']]['impact_score']
        urban = self.scenarios[~self.scenarios['is_rural']]['impact_score']

        t2, p2 = stats.ttest_ind(rural, urban)
        results['rural_impact_test'] = {
            't_statistic': t2,
            'p_value': p2,
            'significant': p2 < 0.05
        }

        results['correlations'] = self.scenarios[[
            'discount_applied', 'impact_score', 'adoption_probability', 'savings_inr'
        ]].corr()

        return results

    # ----------------------------------------------------------
    # PRESENTATION CHARTS (UPDATED)
    # ----------------------------------------------------------
    def create_presentation_charts(self):

        plt.style.use("default")
        fig = plt.figure(figsize=(22, 16))
        plt.subplots_adjust(hspace=0.45, wspace=0.35)

        # ------------------------------------------------------
        # CHART 1 — KEY OUTCOMES
        # ------------------------------------------------------
        ax1 = plt.subplot(2, 2, 1)
        kpis = self.generate_key_performance_indicators()

        categories = ["Interest\nReduction", "Environmental\nImpact", "Adoption\nRate", "Rural\nAccess"]
        values = [
            kpis['avg_interest_reduction'] * 100,
            kpis['avg_impact_score'] * 100,
            kpis['avg_adoption_rate'] * 100,
            kpis['rural_representation']
        ]

        bars = ax1.bar(categories, values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"])
        ax1.set_title("GRIP Model Key Outcomes (%)", fontsize=16, fontweight="bold")
        ax1.set_ylabel("Percentage (%)", fontsize=12)
        ax1.tick_params(axis="x", labelsize=12)

        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{value:.1f}%",
                     ha="center", fontsize=12, fontweight="bold")

        # ------------------------------------------------------
        # CHART 2 — DEMOGRAPHICS
        # ------------------------------------------------------
        ax2 = plt.subplot(2, 2, 2)
        demo_counts = self.scenarios['demographic'].value_counts()

        ax2.pie(
            demo_counts.values,
            labels=demo_counts.index,
            autopct="%1.1f%%",
            textprops={"fontsize": 12},
            startangle=90
        )
        ax2.set_title("Borrower Demographics", fontsize=16, fontweight="bold")

        # ------------------------------------------------------
        # CHART 3 — INTEREST RATE COMPARISON
        # ------------------------------------------------------
        ax3 = plt.subplot(2, 2, 3)
        comp = self.scenarios.groupby("demographic").agg({
            "baseline_rate": "mean",
            "final_interest_rate": "mean"
        }).round(4)

        groups = comp.index
        x = np.arange(len(groups))
        width = 0.35

        ax3.bar(x - width/2, comp['baseline_rate']*100, width, label="Traditional Rate", color="#FF9999")
        ax3.bar(x + width/2, comp['final_interest_rate']*100, width, label="GRIP Rate", color="#99FF99")

        ax3.set_title("Interest Rate Reduction by Demographics", fontsize=16, fontweight="bold")
        ax3.set_ylabel("Interest Rate (%)")

        ax3.set_xticks(x)
        ax3.set_xticklabels(groups, rotation=20, ha="right", fontsize=12)

        ax3.legend(fontsize=12)

        # ------------------------------------------------------
        # CHART 4 — COST BENEFIT SUMMARY
        # ------------------------------------------------------
        ax4 = plt.subplot(2, 2, 4)

        total_loans = self.scenarios['loan_amount'].sum() / 1_000_000
        total_savings = self.scenarios['savings_inr'].sum() / 1_000_000
        program_cost = total_savings * 0.6
        net_benefit = total_savings - program_cost

        cats = ["Total Loans", "Interest Savings", "Program Cost", "Net Benefit"]
        vals = [total_loans, total_savings, program_cost, net_benefit]

        bars = ax4.bar(cats, vals, color=["#5DA5DA", "#60BD68", "#F17CB0", "#B2912F"])

        ax4.set_title("Cost–Benefit Summary (₹ Millions)", fontsize=16, fontweight="bold")
        ax4.set_ylabel("₹ (Millions)", fontsize=12)
        ax4.tick_params(axis="x", labelrotation=15, labelsize=12)

        for bar, value in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{value:.2f}",
                     ha="center", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.show()

