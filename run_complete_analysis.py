"""
Master execution file for GRIP Model + Real FINDEX Analysis
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('TkAgg')


def main():
    print(" GRIP MODEL - FULL ANALYSIS")
    print("=" * 60)

    try:
        # ------------------------------
        # IMPORT MODULES
        # ------------------------------
        print("\n Loading modules...")
        from main_implementation import GRIPModel, GRIPAnalyzer
        from results_analysis import GRIPResultsAnalyzer
        from real_data_analysis import RealDataAnalyzer
        print(" Modules loaded!\n")

        # ==========================================================
        # STEP 0: REAL GLOBAL FINDEX EVIDENCE (2021 & 2024)
        # ==========================================================
        print("\n STEP 0: REAL DATA FROM GLOBAL FINDEX")
        print("-" * 60)

        rda = RealDataAnalyzer("WB_FINDEX_WIDEF.csv")
        rda.load_data()
        rda.build_indicator_summary()

        # 2024 table + chart
        print("\n2024 Snapshot (India):")
        print(rda.get_2024_table())
        rda.plot_gender_2024()

        # 2021 table + chart
        print("\n2021 Snapshot (India):")
        print(rda.get_2021_table())
        rda.plot_gender_2021()

        # Export summary
        os.makedirs("presentation_results", exist_ok=True)
        rda.export_summary("presentation_results/real_data_summary.csv")

        # ==========================================================
        # STEP 1: GRIP MODEL OPTIMIZATION
        # ==========================================================
        print("\n STEP 1: GRIP MODEL OPTIMIZATION")
        print("-" * 60)

        model = GRIPModel()
        analyzer = GRIPAnalyzer(model)

        optimal = model.find_optimal_discount()
        print(f" Optimal δ*: {optimal['optimal_delta']:.4f}")
        print(f" Max Social Value: {optimal['optimal_social_value']:.4f}")

        scenarios = model.simulate_lending_scenarios(num_borrowers=1500)
        print(f" Generated {len(scenarios)} borrower scenarios")

        # ==========================================================
        # STEP 2: ADVANCED ANALYSIS
        # ==========================================================
        print("\n STEP 2: ADVANCED GRIP ANALYSIS")
        print("-" * 60)

        results_analyzer = GRIPResultsAnalyzer(scenarios)

        kpis = results_analyzer.generate_key_performance_indicators()
        print("\n Key KPIs:")
        print(kpis)

        comparison = results_analyzer.comparative_analysis()
        print("\n Comparison vs Traditional:")
        print(comparison["improvement"])

        stats = results_analyzer.statistical_significance_tests()
        print("\n Statistical Tests:")
        print(stats)

        # ==========================================================
        # STEP 3: VISUALIZATIONS
        # ==========================================================
        print("\n STEP 3: VISUALIZATIONS")
        print("-" * 60)

        # --- Core GRIP visuals ---
        analyzer.plot_social_value_curve()
        results_analyzer.create_presentation_charts()
        analyzer.analyze_lending_scenarios(scenarios)

        # --- NEW RESEARCH PLOTS ---
        print("\n Additional Research Visuals...")
        results_analyzer.plot_impact_vs_adoption()
        results_analyzer.plot_savings_distribution_by_segment()
        results_analyzer.plot_rural_urban_comparison()
        results_analyzer.plot_grip_vs_traditional_by_segment()

        # ==========================================================
        # STEP 4: SAVE RESULTS
        # ==========================================================
        print("\n STEP 4: SAVING OUTPUTS")
        print("-" * 60)

        results_dir = "presentation_results"
        os.makedirs(results_dir, exist_ok=True)

        scenarios.to_csv(f"{results_dir}/grip_borrower_scenarios.csv", index=False)

        # Save summaries
        import json
        with open(f"{results_dir}/kpis.json", "w") as f:
            json.dump(kpis, f, indent=2)

        with open(f"{results_dir}/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

        with open(f"{results_dir}/stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print("\n ALL DONE! Your results are saved in:")
        print("  → presentation_results/")
        print(" Charts displayed successfully.")

        return {
            "model": model,
            "scenarios": scenarios,
            "results_analyzer": results_analyzer,
            "real_data": rda
        }

    except Exception as e:
        print("\n ERROR:", e)
        return None


# ------------------------------------------------------------
# REQUIREMENTS CHECK
# ------------------------------------------------------------
def check_requirements():
    required = ["numpy", "pandas", "matplotlib", "seaborn", "scipy"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except:
            missing.append(pkg)

    if missing:
        print("Missing packages:", missing)
        print("Install using: pip install " + " ".join(missing))
        return False
    return True


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    print(" Checking requirements...")
    if check_requirements():
        print(" Requirements OK!\n")
        main()
    else:
        print("\n Requirements missing.")
