import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize_scalar
import seaborn as sns
from dataclasses import dataclass
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

@dataclass
class GRIPParameters:
    """Parameters for the GRIP model as defined in the paper"""
    delta_min: float = 0.05      # Minimum discount sensitivity
    delta_max: float = 0.25      # Maximum discount sensitivity  
    delta_peak: float = 0.16     # Peak discount for optimal social value
    w_L: float = 0.10           # Left width of social value curve
    w_R: float = 0.06           # Right width of social value curve
    rho: float = 1.5            # Asymmetry factor
    sv_max: float = 1.0         # Maximum social value
    k_adopt: float = 20         # Adoption steepness parameter
    delta_c: float = 0.10       # Critical discount for adoption
    alpha_impact: float = 3.0   # Impact curvature parameter
    L_avg: float = 75000        # Average loan amount (INR)

class GRIPModel:
    """
    Gender-Responsive Impact-linked Pricing (GRIP) Model
    Implements the mathematical model proposed in the paper
    for optimizing green finance lending with gender-responsive pricing.
    """
    def __init__(self, params: GRIPParameters = None):
        self.params = params if params else GRIPParameters()
        
    def social_value_function(self, delta: np.ndarray) -> np.ndarray:
        """Compute the social value function as defined in Equation (2)."""
        delta = np.atleast_1d(delta)
        sv = np.zeros_like(delta)

        # Left side (δ ≤ δ_peak): Gaussian-like growth
        left_mask = delta <= self.params.delta_peak
        sv[left_mask] = self.params.sv_max * np.exp(
            -((delta[left_mask] - self.params.delta_peak) / self.params.w_L) ** 2
        )

        # Right side (δ > δ_peak): Steeper exponential decline
        right_mask = delta > self.params.delta_peak
        sv[right_mask] = self.params.sv_max * np.exp(
            -((delta[right_mask] - self.params.delta_peak) / self.params.w_R) ** self.params.rho
        )
        return sv
    
    def impact_function(self, delta: np.ndarray) -> np.ndarray:
        """Environmental impact score based on discount sensitivity."""
        delta = np.atleast_1d(delta)
        return 1 - np.exp(-self.params.alpha_impact * delta)
    
    def adoption_probability(self, delta: np.ndarray) -> np.ndarray:
        """Adoption probability using sigmoid function."""
        delta = np.atleast_1d(delta)
        return 1 / (1 + np.exp(-self.params.k_adopt * (delta - self.params.delta_c)))
    
    def find_optimal_discount(self) -> Dict:
        """Find optimal discount sensitivity using optimization."""
        objective = lambda delta: -self.social_value_function(delta)[0]

        result = minimize_scalar(
            objective,
            bounds=(self.params.delta_min, self.params.delta_max),
            method='bounded'
        )

        optimal_delta = float(result.x)
        optimal_sv = float(-result.fun)

        return {
            'optimal_delta': optimal_delta,
            'optimal_social_value': optimal_sv,
            'optimization_success': result.success
        }
    
    def simulate_lending_scenarios(self, num_borrowers: int = 1000) -> pd.DataFrame:
        """Simulate lending scenarios for different borrower profiles."""
        np.random.seed(42)  

        scenarios = []
        optimal_result = self.find_optimal_discount()
        optimal_delta = optimal_result['optimal_delta']

        for i in range(num_borrowers):
            is_rural = np.random.choice([True, False], p=[0.7, 0.3])
            is_shg_member = np.random.choice([True, False], p=[0.65, 0.35]) if is_rural else np.random.choice([True, False], p=[0.2, 0.8])
            baseline_rate = 0.12 + np.random.normal(0, 0.02)

            discount_rate = optimal_delta * (1.5 if is_shg_member else 1.0)
            final_rate = max(0.06, baseline_rate - discount_rate)

            impact_score = self.impact_function(discount_rate)[0]
            adoption_prob = self.adoption_probability(discount_rate)[0]

            loan_amount = self.params.L_avg * (0.8 if is_rural else 1.2) * np.random.uniform(0.7, 1.3)

            scenarios.append({
                'borrower_id': i,
                'is_rural': is_rural,
                'is_shg_member': is_shg_member,
                'baseline_rate': baseline_rate,
                'discount_applied': discount_rate,
                'final_interest_rate': final_rate,
                'loan_amount': loan_amount,
                'impact_score': impact_score,
                'adoption_probability': adoption_prob,
                'savings_inr': loan_amount * discount_rate
            })

        return pd.DataFrame(scenarios)

class GRIPAnalyzer:
    """Analysis and visualization tools for GRIP model results."""
    def __init__(self, model: GRIPModel):
        self.model = model
    
    def plot_social_value_curve(self):
        """Plot the social value function and show optimal point."""
        delta_range = np.linspace(self.model.params.delta_min, self.model.params.delta_max, 200)
        sv_values = self.model.social_value_function(delta_range)
        optimal_result = self.model.find_optimal_discount()

        plt.figure(figsize=(12, 8))

        # Main social value curve
        plt.subplot(2, 2, 1)
        plt.plot(delta_range, sv_values, 'b-', linewidth=2.5, label='Social Value Function')
        plt.axvline(optimal_result['optimal_delta'], color='red', linestyle='--',
                   label=f'Optimal δ* = {optimal_result["optimal_delta"]:.3f}')
        plt.axhline(optimal_result['optimal_social_value'], color='red', linestyle=':', alpha=0.7)
        plt.xlabel('Discount Sensitivity (δ)')
        plt.ylabel('Social Value')
        plt.title('GRIP Social Value Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Component functions
        plt.subplot(2, 2, 2)
        impact_values = self.model.impact_function(delta_range)
        adoption_values = self.model.adoption_probability(delta_range)
        plt.plot(delta_range, impact_values, 'g-', label='Impact Score', linewidth=2)
        plt.plot(delta_range, adoption_values, 'orange', label='Adoption Probability', linewidth=2)
        plt.xlabel('Discount Sensitivity (δ)')
        plt.ylabel('Score / Probability')
        plt.title('Component Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Sensitivity analysis
        plt.subplot(2, 2, 3)
        peak_values = [0.12, 0.16, 0.20]
        optimal_deltas = []
        for peak in peak_values:
            temp_params = GRIPParameters()
            temp_params.delta_peak = peak
            temp_model = GRIPModel(temp_params)
            result = temp_model.find_optimal_discount()
            optimal_deltas.append(result['optimal_delta'])

        plt.plot(peak_values, optimal_deltas, 'ro-', linewidth=2, markersize=8)
        plt.xlabel('Peak Position (δ_peak)')
        plt.ylabel('Optimal Discount (δ*)')
        plt.title('Sensitivity Analysis')
        plt.grid(True, alpha=0.3)

        # Cost-benefit visualization
        plt.subplot(2, 2, 4)
        cost_benefit = delta_range * sv_values * self.model.params.L_avg / 1000
        plt.fill_between(delta_range, 0, cost_benefit, alpha=0.3, color='purple')
        plt.plot(delta_range, cost_benefit, 'purple', linewidth=2)
        plt.xlabel('Discount Sensitivity (δ)')
        plt.ylabel('Net Benefit (₹ thousands)')
        plt.title('Cost-Benefit Analysis')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def main():
    print("=" * 60)
    print("GRIP MODEL: Gender-Responsive Impact-Linked Pricing")
    print("Implementation for Green Finance in India")
    print("=" * 60)

    model = GRIPModel()
    analyzer = GRIPAnalyzer(model)

    print("\n1. OPTIMIZATION RESULTS:")
    print("-" * 30)
    optimal_result = model.find_optimal_discount()
    print(f"Optimal Discount Sensitivity (δ*): {optimal_result['optimal_delta']:.4f}")
    print(f"Maximum Social Value: {optimal_result['optimal_social_value']:.4f}")
    print(f"Optimization Success: {optimal_result['optimization_success']}")

    print("\n2. SIMULATING LENDING SCENARIOS:")
    print("-" * 35)
    scenarios = model.simulate_lending_scenarios(num_borrowers=1000)
    print(f"Generated {len(scenarios)} borrower scenarios")

    print("\n3. GENERATING VISUALIZATIONS:")
    print("-" * 35)
    print("Plotting social value optimization curves...")
    analyzer.plot_social_value_curve()

    return model, scenarios, analyzer

if __name__ == "__main__":
    model, scenarios_df, analyzer = main()
