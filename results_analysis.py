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
    
    def prepare_analysis_data(self):
        """Prepare additional calculated fields for analysis"""
        # Create demographic segments
        self.scenarios['demographic'] = self.scenarios.apply(
            lambda row: 'Rural SHG' if row['is_rural'] and row['is_shg_member']
                       else 'Rural Non-SHG' if row['is_rural'] and not row['is_shg_member']
                       else 'Urban SHG' if not row['is_rural'] and row['is_shg_member']
                       else 'Urban Non-SHG', axis=1
        )
        
        # Calculate effectiveness metrics
        self.scenarios['interest_savings_percent'] = (
            self.scenarios['discount_applied'] / self.scenarios['baseline_rate'] * 100
        )
        
        # Risk-adjusted returns
        self.scenarios['risk_adjusted_return'] = (
            self.scenarios['impact_score'] * self.scenarios['adoption_probability']
        )
    
    def generate_key_performance_indicators(self):
        """Generate KPIs for presentation"""
        kpis = {}
        
        # Financial Impact KPIs
        kpis['total_borrowers'] = len(self.scenarios)
        kpis['total_loan_amount'] = self.scenarios['loan_amount'].sum()
        kpis['total_annual_savings'] = self.scenarios['savings_inr'].sum()
        kpis['avg_interest_reduction'] = self.scenarios['discount_applied'].mean()
        kpis['avg_final_rate'] = self.scenarios['final_interest_rate'].mean()
        
        # Gender & Inclusion KPIs
        kpis['rural_representation'] = self.scenarios['is_rural'].mean() * 100
        kpis['shg_representation'] = self.scenarios['is_shg_member'].mean() * 100
        kpis['rural_avg_savings'] = self.scenarios[self.scenarios['is_rural']]['savings_inr'].mean()
        kpis['urban_avg_savings'] = self.scenarios[~self.scenarios['is_rural']]['savings_inr'].mean()
        
        # Environmental & Social KPIs
        kpis['avg_impact_score'] = self.scenarios['impact_score'].mean()
        kpis['avg_adoption_rate'] = self.scenarios['adoption_probability'].mean()
        kpis['high_impact_borrowers'] = (self.scenarios['impact_score'] > 0.8).mean() * 100
        
        return kpis
    
    def comparative_analysis(self):
        """Compare GRIP vs traditional lending outcomes"""
        
        # Simulate traditional lending scenario (without GRIP discounts)
        traditional_rate = 0.12  # Standard rate
        traditional_scenarios = self.scenarios.copy()
        traditional_scenarios['traditional_rate'] = traditional_rate
        traditional_scenarios['traditional_cost'] = (
            traditional_scenarios['loan_amount'] * traditional_rate
        )
        traditional_scenarios['grip_cost'] = (
            traditional_scenarios['loan_amount'] * traditional_scenarios['final_interest_rate']
        )
        traditional_scenarios['cost_savings'] = (
            traditional_scenarios['traditional_cost'] - traditional_scenarios['grip_cost']
        )
        
        comparison = {
            'Traditional Lending': {
                'avg_rate': traditional_rate,
                'total_cost': traditional_scenarios['traditional_cost'].sum(),
                'avg_adoption': 0.3,  # Assumed lower adoption without incentives
                'rural_access': 0.4   # Assumed lower rural access
            },
            'GRIP Model': {
                'avg_rate': self.scenarios['final_interest_rate'].mean(),
                'total_cost': traditional_scenarios['grip_cost'].sum(),
                'avg_adoption': self.scenarios['adoption_probability'].mean(),
                'rural_access': self.scenarios['is_rural'].mean()
            }
        }
        
        comparison['improvement'] = {
            'cost_reduction': (comparison['Traditional Lending']['total_cost'] - 
                             comparison['GRIP Model']['total_cost']) / comparison['Traditional Lending']['total_cost'] * 100,
            'adoption_increase': (comparison['GRIP Model']['avg_adoption'] - 
                                comparison['Traditional Lending']['avg_adoption']) / comparison['Traditional Lending']['avg_adoption'] * 100,
            'rural_access_increase': (comparison['GRIP Model']['rural_access'] - 
                                    comparison['Traditional Lending']['rural_access']) / comparison['Traditional Lending']['rural_access'] * 100
        }
        
        return comparison
    
    def statistical_significance_tests(self):
        """Perform statistical tests to validate findings"""
        results = {}
        
        # Test difference in savings between SHG and non-SHG members
        shg_savings = self.scenarios[self.scenarios['is_shg_member']]['savings_inr']
        non_shg_savings = self.scenarios[~self.scenarios['is_shg_member']]['savings_inr']
        
        t_stat, p_value = stats.ttest_ind(shg_savings, non_shg_savings)
        results['shg_savings_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }
        
        # Test difference in impact scores between rural and urban
        rural_impact = self.scenarios[self.scenarios['is_rural']]['impact_score']
        urban_impact = self.scenarios[~self.scenarios['is_rural']]['impact_score']
        
        t_stat, p_value = stats.ttest_ind(rural_impact, urban_impact)
        results['rural_impact_test'] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Significant difference' if p_value < 0.05 else 'No significant difference'
        }
        
        # Correlation analysis
        correlation_matrix = self.scenarios[['discount_applied', 'impact_score', 
                                           'adoption_probability', 'savings_inr']].corr()
        results['correlations'] = correlation_matrix
        
        return results
    
    def create_presentation_charts(self):
        """Create specific charts optimized for presentations"""
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Chart 1: GRIP Model Impact Overview
        ax1 = plt.subplot(3, 4, 1)
        kpis = self.generate_key_performance_indicators()
        
        categories = ['Interest\nReduction', 'Environmental\nImpact', 'Adoption\nRate', 'Rural\nAccess']
        values = [kpis['avg_interest_reduction']*100, 
                 kpis['avg_impact_score']*100,
                 kpis['avg_adoption_rate']*100,
                 kpis['rural_representation']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars = plt.bar(categories, values, color=colors)
        plt.title('GRIP Model Key Outcomes (%)', fontweight='bold', fontsize=12)
        plt.ylabel('Percentage (%)')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Chart 2: Demographic Breakdown
        ax2 = plt.subplot(3, 4, 2)
        demo_counts = self.scenarios['demographic'].value_counts()
        colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
        
        wedges, texts, autotexts = plt.pie(demo_counts.values, labels=demo_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Borrower Demographics', fontweight='bold', fontsize=12)
        
        # Chart 3: Interest Rate Comparison
        ax3 = plt.subplot(3, 4, 3)
        comparison_data = self.scenarios.groupby('demographic').agg({
            'baseline_rate': 'mean',
            'final_interest_rate': 'mean'
        }).round(4)
        
        x_pos = np.arange(len(comparison_data))
        width = 0.35
        
        plt.bar(x_pos - width/2, comparison_data['baseline_rate']*100, width, 
                label='Traditional Rate', color='lightcoral', alpha=0.8)
        plt.bar(x_pos + width/2, comparison_data['final_interest_rate']*100, width, 
                label='GRIP Rate', color='lightgreen', alpha=0.8)
        
        plt.xlabel('Demographic Groups')
        plt.ylabel('Interest Rate (%)')
        plt.title('Interest Rate Reduction by Demographics', fontweight='bold', fontsize=12)
        plt.xticks(x_pos, comparison_data.index, rotation=45, ha='right')
        plt.legend()
        
        # Chart 4: Cost-Benefit Analysis
        ax4 = plt.subplot(3, 4, 4)
        total_loans = self.scenarios['loan_amount'].sum() / 1000000  # in millions
        total_savings = self.scenarios['savings_inr'].sum() / 1000000  # in millions
        estimated_cost = total_savings * 0.6  # Assume 60% is actual cost to lender
        net_benefit = total_savings - estimated_cost
        
        categories = ['Total Loans', 'Interest Savings', 'Program Cost', 'Net Benefit']