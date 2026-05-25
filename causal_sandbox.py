import pandas as pd
import numpy as np
import os
import dowhy
from dowhy import CausalModel


class CausalSandbox:
    """
    The Intervention Simulator: Estimates the causal impact of policy interventions
    on the structural void risk score.
    """

    def __init__(self, data):
        self.data = data

    def simulate_intervention(self, treatment_col, outcome_col):
        """
        Uses DoWhy to estimate the causal effect of a 'treatment' (e.g., adding resources)
        on the 'outcome' (void risk).
        """
        print(f"--- Simulating Causal Impact of {treatment_col} ---")

        # 1. Define the Causal Model (The Directed Acyclic Graph)
        # We assume the void risk is confounded by population and existing access (expected_cases)
        model = CausalModel(
            data=self.data,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=['population', 'expected_cases'],  # Confounders
            effect_modifiers=None
        )

        # 2. Identify the estimand
        identified_estimand = model.identify_effect()

        # 3. Estimate the effect using Linear Regression
        estimate = model.estimate_effect(
            identified_estimand,
            method_name="backdoor.linear_regression"
        )

        print(f"Causal Effect Estimation Complete.")
        print(f"Average Treatment Effect: {estimate.value:.4f}")
        return estimate


if __name__ == "__main__":
    # Test on the results from the pipeline
    results_path = os.path.join("data", "processed", "results_test_run.csv")

    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        # Simulate a dummy intervention variable (0=No resources, 1=Increased resources)
        df['resource_allocation'] = np.random.choice([0, 1], size=len(df))

        sandbox = CausalSandbox(df)
        sandbox.simulate_intervention(treatment_col='resource_allocation', outcome_col='void_risk_score')
    else:
        print("Pipeline results not found. Run main_pipeline.py first to generate data.")