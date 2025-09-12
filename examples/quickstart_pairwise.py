#!/usr/bin/env python3
"""
Quickstart example for pairwise evaluation with skdr-eval.

This example demonstrates:
1. Generating synthetic pairwise data
2. Training sklearn models on pairwise features
3. Running pairwise evaluation with autoscaling
4. Comparing regression and binary classification tasks
"""

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

from skdr_eval import evaluate_pairwise_models, make_pairwise_synth
from skdr_eval.pairwise import PairwiseDesign


def main():
    """Run pairwise evaluation quickstart."""
    print("🚀 skdr-eval Pairwise Evaluation Quickstart")
    print("=" * 50)

    # Generate synthetic pairwise data
    print("\n📊 Generating synthetic pairwise data...")

    # Small dataset for quick demonstration
    logs_df, op_daily_df = make_pairwise_synth(
        n_days=3,
        n_clients_day=500,  # 500 clients per day
        n_ops=20,  # 20 operators
        seed=42,
        binary=False,  # Continuous outcomes (service times)
    )

    print(
        f"Generated {len(logs_df):,} decisions over {len(op_daily_df['arrival_day'].unique())} days"
    )
    print(f"Dataset shape: {logs_df.shape}")
    print(
        f"Operators per day: {len(op_daily_df) // len(op_daily_df['arrival_day'].unique())}"
    )

    # Show data structure
    print("\n📋 Data structure:")
    print("Logs DataFrame columns:", list(logs_df.columns))
    print("Operator DataFrame columns:", list(op_daily_df.columns))

    # Show sample eligibility
    sample_elig = logs_df["elig_mask"].iloc[0]
    print(f"Sample eligibility mask: {len(sample_elig)} operators eligible")

    # Prepare features for model training
    print("\n🔧 Training models on pairwise features...")

    feature_cols = [col for col in logs_df.columns if col.startswith(("cli_", "op_"))]
    X = logs_df[feature_cols].values
    y = logs_df["service_time"].values

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target range: [{y.min():.1f}, {y.max():.1f}] minutes")

    # Create and train models
    models_regression = {
        "Ridge": Ridge(random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }

    print("Training models...")
    for name, model in models_regression.items():
        model.fit(X, y)
        train_score = model.score(X, y)
        print(f"  {name}: R² = {train_score:.3f}")

    # Run pairwise evaluation for regression task
    print("\n🎯 Running pairwise evaluation (regression, minimize service time)...")

    report_reg, detailed_reg = evaluate_pairwise_models(
        logs_df=logs_df,
        op_daily_df=op_daily_df,
        models=models_regression,
        metric_col="service_time",
        task_type="regression",
        direction="min",  # Minimize service time
        n_splits=3,
        strategy="auto",  # Let autoscale choose strategy
        propensity="auto",  # Let system choose propensity method
        random_state=42,
    )

    print("\n📊 Regression Results:")
    print(
        report_reg[["model", "estimator", "V_hat", "ESS", "match_rate", "clip"]].round(
            3
        )
    )

    # Generate binary data for classification example
    print("\n🔄 Generating binary outcome data...")

    logs_binary, op_daily_binary = make_pairwise_synth(
        n_days=2,
        n_clients_day=400,
        n_ops=15,
        seed=123,
        binary=True,  # Binary outcomes (success/failure)
    )

    print(f"Binary dataset: {len(logs_binary):,} decisions")
    success_rate = logs_binary["success"].mean()
    print(f"Overall success rate: {success_rate:.1%}")

    # Train binary classification models
    feature_cols_binary = [
        col for col in logs_binary.columns if col.startswith(("cli_", "op_"))
    ]
    X_binary = logs_binary[feature_cols_binary].values
    y_binary = logs_binary["success"].values

    models_binary = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=42
        ),
    }

    print("Training binary classification models...")
    for name, model in models_binary.items():
        model.fit(X_binary, y_binary)
        train_score = model.score(X_binary, y_binary)
        print(f"  {name}: Accuracy = {train_score:.3f}")

    # Run pairwise evaluation for binary task
    print("\n🎯 Running pairwise evaluation (binary, maximize success rate)...")

    report_binary, _ = evaluate_pairwise_models(
        logs_df=logs_binary,
        op_daily_df=op_daily_binary,
        models=models_binary,
        metric_col="success",
        task_type="binary",
        direction="max",  # Maximize success probability
        n_splits=3,
        strategy="auto",
        propensity="auto",
        random_state=42,
    )

    print("\n📊 Binary Classification Results:")
    print(
        report_binary[
            ["model", "estimator", "V_hat", "ESS", "match_rate", "clip"]
        ].round(3)
    )

    # Show autoscale information
    print("\n⚡ Autoscale Information:")

    design_reg = PairwiseDesign.from_dataframes(logs_df, op_daily_df)
    stats_reg = design_reg.get_stats()

    design_binary = PairwiseDesign.from_dataframes(logs_binary, op_daily_binary)
    stats_binary = design_binary.get_stats()

    print("Regression dataset:")
    print(f"  Candidate pairs: {stats_reg['candidate_pairs']:,}")
    print(f"  Estimated memory: {stats_reg['memory_gb']:.2f} GB")

    print("Binary dataset:")
    print(f"  Candidate pairs: {stats_binary['candidate_pairs']:,}")
    print(f"  Estimated memory: {stats_binary['memory_gb']:.2f} GB")

    # Show detailed results for one model
    print("\n🔍 Detailed Results (Ridge regression):")
    if "Ridge" in detailed_reg:
        ridge_results = detailed_reg["Ridge"]
        for estimator_name, result in ridge_results.items():
            print(f"  {estimator_name}:")
            print(f"    Policy Value: {result.V_hat:.3f}")
            print(f"    Standard Error: {result.SE_if:.3f}")
            print(f"    Effective Sample Size: {result.ESS:.1f}")
            print(f"    Match Rate: {result.match_rate:.3f}")
            print(
                f"    Propensity Score Quantiles: "
                f"[{result.pscore_q01:.4f}, {result.pscore_q05:.4f}, "
                f"{result.pscore_q10:.4f}]"
            )

    print("\n✅ Pairwise evaluation completed successfully!")
    print("\n💡 Key takeaways:")
    print("  • Autoscale automatically chose the optimal strategy")
    print("  • Both regression and binary tasks are supported")
    print("  • Eligibility constraints are properly handled")
    print("  • DR and SNDR estimators provide robust policy evaluation")

    print("\n🔧 Next steps:")
    print("  • Try with your own data using the same API")
    print("  • Experiment with different strategies: 'direct', 'stream', 'stream_topk'")
    print("  • Install optional extras: pip install skdr-eval[choice,speed]")
    print("  • Check the documentation for advanced features")


if __name__ == "__main__":
    main()
