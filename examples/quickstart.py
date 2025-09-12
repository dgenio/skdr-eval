#!/usr/bin/env python3
"""Quickstart example for skdr-eval library."""

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skdr_eval


def main():
    """Run the quickstart example."""
    print("skdr-eval Quickstart Example")
    print("=" * 50)

    # 1. Generate synthetic logs
    print("\n1. Generating synthetic service logs...")
    logs, ops_all, _ = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)

    print(f"   Generated {len(logs)} log entries")
    print(f"   Operators: {list(ops_all)}")
    print(
        f"   Service time range: {logs['service_time'].min():.2f} - {logs['service_time'].max():.2f}"
    )
    print(f"   Mean service time: {logs['service_time'].mean():.2f}")

    # 2. Define candidate models
    print("\n2. Defining candidate models...")
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=100, max_depth=6, random_state=42
        ),
    }

    print(f"   Models to evaluate: {list(models.keys())}")

    # 3. Evaluate models using DR and SNDR
    print("\n3. Evaluating models with DR and SNDR...")
    print("   (This may take a moment...)")

    report, detailed_results = skdr_eval.evaluate_sklearn_models(
        logs=logs,
        models=models,
        fit_models=True,
        n_splits=3,
        outcome_estimator="hgb",
        random_state=42,
        policy_train="pre_split",
        policy_train_frac=0.8,
    )

    # 4. Display results
    print("\n4. Results Summary")
    print("-" * 50)

    # Format and display the report
    display_cols = [
        "model",
        "estimator",
        "V_hat",
        "SE_if",
        "clip",
        "ESS",
        "match_rate",
        "min_pscore",
    ]

    report_display = report[display_cols].copy()

    # Round numeric columns for better display
    numeric_cols = ["V_hat", "SE_if", "ESS", "match_rate", "min_pscore"]
    for col in numeric_cols:
        if col in report_display.columns:
            report_display[col] = report_display[col].round(4)

    print(report_display.to_string(index=False))

    # 5. Show detailed analysis for best model
    print("\n5. Detailed Analysis")
    print("-" * 50)

    # Find best model by DR estimate (lowest service time)
    dr_results = report[report["estimator"] == "DR"]
    best_model_name = dr_results.loc[dr_results["V_hat"].idxmin(), "model"]
    best_dr_value = dr_results.loc[dr_results["V_hat"].idxmin(), "V_hat"]

    print(f"Best model: {best_model_name}")
    print(f"Estimated service time (DR): {best_dr_value:.3f}")

    # Show clipping grid for best model
    best_detailed = detailed_results[best_model_name]
    dr_grid = best_detailed["DR"].grid
    sndr_grid = best_detailed["SNDR"].grid

    print(f"\nClipping analysis for {best_model_name}:")
    print("Clip | DR_Value | SNDR_Value | ESS   | Tail_Mass")
    print("-" * 50)

    for _, row in dr_grid.iterrows():
        sndr_value = sndr_grid[sndr_grid["clip"] == row["clip"]]["V_SNDR"].iloc[0]
        print(
            f"{row['clip']:4.0f} | {row['V_DR']:8.3f} | {sndr_value:10.3f} | "
            f"{row['ESS']:5.1f} | {row['tail_mass']:9.3f}"
        )

    # 6. Policy comparison
    print("\n6. Policy Performance Comparison")
    print("-" * 50)

    # Compare with baseline (mean observed service time)
    baseline_performance = logs["service_time"].mean()
    print(f"Baseline (observed): {baseline_performance:.3f}")

    for model_name in models:
        dr_value = report[
            (report["model"] == model_name) & (report["estimator"] == "DR")
        ]["V_hat"].iloc[0]
        sndr_value = report[
            (report["model"] == model_name) & (report["estimator"] == "SNDR")
        ]["V_hat"].iloc[0]

        dr_improvement = (
            (baseline_performance - dr_value) / baseline_performance
        ) * 100
        sndr_improvement = (
            (baseline_performance - sndr_value) / baseline_performance
        ) * 100

        print(f"{model_name}:")
        print(f"  DR:   {dr_value:.3f} ({dr_improvement:+.1f}% vs baseline)")
        print(f"  SNDR: {sndr_value:.3f} ({sndr_improvement:+.1f}% vs baseline)")

    # 7. Diagnostics
    print("\n7. Evaluation Diagnostics")
    print("-" * 50)

    for model_name in models:
        model_report = report[report["model"] == model_name]
        dr_row = model_report[model_report["estimator"] == "DR"].iloc[0]

        print(f"{model_name}:")
        print(f"  Match rate: {dr_row['match_rate']:.3f}")
        print(f"  Min p-score: {dr_row['min_pscore']:.6f}")
        print(f"  P-score Q01: {dr_row['pscore_q01']:.6f}")
        print(f"  P-score Q05: {dr_row['pscore_q05']:.6f}")
        print(f"  P-score Q10: {dr_row['pscore_q10']:.6f}")
        print(f"  ESS: {dr_row['ESS']:.1f}")

    print("\nQuickstart example completed!")
    print("\nNext steps:")
    print("- Try different models or hyperparameters")
    print("- Experiment with different clip_grid values")
    print("- Use bootstrap confidence intervals with ci_bootstrap=True")
    print("- Explore the detailed_results for more insights")


if __name__ == "__main__":
    main()
