# skdr-eval v0.1.0 Release Notes

🎉 **First stable release of skdr-eval!**

We're excited to announce the initial release of `skdr-eval`, a Python library for offline policy evaluation in service-time minimization scenarios using Doubly Robust (DR) and Stabilized Doubly Robust (SNDR) estimators.

## 🚀 What is skdr-eval?

`skdr-eval` provides robust tools for evaluating policies offline when your goal is to minimize service times. It's particularly useful for:

- **Service optimization**: Evaluate routing policies, resource allocation strategies
- **A/B testing**: Compare policies using historical data without online experiments
- **Causal inference**: Estimate counterfactual performance with confidence intervals
- **Production systems**: Time-aware evaluation with proper train/test splits

## ✨ Key Features

### 🎯 **Doubly Robust Estimation**
- **DR (Doubly Robust)**: Combines outcome modeling and importance sampling for robust estimates
- **SNDR (Stabilized DR)**: Normalized version that reduces variance in challenging scenarios
- **Automatic clipping**: Smart threshold selection to balance bias-variance tradeoff

### ⏰ **Time-Aware Evaluation**
- **Time-series splits**: Respects temporal order in train/test splits
- **Calibrated propensity scores**: Uses Platt scaling for better probability estimates
- **Cross-fitting**: Reduces overfitting bias in outcome model predictions

### 🔧 **Scikit-learn Integration**
- **Easy model evaluation**: Works with any sklearn-compatible estimator
- **Flexible API**: Evaluate single models or compare multiple candidates
- **Built-in models**: Includes HistGradientBoosting and RandomForest defaults

### 📊 **Comprehensive Diagnostics**
- **Effective Sample Size (ESS)**: Measures quality of importance sampling
- **Match rates**: Proportion of samples with valid propensity scores
- **Propensity analysis**: Quantile statistics and distribution insights
- **Confidence intervals**: Moving-block bootstrap for time-series data

## 📦 Installation

```bash
pip install skdr-eval
```

## 🏃‍♂️ Quick Start

```python
import skdr_eval
from sklearn.ensemble import RandomForestRegressor

# Generate synthetic service logs
logs, ops_all, true_q = skdr_eval.make_synth_logs(n=5000, n_ops=5, seed=42)

# Define candidate models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
}

# Evaluate using DR and SNDR
report, detailed_results = skdr_eval.evaluate_sklearn_models(
    logs=logs,
    models=models,
    n_splits=3,
    random_state=42
)

print(report)
```

## 🔍 What's Included

### Core Functions
- `evaluate_sklearn_models()`: Main evaluation function for sklearn models
- `make_synth_logs()`: Generate synthetic service logs for testing
- `EvalDesign`: Data structure for evaluation setup
- `DRResult`: Comprehensive results with diagnostics

### Advanced Features
- **Custom clipping grids**: Configure bias-variance tradeoff
- **Bootstrap confidence intervals**: Statistical uncertainty quantification
- **Policy training modes**: Flexible data splitting strategies
- **Outcome estimator options**: Choose from multiple regression approaches

### Quality Assurance
- **100% type hinted**: Full mypy compatibility
- **Comprehensive tests**: Unit tests and smoke tests included
- **Linted codebase**: Passes all ruff quality checks
- **CI/CD pipeline**: Automated testing and publishing

## 📈 Example Output

The library provides detailed evaluation reports:

```
         model  estimator      V_hat     SE_if       ESS  clip  tail_mass  match_rate
0  RandomForest         DR  12.345678  0.123456  1234.56   5.0       0.12        0.85
1  RandomForest       SNDR  12.234567  0.098765  1456.78  10.0       0.08        0.85
```

Plus detailed diagnostics:
- Propensity score quantiles (q01, q05, q10)
- Clipping threshold selection rationale
- Grid search results for optimal parameters

## 🛠️ Technical Details

### Requirements
- **Python**: 3.9+
- **Dependencies**: numpy, pandas, scikit-learn
- **Optional**: matplotlib (for visualization examples)

### Performance
- Efficient numpy-based computations
- Memory-conscious design for large datasets
- Parallel-friendly sklearn integration

### Methodology
Based on established research in:
- Doubly robust estimation for causal inference
- Importance sampling with clipping
- Time-series cross-validation
- Propensity score calibration

## 🔮 What's Next?

Future releases will include:
- Additional estimators (MAGIC, DM variants)
- Built-in visualization tools
- More synthetic data generators
- Advanced bootstrap methods
- Integration with popular ML frameworks

## 🤝 Contributing

We welcome contributions! Please see our GitHub repository for:
- Issue reporting and feature requests
- Development setup instructions
- Code contribution guidelines
- Documentation improvements

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

Built with modern Python best practices and inspired by the causal inference and offline evaluation research communities.

---

**Ready to get started?** Install with `pip install skdr-eval` and check out our [quickstart example](https://github.com/dgenio/skdr-eval/blob/main/examples/quickstart.py)!

For questions, issues, or feature requests, visit our [GitHub repository](https://github.com/dgenio/skdr-eval).
