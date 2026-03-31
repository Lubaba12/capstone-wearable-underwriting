# Wearable-Enhanced Insurance Underwriting

**Learning Classifier Methods for Financial Decision Making: Wearable-Enhanced Insurance Underwriting**

Final Year Project — BSc Data Science and Analytics, The Hong Kong Polytechnic University

**Author:** Lubaba Hassan | 22097014D

---

## Overview

This project evaluates whether wearable-derived physical activity features improve machine learning classifiers for insurance underwriting risk stratification. Using NHANES 2003–2004 public health data, four classifiers are compared across three feature scenarios, evaluated on predictive performance, SHAP-based interpretability, and demographic fairness.

### Research Questions

- **RQ1:** Do classifier performance patterns from Wang (2021) generalise to three-class risk classification on NHANES data?
- **RQ2:** What is the incremental value of wearable-derived activity features for underwriting classification?
- **RQ3:** Which classifier/scenario combination best balances performance, interpretability, and demographic fairness?

## Key Findings

| Finding | Detail |
|---------|--------|
| Classifier ranking | XGBoost > Random Forest > Decision Tree > Logistic Regression (matches Wang 2021) |
| Best model | XGBoost on Scenario B — Macro-F1: 0.90, AUC: 0.98 |
| Wearable increment | +0.003 F1 (negligible aggregate improvement) |
| Wearable SHAP contribution | 5.8% of total feature importance in Scenario B |
| Counterfactual impact | Borderline case: activity change produced 16.4pp probability swing, flipping High → Intermediate |
| Top SHAP features | Age (0.854), hypertension (0.744), male (0.713), smoking (0.479), BMI (0.452) |

## Three Feature Scenarios

| Scenario | Features | Participants | Description |
|----------|----------|-------------|-------------|
| A | 13 traditional clinical | 3,388 | Baseline — replicates traditional underwriting |
| B | 13 traditional + 7 wearable | 1,887 | Main contribution — tests wearable enhancement |
| C | 7 wearable only | 1,887 | Supplementary — tests wearable-only feasibility |

## Composite Underwriting Risk Label

Three-class label combining multiple clinical criteria, mirroring real-world underwriting practice:

- **High (2):** FRS ≥ 20% OR diabetes OR (hypertension + additional risk factor)
- **Intermediate (1):** FRS 10–19% OR severe obesity (BMI ≥ 35) OR hypertension alone
- **Low (0):** None of the above

Distribution: Low 39.8% | Intermediate 20.7% | High 39.5%

## Project Structure

```
capstone-wearable-underwriting/
├── data/
│   └── processed/              # Scenario CSVs, model results, mortality data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb    # NHANES loading, label construction, feature engineering
│   ├── 02_eda.ipynb                   # Exploratory data analysis (11 figures)
│   ├── 03_modelling.ipynb             # 4 classifiers × 3 scenarios + reclassification analysis
│   ├── 04_shap_interpretability.ipynb # SHAP analysis + counterfactual simulation
│   ├── 05_fairness.ipynb              # Demographic fairness + calibration curves
│   └── 06_mortality_exploration.ipynb # CDC linked mortality validation (15-year follow-up)
├── reports/
│   └── figures/                # All generated figures
└── README.md
```

## Methodology

### Data Source
NHANES 2003–2004 (CDC) — 12 files, 10,122 participants, filtered to 3,388 adults aged 30+ with complete lab data.

Wearable-proxy features engineered from PAQIAF (Individual Activities File): total MET-hours, mean MET intensity, vigorous/moderate activity counts and ratio, activity category.

### Classifiers
| Classifier | Role | Interpretability |
|------------|------|-----------------|
| Logistic Regression | Linear baseline | High — direct coefficients |
| Decision Tree | Tree baseline | High — visualisable rules |
| Random Forest | Ensemble performance | Medium — feature importance |
| XGBoost | Best performance | Low — requires SHAP |

### Evaluation Framework
- **Performance:** Macro AUC-ROC, Macro F1, Weighted F1, Brier Score, per-class F1
- **Interpretability:** SHAP global importance, per-class analysis, waterfall plots, counterfactual simulation
- **Fairness:** AUC/FPR by age band and gender, Disparate Impact Ratio

## Data Sources

| Source | DOI / URL | Description |
|--------|-----------|-------------|
| NHANES 2003–2004 | [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2003) | Primary dataset (12 files) |
| CDC Linked Mortality | doi:10.15620/cdc:117142 | 15-year mortality follow-up through Dec 2019 |

## Key References

- Wang, Y. (2021). Predictive machine learning for underwriting life and health insurance. *Actuarial Society 2021 Virtual Convention.*
- D'Agostino, R.B. et al. (2008). General cardiovascular risk profile for use in primary care. *Circulation*, 117(6), 743–753.
- National Cholesterol Education Program, ATP III (2001). *JAMA*, 285(19):2486–97.
- American Academy of Actuaries (2009). Risk Classification Issue Brief.

## Setup

```bash
# Clone the repository
git clone https://github.com/Lubaba12/capstone-wearable-underwriting.git
cd capstone-wearable-underwriting

# Create virtual environment
py -m venv venv
venv\Scripts\activate.bat

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap scipy jupyter

# Run notebooks sequentially
# Notebooks 01 → 02 → 03 → 04 → 05 → 06
```

## License

This project is for academic purposes only. NHANES data is public domain (CDC/NCHS).
