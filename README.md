# Predicting-stock-crashes-supervised-10k-reports

Predicting whether a firm will experience a stock price crash (crash_new320Plus) from firm fundamentals, audit characteristics and the textual tone of annual reports.

**Overview**

Stock price crashes are rare, sharp drops in a firm's share price. This project asks whether they can be predicted in advance from information available in a firm's financial statements and its annual report (10-K) text. It compares a random baseline with logistic regression, random forest and a neural network (MLP).

**Dataset**
~**61,800 firm-year observations** (US public companies, 1993 onwards), panel data keyed by gvkey and fyear
**Target**: crash_new320Plus (1 = crash year). Only ~18% of observations are crashes, so the classes are imbalanced

**Features (22):**
Firm fundamentals: market value, total assets, asset/liability/sales growth, market-to-book, ROA, leverage, firm age, Altman Z-score
Audit & risk: Big 4 auditor, modified audit opinion, high-litigation industry
Industry: SIC code
Text of the annual report: file size, word count, and counts of negative, positive, uncertainty and litigious words
ncskew_new_plus (a continuous crash-risk measure) is excluded from modelling because it is derived from the same returns as the target

The data file is not included in this repository (see Setup).

**Methodology**
**Exploration**: checks for duplicates and missing values, descriptive statistics, skewness/kurtosis, distributions, IQR outlier inspection, Kolmogorov-Smirnov normality tests, and Spearman correlations (no variable is normally distributed)
Preprocessing
80/20 train/test split
SIC codes (400+ categories) grouped so rare codes (<11 firms) become "Other", then target (mean) encoded on the training set only
Continuous features standardised, with the scaler fitted on the training set only
Class imbalance: the minority class is randomly oversampled. During cross-validation this happens inside each fold (via an imblearn pipeline) so no duplicated rows leak into the validation folds
Model selection: stratified 5-fold cross-validated grid search, optimising F1 for the crash class
Evaluation: accuracy, per-class F1, AUC, sensitivity and specificity on the held-out test set

**Notebooks**

01_data_exploration.ipynb	
02_baseline.ipynb
03_logistic_regression_validation.ipynb	
04_model_comparison_cv.ipynb


**Key Findings**

All models roughly double the crash-class F1 of the random baseline (≈0.34 vs 0.18), but overall predictive power is modest (AUC ≈ 0.59). Crash risk is hard to predict from these features alone

Accuracy is misleading here: the random baseline has the highest accuracy only because it mostly predicts "no crash". The trained models trade accuracy for catching ~55–65% of crashes
The three models perform very similarly, which suggests the limit comes from the information in the features rather than from model complexity
Deeper random forests overfit badly: CV F1 falls from 0.357 at depth 10 to ~0.09 at depth 30
Regularisation helps the MLP: a single hidden layer with strong L2 (α = 0.1) beat an 8-layer network
Limitations & Next Steps
Time and firm leakage: the split is random, so the same firm (and future years) can appear in both train and test. A time-based split (train on earlier years, test on later years) or grouping by gvkey would give a more realistic estimate
gvkey is a firm identifier and is used as a numeric feature. Removing it would be cleaner
AUC is computed from hard 0/1 predictions. Using predicted probabilities (predict_proba) would give the true ROC AUC and allow tuning the decision threshold
Possible extensions: gradient boosting (XGBoost/LightGBM), feature importance / SHAP analysis, lagged features, and alternatives to oversampling such as class weights or SMOTE
