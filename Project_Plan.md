# FINTECH 540: ML Project Plan

**Project:** Forecasting Volatility and Detecting Abnormal Regimes in DJIA Stocks
Last Updated: 2025-11-01

This document outlines the step-by-step plan for our FINTECH 540 project, based on the dataset provided and specific feedback from the professor.

---

## **Phase 1: Data Restructuring & Feature Engineering**

**Goal:** Transform our data from separate Excel sheets into a single, unified "panel" DataFrame. This new structure (one row per date per stock) is essential for building per-stock models.

* **Task 1.1: Update the Data Loader**
  * **What:** Modify `scripts/data_loader.py`. Add a new function `create_panel_data()` that loads all 5-minute volatility sheets (`RV_5`, `BPV_5`, `Good_5`, `Bad_5`, `RQ_5`), "stacks" them to create a `(Date, Stock)` MultiIndex, and merges them into a single panel DataFrame.
  * **Why:** We must analyze each stock individually. This "long" data format is the standard for panel data analysis.

* **Task 1.2: Create a New Feature Engineering Notebook**
  * **What:** Create `notebooks/02_feature_engineering.ipynb`.
  * **Why:** This notebook will be the single source of truth for creating our final model-ready dataset.

* **Task 1.3: Integrate External Data (VIX)**
  * **What:** In the new notebook, use `VIX_History.csv` from the `/data` folder. Merge this into the panel DataFrame.
  * **Why:** To test the professor's hypothesis that this market-wide "fear index" helps predict per-stock volatility.

* **Task 1.4: Define Target Variable (Y_reg)**
  * **What:** In our panel DataFrame, create the regression target `Y_reg` by shifting the `rv` column backward by one day (`.shift(-1)`), grouped by stock.
  * **Why:** This prevents data leakage. We must use "today's" features to predict "tomorrow's" volatility.

* **Task 1.5: Engineer Features (X)**
  * **What:** Create our feature set (`X`) using lagged and rolling data. **All operations must be grouped by stock** (`.groupby('Stock')`).
  * **Why:** This creates the "today" data used to predict the "tomorrow" targets.
  * **Sub-Tasks:**
    * **HAR Features:** Create the baseline model features: `rv_lag_1`, `rv_rolling_5`, `rv_rolling_22`.
    * **Advanced Features:** Create a richer set: Lags of all other variables (`bpv_lag_1`, `good_lag_1`, `bad_lag_1`, `rq_lag_1`, `vix_lag_1`) and key ratios (e.g., `bad_good_ratio_lag_1`).

---

### **Phase 2: Baseline Model (HAR) & Validation Setup**

**Goal:** Establish the performance benchmark to beat, as required by the professor. This model defines "normal" volatility.

* **Task 2.1: Create a New Modeling Notebook**
  * **What:** Create `notebooks/03_model_building.ipynb`. Load the feature-engineered panel data from Phase 1.
  * **Why:** This notebook will contain our model training, evaluation, and comparison.

* **Task 2.2: Split Your Data (Rolling Window Approach)**
  * **What:** Instead of a single train/test split, we will use a **rolling window** (or "expanding window") for validation, as suggested by the professor. We can use `sklearn.model_selection.TimeSeriesSplit`.
  * **Why:** This is a more robust method for time-series data. It simulates how a model would have *actually* performed in practice (e.g., train on 2003-2010, predict 2011; then train on 2003-2011, predict 2012).

* **Task 2.3: Train the HAR Baseline Model**
  * **What:** Train a `sklearn.linear_model.LinearRegression` model using only the HAR features (`rv_lag_1`, `rv_rolling_5`, `rv_rolling_22`) to predict `Y_reg`.
  * **Why:** This creates the simple benchmark model specified in the project guidelines.

* **Task 2.4: Evaluate the Baseline**
  * **What:** Get predictions on the test set(s) from your rolling window split. Calculate the average **Mean Squared Error (MSE)** across all splits.
  * **Why:** This gives us the robust benchmark MSE. The primary goal of our advanced regression model is to achieve a *lower* MSE on the same validation splits.

---

### **Phase 3: Defining "Abnormal Volatility" (Isolation Forest)**

**Goal:** Use the **Isolation Forest**, as recommended by the professor in our direct feedback, to programmatically define "abnormal" volatility for each stock.

* **Task 3.1: Calculate Residuals (Errors)**
  * **What:** Use the trained HAR model from Phase 2 to get predictions for the **entire dataset**. Calculate the error: `residuals = actual_rv - har_predicted_rv`.
  * **Why:** These residuals represent the "unexplained" volatility that the simple model *could not* predict. This is the perfect input for an anomaly detector.

* **Task 3.2: Train the Isolation Forest**
  * **What:** Train a `sklearn.ensemble.IsolationForest` on the `residuals` from the **training set only**.
  * **Why:** We must train the anomaly detector on our "normal" training history to avoid biasing it with future crises.

* **Task 3.3: Create Anomaly Labels (Y_class)**
  * **What:** Use the trained Isolation Forest to `.predict()` on the residuals for the *entire dataset*. This will tag days as normal (`1`) or abnormal (`-1`). Create a new column `is_abnormal` (with `0` for normal, `1` for abnormal).
  * **Why:** This is our new classification target. We have successfully used an ML model to define "abnormal volatility."
  * **Note on Alternative Method:** The second checkpoint also suggested **expanding-window quantiles** to define regimes. This is a simpler, valid alternative we can also test if the Isolation Forest proves too complex or does not yield good results.

---

### **Phase 4: Advanced ML Modeling & Interpretation**

**Goal:** Use our full feature set to (1) beat the HAR model's MSE, (2) predict the anomalies we just defined, and (3) interpret the results.

* **Task 4.1: Advanced Regression (XGBoost)**
  * **What:** Train an `XGBRegressor` model using our **full feature set (X)** from Phase 1 to predict `Y_reg`. Evaluate using the same rolling window split from Task 2.2.
  * **Why:** To directly compare a complex, non-linear model against the simple linear HAR model on a pure forecasting task.
  * **Deliverable:** A clear comparison: `HAR Avg. MSE vs. XGBoost Avg. MSE`.

* **Task 4.2: Anomaly Classification (XGBoost)**
  * **What:** Train an `XGBClassifier` model using our **full feature set (X)**. The target will be `Y_class_shifted = df['is_abnormal'].shift(-1)`.
  * **Why:** To achieve the main project goal: predicting if *tomorrow* will be an "abnormal" volatility day.
  * **Deliverable:** A classification report and confusion matrix, focusing on **Precision** and **Recall** for the "Abnormal" class.

* **Task 4.3: Model Interpretation (SHAP)**
  * **What:** Apply the **SHAP** library to our trained `XGBRegressor` and `XGBClassifier` models, as suggested by the professor.
  * **Why:** To understand *which* features are driving the predictions (e.g., "Is VIX the most important feature?"). This adds critical depth and interpretability to our report.
  * **Deliverable:** SHAP summary plots (feature importance) for our report.

---

### **Phase 5: Exploratory Regime Analysis (HMM)**

**Goal:** Address the professor's direct feedback to "check the Hidden Markov Model" for regime switching.

* **Task 5.1: Create a New HMM Notebook**
  * **What:** Create `notebooks/04_hmm_analysis.ipynb`.
  * **Why:** This is a separate, exploratory analysis that adds significant depth to our report.

* **Task 5.2: Fit an HMM Model**
  * **What:** Pick one representative stock (e.g., 'AAPL') and fit a `GaussianHMM` (from the `hmmlearn` package) with `n_components=3` on its `rv` time series.
  * **Why:** The HMM will automatically find 3 "hidden" states (regimes) which we can label as `Low-Vol`, `Mid-Vol`, and `High-Vol`.

* **Task 5.3: Visualize and Compare**
  * **What:** Create a plot of the 'AAPL' `rv` time series, coloring the plot's background based on the HMM regime.
  * **Why:** This is a powerful visualization for our final presentation. We can then compare the HMM's `High-Vol` regime days to our `is_abnormal` days from the Isolation Forest to cross-validate our findings.

---

### **Phase 6: Final Deliverables**

**Goal:** Assemble our findings into the final report and presentation.

* **Task 6.1: Write the 4-page Report**
  * **What:** Structure the report around this plan: 1) Introduction, 2) Baseline HAR Model & Validation, 3) Defining Abnormality (Isolation Forest), 4) Advanced Model Results (Regression & Classification), 5) **Model Interpretation (SHAP)**, 6) HMM Analysis, 7) Conclusion.

* **Task 6.2: Create the 20-minute Slide Deck**
  * **What:** Create slides for each key part of our analysis. The HMM visualization, the MSE comparison, and the **SHAP feature importance plots** will be critical slides.

* **Task 6.3: Clean and Commit All Code**
  * **What:** Ensure all notebooks (`02_feature_engineering.ipynb`, `03_model_building.ipynb`, `04_hmm_analysis.ipynb`) and scripts (`data_loader.py`) are clean, commented, and pushed to GitHub.
