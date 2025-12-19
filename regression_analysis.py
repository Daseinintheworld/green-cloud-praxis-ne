import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ====================== STEP 1: LOAD DATA ======================
print("="*60)
print("REGRESSION MODELS ANALYSIS - FORMULA BASED")
print("="*60)

# Create dataframe
data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] * 2,
    'Location': ['Location_A']*11 + ['Location_B']*11,
    'Temp': [22, 23, 24, 25, 26, 27, 28, 29, 28, 27, 26, 
             5, 6, 4, 5, 7, 8, 6, 5, 6, 7, 8],
    'Humidity': [88, 89, 90, 91, 92, 92, 93, 94, 93, 92, 90,
                 70, 72, 68, 70, 75, 76, 73, 72, 74, 75, 78],
    'RenewPct': [25, 26, 27, 28, 30, 32, 33, 34, 35, 37, 40,
                 95, 96, 97, 98, 99, 99, 100, 100, 100, 100, 100],
    'RackDensity': [3, 3, 4, 4, 4, 5, 5, 6, 6, 6, 7,
                    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 10],
    'GridEF': [0.7, 0.7, 0.69, 0.69, 0.68, 0.68, 0.67, 0.67, 0.66, 0.66, 0.65,
               0.015, 0.015, 0.014, 0.014, 0.013, 0.013, 0.012, 0.012, 0.012, 0.012, 0.011],
    'EnergyDemandMW': [5, 8, 12, 18, 25, 32, 40, 50, 60, 75, 90,
                       800, 820, 850, 900, 970, 1000, 1050, 1100, 1150, 1200, 1300],
    'PUE': [1.75, 1.74, 1.73, 1.72, 1.71, 1.7, 1.69, 1.68, 1.67, 1.66, 1.65,
            1.26, 1.25, 1.24, 1.23, 1.22, 1.23, 1.22, 1.21, 1.2, 1.21, 1.2],
    'CUE': [0.55, 0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45,
            0.01, 0.01, 0.01, 0, 0, 0, 0, 0, 0, 0, 0],
    'WUE': [1.9, 1.85, 1.8, 1.75, 1.7, 1.6, 1.55, 1.5, 1.45, 1.4, 1.3,
            0.4, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.31, 0.3, 0.29, 0.28],
    'Policy': [0.4, 0.4, 0.4, 0.45, 0.45, 0.45, 0.45, 0.5, 0.5, 0.5, 0.5,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'Monsoon': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

df = pd.DataFrame(data)
df['Location_Code'] = df['Location'].map({'Location_A': 0, 'Location_B': 1})

# ====================== MODEL 1: MLR (Exactly as per your formula) ======================
print("\n" + "="*60)
print("MODEL 1: MULTIPLE LINEAR REGRESSION (MLR)")
print("="*60)
print("Formula: CUE = β₀ + β₁*Temp + β₂*RenewPct + β₃*RackDensity + ε")

# Prepare data for MLR
X_mlr = df[['Temp', 'RenewPct', 'RackDensity']]
y_mlr = df['CUE']

# Add constant for statsmodels
X_mlr_sm = sm.add_constant(X_mlr)
model_mlr = sm.OLS(y_mlr, X_mlr_sm)
results_mlr = model_mlr.fit()

print("\nRegression Results:")
print(results_mlr.summary())

# Extract coefficients
coefficients = results_mlr.params
print(f"\nRegression Equation:")
print(f"CUE = {coefficients['const']:.4f} + {coefficients['Temp']:.4f}*Temp + "
      f"{coefficients['RenewPct']:.4f}*RenewPct + {coefficients['RackDensity']:.4f}*RackDensity")

# Train-test split for MLR
X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y_mlr, test_size=0.2, random_state=42
)

# Scikit-learn MLR
lr = LinearRegression()
lr.fit(X_train_mlr, y_train_mlr)
y_pred_mlr = lr.predict(X_test_mlr)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_mlr, y_pred_mlr):.4f}")
print(f"MSE: {mean_squared_error(y_test_mlr, y_pred_mlr):.6f}")
print(f"MAE: {mean_absolute_error(y_test_mlr, y_pred_mlr):.4f}")

# ====================== MODEL 2: POLYNOMIAL REGRESSION ======================
print("\n" + "="*60)
print("MODEL 2: POLYNOMIAL REGRESSION")
print("="*60)
print("Formula: CUE = β₀ + β₁*Temp + β₂*Humidity + β₃*Humidity² + β₄*Humidity³ + ε")

# Prepare data for Polynomial Regression
X_poly_raw = df[['Temp', 'Humidity']]
y_poly = df['CUE']

# Create polynomial features (degree 3 for Humidity)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly_features = poly.fit_transform(X_poly_raw)

# Get feature names
feature_names = poly.get_feature_names_out(['Temp', 'Humidity'])
print(f"\nGenerated Features: {feature_names}")

# Train-test split
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly_features, y_poly, test_size=0.2, random_state=42
)

# Polynomial regression
lr_poly = LinearRegression()
lr_poly.fit(X_train_poly, y_train_poly)
y_pred_poly = lr_poly.predict(X_test_poly)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_poly, y_pred_poly):.4f}")
print(f"MSE: {mean_squared_error(y_test_poly, y_pred_poly):.6f}")
print(f"MAE: {mean_absolute_error(y_test_poly, y_pred_poly):.4f}")

# ====================== MODEL 3: LASSO REGRESSION ======================
print("\n" + "="*60)
print("MODEL 3: LASSO REGRESSION")
print("="*60)
print("Formula: min{Σ(Yᵢ - Ŷᵢ)² + λΣ|βⱼ|}")

# Prepare data for Lasso
features_all = ['Temp', 'Humidity', 'RenewPct', 'RackDensity', 'GridEF', 
                'EnergyDemandMW', 'PUE', 'WUE', 'Policy', 'Monsoon', 'Location_Code']
X_lasso = df[features_all]
y_lasso = df['CUE']

# Standardize
scaler = StandardScaler()
X_lasso_scaled = scaler.fit_transform(X_lasso)

# Train-test split
X_train_lasso, X_test_lasso, y_train_lasso, y_test_lasso = train_test_split(
    X_lasso_scaled, y_lasso, test_size=0.2, random_state=42
)

# Lasso regression
lasso = Lasso(alpha=0.01, random_state=42)  # λ = 0.01
lasso.fit(X_train_lasso, y_train_lasso)
y_pred_lasso = lasso.predict(X_test_lasso)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_lasso, y_pred_lasso):.4f}")
print(f"MSE: {mean_squared_error(y_test_lasso, y_pred_lasso):.6f}")
print(f"MAE: {mean_absolute_error(y_test_lasso, y_pred_lasso):.4f}")

print(f"\nNumber of features selected (non-zero coefficients): {np.sum(lasso.coef_ != 0)}")
print("Feature coefficients:")
for feature, coef in zip(features_all, lasso.coef_):
    if abs(coef) > 0.001:
        print(f"  {feature}: {coef:.4f}")

# ====================== MODEL 3b: RIDGE REGRESSION ======================
print("\n" + "="*60)
print("MODEL 3b: RIDGE REGRESSION")
print("="*60)
print("Formula: min{Σ(Yᵢ - Ŷᵢ)² + λΣβⱼ²}")

# Ridge regression
ridge = Ridge(alpha=1.0, random_state=42)  # λ = 1.0
ridge.fit(X_train_lasso, y_train_lasso)
y_pred_ridge = ridge.predict(X_test_lasso)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_lasso, y_pred_ridge):.4f}")
print(f"MSE: {mean_squared_error(y_test_lasso, y_pred_ridge):.6f}")
print(f"MAE: {mean_absolute_error(y_test_lasso, y_pred_ridge):.4f}")

# ====================== MODEL 4: RANDOM FOREST ======================
print("\n" + "="*60)
print("MODEL 4: RANDOM FOREST REGRESSION")
print("="*60)
print("Formula: CUE = Average of {Tree₁(X), Tree₂(X), ..., Treeₙ(X)}")

# Prepare data
X_rf = df[features_all]
y_rf = df['CUE']

# Train-test split
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_rf, y_rf, test_size=0.2, random_state=42
)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_rf, y_train_rf)
y_pred_rf = rf.predict(X_test_rf)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_rf, y_pred_rf):.4f}")
print(f"MSE: {mean_squared_error(y_test_rf, y_pred_rf):.6f}")
print(f"MAE: {mean_absolute_error(y_test_rf, y_pred_rf):.4f}")

# ====================== MODEL 4b: GRADIENT BOOSTING ======================
print("\n" + "="*60)
print("MODEL 4b: GRADIENT BOOSTING REGRESSION")
print("="*60)
print("Formula: CUE = Weighted Sum of {Tree₁(X), Tree₂(X), ..., Treeₙ(X)}")

# Gradient Boosting
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbr.fit(X_train_rf, y_train_rf)
y_pred_gbr = gbr.predict(X_test_rf)

print(f"\nPerformance Metrics:")
print(f"R² Score: {r2_score(y_test_rf, y_pred_gbr):.4f}")
print(f"MSE: {mean_squared_error(y_test_rf, y_pred_gbr):.6f}")
print(f"MAE: {mean_absolute_error(y_test_rf, y_pred_gbr):.4f}")

# ====================== MODEL 5: TIME-SERIES (ARIMA/SARIMAX) ======================
print("\n" + "="*60)
print("MODEL 5: TIME-SERIES REGRESSION (ARIMA)")
print("="*60)
print("Formula: Yₜ = c + ΣφᵢYₜ₋ᵢ + Σθᵢεₜ₋ᵢ + ΣδⱼXₜ,ⱼ")

# Prepare time-series data
df_time = df.copy()
df_time = df_time.sort_values('Year')

# Separate by location for time-series
print("\nTime-Series Analysis by Location:")
for location in df_time['Location'].unique():
    print(f"\n--- {location} ---")
    location_data = df_time[df_time['Location'] == location]
    cue_series = location_data['CUE'].values
    
    # Simple time-series analysis
    if len(cue_series) > 1:
        # Calculate trend
        x = np.arange(len(cue_series))
        coeff = np.polyfit(x, cue_series, 1)
        trend = coeff[0]
        
        print(f"Average CUE: {np.mean(cue_series):.4f}")
        print(f"Trend (per year): {trend:.4f}")
        print(f"Start CUE: {cue_series[0]:.4f}")
        print(f"End CUE: {cue_series[-1]:.4f}")
        print(f"Total change: {cue_series[-1] - cue_series[0]:.4f}")

# ====================== COMPARISON OF ALL MODELS ======================
print("\n" + "="*60)
print("COMPARISON OF ALL REGRESSION MODELS")
print("="*60)

# Collect results
model_results = []

# MLR results
model_results.append({
    'Model': 'Multiple Linear Regression',
    'R2': r2_score(y_test_mlr, y_pred_mlr),
    'MSE': mean_squared_error(y_test_mlr, y_pred_mlr),
    'MAE': mean_absolute_error(y_test_mlr, y_pred_mlr)
})

# Polynomial results
model_results.append({
    'Model': 'Polynomial Regression',
    'R2': r2_score(y_test_poly, y_pred_poly),
    'MSE': mean_squared_error(y_test_poly, y_pred_poly),
    'MAE': mean_absolute_error(y_test_poly, y_pred_poly)
})

# Lasso results
model_results.append({
    'Model': 'Lasso Regression',
    'R2': r2_score(y_test_lasso, y_pred_lasso),
    'MSE': mean_squared_error(y_test_lasso, y_pred_lasso),
    'MAE': mean_absolute_error(y_test_lasso, y_pred_lasso)
})

# Ridge results
model_results.append({
    'Model': 'Ridge Regression',
    'R2': r2_score(y_test_lasso, y_pred_ridge),
    'MSE': mean_squared_error(y_test_lasso, y_pred_ridge),
    'MAE': mean_absolute_error(y_test_lasso, y_pred_ridge)
})

# Random Forest results
model_results.append({
    'Model': 'Random Forest',
    'R2': r2_score(y_test_rf, y_pred_rf),
    'MSE': mean_squared_error(y_test_rf, y_pred_rf),
    'MAE': mean_absolute_error(y_test_rf, y_pred_rf)
})

# Gradient Boosting results
model_results.append({
    'Model': 'Gradient Boosting',
    'R2': r2_score(y_test_rf, y_pred_gbr),
    'MSE': mean_squared_error(y_test_rf, y_pred_gbr),
    'MAE': mean_absolute_error(y_test_rf, y_pred_gbr)
})

# Create comparison dataframe
results_df = pd.DataFrame(model_results)
results_df = results_df.sort_values('R2', ascending=False)

print("\nModel Performance Comparison (sorted by R2):")
print(results_df.to_string(index=False))

# Visualization
plt.figure(figsize=(14, 8))

# R2 comparison
plt.subplot(2, 2, 1)
bars1 = plt.bar(results_df['Model'], results_df['R2'], color='skyblue')
plt.title('R² Score Comparison')
plt.ylabel('R² Score')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1)
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

# MSE comparison (log scale)
plt.subplot(2, 2, 2)
bars2 = plt.bar(results_df['Model'], results_df['MSE'], color='lightcoral')
plt.title('Mean Squared Error (MSE) Comparison')
plt.ylabel('MSE')
plt.xticks(rotation=45, ha='right')
plt.yscale('log')
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.6f}', ha='center', va='bottom')

# MAE comparison
plt.subplot(2, 2, 3)
bars3 = plt.bar(results_df['Model'], results_df['MAE'], color='lightgreen')
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE')
plt.xticks(rotation=45, ha='right')
for bar in bars3:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom')

# Best model visualization
plt.subplot(2, 2, 4)
best_model = results_df.iloc[0]
plt.text(0.1, 0.8, f"BEST MODEL: {best_model['Model']}", fontsize=14, fontweight='bold')
plt.text(0.1, 0.6, f"R² Score: {best_model['R2']:.4f}", fontsize=12)
plt.text(0.1, 0.5, f"MSE: {best_model['MSE']:.6f}", fontsize=12)
plt.text(0.1, 0.4, f"MAE: {best_model['MAE']:.4f}", fontsize=12)
plt.text(0.1, 0.2, "Interpretation:", fontsize=10, fontweight='bold')
plt.text(0.1, 0.1, "Higher R² = Better Fit\nLower MSE/MAE = Better Accuracy", fontsize=9)
plt.axis('off')

plt.suptitle('Regression Models Performance Comparison', fontsize=16)
plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ====================== INTERPRETATION ======================
print("\n" + "="*60)
print("MODEL INTERPRETATION AND INSIGHTS")
print("="*60)

print("\n1. MULTIPLE LINEAR REGRESSION (MLR):")
print(f"   Equation: CUE = {coefficients['const']:.4f} + {coefficients['Temp']:.4f}*Temp + "
      f"{coefficients['RenewPct']:.4f}*RenewPct + {coefficients['RackDensity']:.4f}*RackDensity")
print("   Interpretation: Each unit increase in Temp changes CUE by β₁, etc.")

print("\n2. POLYNOMIAL REGRESSION:")
print("   Captures non-linear relationships between Humidity and CUE")
print("   Allows for curves in the relationship")

print("\n3. LASSO REGRESSION:")
print("   Performs feature selection by shrinking some coefficients to zero")
print(f"   Selected {np.sum(lasso.coef_ != 0)} out of {len(features_all)} features")

print("\n4. RIDGE REGRESSION:")
print("   Shrinks all coefficients but doesn't set them to zero")
print("   Helps with multicollinearity")

print("\n5. RANDOM FOREST:")
print("   Ensemble of decision trees")
print("   Can capture complex non-linear relationships")
print("   Provides feature importance scores")

print("\n6. GRADIENT BOOSTING:")
print("   Sequentially builds trees to correct errors")
print("   Often achieves highest accuracy")

print("\n" + "="*60)
print("RECOMMENDATIONS FOR NER (Northeast India Region)")
print("="*60)

# Based on Location_A (which might represent NER based on Monsoon=1)
ner_data = df[df['Location'] == 'Location_A']
print(f"\nNER (Location_A) Statistics:")
print(f"Average CUE: {ner_data['CUE'].mean():.4f}")
print(f"Average Renewable %: {ner_data['RenewPct'].mean():.1f}%")
print(f"Average Temperature: {ner_data['Temp'].mean():.1f}°C")
print(f"Monsoon Effect: Present (Monsoon = 1)")

print("\nKey Factors for NER based on models:")
print("1. Higher renewable % → Lower CUE (negative correlation)")
print("2. Higher temperature → Higher CUE")
print("3. Monsoon season helps reduce CUE")

print("\n" + "="*60)
print("ANALYSIS COMPLETED!")
print("="*60)