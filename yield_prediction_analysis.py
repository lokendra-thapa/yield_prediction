import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Read the dataset
df = pd.read_csv('dataset_for_yield_prediction.csv')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())

# Create a copy of the dataframe for visualization
df_viz = df.copy()

# Data Preprocessing
# Convert categorical variables to numerical using Label Encoding
categorical_columns = ['Crop_Type', 'Irrigation_Type', 'Soil_Type', 'Season']
df_encoded = df.copy()

# Create and fit separate label encoders for each categorical column
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df_encoded[column] = label_encoders[column].fit_transform(df_encoded[column])

# Add derived features
df_encoded['Fertilizer_per_acre'] = df_encoded['Fertilizer_Used(tons)'] / df_encoded['Farm_Area(acres)']
df_encoded['Pesticide_per_acre'] = df_encoded['Pesticide_Used(kg)'] / df_encoded['Farm_Area(acres)']
df_encoded['Water_per_acre'] = df_encoded['Water_Usage(cubic meters)'] / df_encoded['Farm_Area(acres)']

# Separate features and target
X = df_encoded.drop(['Farm_ID', 'Yield(tons)'], axis=1)  # Keep all features including derived ones
y = df_encoded['Yield(tons)']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
}

best_model = None
best_score = float('-inf')

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n{name} Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    print(f"Mean Absolute Error: {mae:.2f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
    
    if r2 > best_score:
        best_score = r2
        best_model = model

print(f"\nBest model: {list(models.keys())[list(models.values()).index(best_model)]}")

# Feature Importance for the best model
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Visualizations
plt.figure(figsize=(15, 10))

# 1. Crop Type vs Yield
plt.subplot(2, 2, 1)
sns.boxplot(x='Crop_Type', y='Yield(tons)', data=df_viz)
plt.xticks(rotation=45)
plt.title('Crop Type vs Yield')

# 2. Irrigation Type vs Yield
plt.subplot(2, 2, 2)
sns.boxplot(x='Irrigation_Type', y='Yield(tons)', data=df_viz)
plt.xticks(rotation=45)
plt.title('Irrigation Type vs Yield')

# 3. Feature Importance
plt.subplot(2, 2, 3)
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')

# 4. Correlation Heatmap (using only numeric columns)
plt.subplot(2, 2, 4)
numeric_cols = ['Farm_Area(acres)', 'Fertilizer_Used(tons)', 'Pesticide_Used(kg)', 
                'Yield(tons)', 'Water_Usage(cubic meters)']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.savefig('static/analysis_plots.png')
plt.close()

# Save the model and preprocessing objects
import joblib
joblib.dump(best_model, 'yield_prediction_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoders, 'label_encoders.joblib')

print("\nAnalysis complete! Check 'static/analysis_plots.png' for visualizations.")
print("Model and preprocessing objects have been saved.") 