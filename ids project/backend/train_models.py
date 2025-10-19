
# FILENAME: generate_model_files.py
# PLACEMENT: backend/generate_model_files.py
# Run this script to generate all 5 missing .pkl files

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import joblib

print("=" * 70)
print("GENERATING MODEL FILES FOR BANK CHURN PREDICTOR")
print("=" * 70)

# Load data
print("\n1. Loading Churn_Modelling.csv...")
df = pd.read_csv('Churn_Modelling.csv')
print(f"   ✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Preprocessing
print("\n2. Preprocessing data...")
df_clean = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical variables
le_geo = LabelEncoder()
le_gender = LabelEncoder()
df_clean['Geography'] = le_geo.fit_transform(df_clean['Geography'])
df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])

# Remove NaN
df_clean = df_clean.dropna()
print(f"   ✓ Clean dataset: {df_clean.shape[0]} rows")

# Split features and target
X = df_clean.drop('Exited', axis=1)
y = df_clean['Exited']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
print("\n3. Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"   ✓ Features scaled")

# Train model
print("\n4. Training Gradient Boosting model...")
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import accuracy_score, roc_auc_score
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   ✓ Model trained!")
print(f"   ✓ Accuracy: {accuracy:.4f}")
print(f"   ✓ ROC-AUC: {roc_auc:.4f}")

# Save all files
print("\n5. Saving model files...")

joblib.dump(model, 'best_model_pipeline.pkl')
print("   ✓ best_model_pipeline.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("   ✓ scaler.pkl")

joblib.dump(le_geo, 'label_encoder_geography.pkl')
print("   ✓ label_encoder_geography.pkl")

joblib.dump(le_gender, 'label_encoder_gender.pkl')
print("   ✓ label_encoder_gender.pkl")

feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("   ✓ feature_names.pkl")

print("\n" + "=" * 70)
print("✅ SUCCESS! ALL 5 MODEL FILES GENERATED!")
print("=" * 70)
print("\nGenerated files:")
print("  • best_model_pipeline.pkl")
print("  • scaler.pkl")
print("  • label_encoder_geography.pkl")
print("  • label_encoder_gender.pkl")
print("  • feature_names.pkl")
print("\nYou can now run: python app.py")
print("=" * 70)
