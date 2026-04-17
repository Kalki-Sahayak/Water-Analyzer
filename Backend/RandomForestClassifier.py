import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Project_Water_Dataset.csv")
target = "potable_label"

X = df[["pH", "TDS_mg_L", "Turbidity_NTU", "Hardness_mg_L", "Nitrates_mg_L", "Fluoride_mg_L", "Iron_mg_L",
        "Total_Coliform_MPN", "Ecoli_MPN", "Lead_ppb", "Arsenic_ppb" ]]
y = df[target]

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns

numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),     
    ('scaler', StandardScaler())                    
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('encoder', OneHotEncoder(handle_unknown='ignore')) 
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
         n_estimators=100,
         max_depth=30,
         min_samples_split=2,
         min_samples_leaf=1,
         max_features='sqrt',
         random_state=42
    ))   
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
model_pipeline.fit(X_train, y_train)

joblib.dump(model_pipeline, 'rf_model.pkl')
print("Random Forest pipeline saved successfully to 'rf_model.pkl'!")

# probabilities = model_pipeline.predict_proba(X_test)[:, 1]
# STRICT_THRESHOLD = 0.95
# y_pred = (probabilities >= STRICT_THRESHOLD).astype(int)

y_pred = model_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cv_score = cross_val_score(model_pipeline, X, y, cv=5, scoring='accuracy')
print("\n5-fold cross validation score:")
print(cv_score.mean())

print()

feature_cols = ["pH","TDS_mg_L","Turbidity_NTU","Hardness_mg_L","Nitrates_mg_L","Fluoride_mg_L",
                "Iron_mg_L","Total_Coliform_MPN","Ecoli_MPN","Lead_ppb","Arsenic_ppb"]


# rf = model_pipeline.named_steps['model']
# importances = rf.feature_importances_

# feature_names = model_pipeline.named_steps['preprocess'].get_feature_names_out()

# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)

# fi_df = feature_importance_df.set_index("Feature")

# fi = fi_df[["Importance"]]

# print(fi_df)

# plt.figure(figsize=(5, 7))
# sns.heatmap(fi, annot=True, cmap="YlOrRd")
# plt.title("Feature Importance-Potability Heatmap")
# plt.xlabel("Importance")
# plt.ylabel("Feature")
# plt.savefig("feature_importance_heatmap.png", dpi=100, bbox_inches='tight')
# plt.close()


# corr_with_target = df[feature_cols].corrwith(df[target])

# corr_with_target = corr_with_target.sort_values(ascending=False)

# print("\nCorrelation of each feature with Potability:")
# print(corr_with_target)

# plt.figure(figsize=(8, 5))
# sns.barplot(x=corr_with_target.values, y=corr_with_target.index)
# plt.title("Correlation of Features with Potability")
# plt.xlabel("Correlation coefficient")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.savefig("feature_correlation_barplot.png", dpi=100, bbox_inches='tight')
# plt.close()

