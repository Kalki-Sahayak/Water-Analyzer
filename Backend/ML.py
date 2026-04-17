import joblib
import pandas as pd
import shap
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings('ignore')


IS_10500_LIMITS = {
    "pH": {"min": 6.5, "max_acc": 8.5, "max_perm": 8.5}, # No relaxation
    "TDS_mg_L": {"max_acc": 500, "max_perm": 2000},
    "Turbidity_NTU": {"max_acc": 1, "max_perm": 5},
    "Hardness_mg_L": {"max_acc": 200, "max_perm": 600},
    "Nitrates_mg_L": {"max_acc": 45, "max_perm": 45},    # No relaxation
    "Fluoride_mg_L": {"max_acc": 1.0, "max_perm": 1.5},
    "Iron_mg_L": {"max_acc": 0.3, "max_perm": 0.3},      # No relaxation
    "Total_Coliform_MPN": {"max_acc": 0, "max_perm": 0}, # Strictly Zero
    "Ecoli_MPN": {"max_acc": 0, "max_perm": 0},          # Strictly Zero
    "Lead_ppb": {"max_acc": 10, "max_perm": 10},         # No relaxation (0.01 mg/L)
    "Arsenic_ppb": {"max_acc": 10, "max_perm": 50}       # 0.01 to 0.05 mg/L
}

TREATMENT_METHODS = {
    "pH": "pH Adjustment: Inject acidic (e.g., CO2) or alkaline (e.g., Sodium Hydroxide) chemicals.",
    "TDS_mg_L": "Reverse Osmosis (RO): Use semi-permeable membranes to remove dissolved solids.",
    "Turbidity_NTU": "Coagulation & Filtration: Add alum to clump particles, followed by sand filtration.",
    "Hardness_mg_L": "Water Softening: Use ion-exchange resins to replace calcium/magnesium with sodium.",
    "Nitrates_mg_L": "Ion Exchange / Biological Denitrification: Target nitrate removal using specific resins.",
    "Fluoride_mg_L": "Activated Alumina / Nalgonda Technique: Filter through activated alumina.",
    "Iron_mg_L": "Aeration & Filtration: Expose water to air to oxidize iron, then filter out rust.",
    "Total_Coliform_MPN": "Chlorination / UV Treatment: Disinfect water using chlorine dosing or UV light.",
    "Ecoli_MPN": "Aggressive Disinfection: Immediate boiling or high-dose chlorination/Ozone treatment.",
    "Lead_ppb": "Reverse Osmosis / Carbon Filtration: Utilize specialized filters to capture heavy metals.",
    "Arsenic_ppb": "Coagulation-Assisted Filtration: Co-precipitate with iron or use granular ferric oxide."
}


def analyze_water_sample(sample_dict):
    warnings_list = []
    critical_list = []

    for param, value in sample_dict.items():
        if param not in IS_10500_LIMITS:
            continue
            
        limit = IS_10500_LIMITS[param]

        # 1. pH Check (Strict Range: 6.5 - 8.5)
        if param == "pH":
            if value < limit["min"] or value > limit["max_perm"]:
                critical_list.append({
                    "Parameter": param, "Value": value, 
                    "Limit_Text": "Range: 6.5 - 8.5", 
                    "Treatment": TREATMENT_METHODS[param]
                })

        # 2. TDS Check (Acceptable: 500, Permissible: 2000)
        elif param == "TDS_mg_L":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Max 2000 (Permissible Exceeded)", "Treatment": TREATMENT_METHODS[param]})
            elif value > limit["max_acc"]:
                warnings_list.append({"Parameter": param, "Value": value, "Limit_Text": "Acceptable: 500 | Permissible: 2000", "Treatment": TREATMENT_METHODS[param]})

        # 3. Turbidity Check (Acceptable: 1, Permissible: 5)
        elif param == "Turbidity_NTU":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Max 5 (Permissible Exceeded)", "Treatment": TREATMENT_METHODS[param]})
            elif value > limit["max_acc"]:
                warnings_list.append({"Parameter": param, "Value": value, "Limit_Text": "Acceptable: 1 | Permissible: 5", "Treatment": TREATMENT_METHODS[param]})

        # 4. Hardness Check (Acceptable: 200, Permissible: 600)
        elif param == "Hardness_mg_L":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Max 600 (Permissible Exceeded)", "Treatment": TREATMENT_METHODS[param]})
            elif value > limit["max_acc"]:
                warnings_list.append({"Parameter": param, "Value": value, "Limit_Text": "Acceptable: 200 | Permissible: 600", "Treatment": TREATMENT_METHODS[param]})

        # 5. Nitrates Check (Strict Maximum: 45. No relaxation.)
        elif param == "Nitrates_mg_L":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Strict Max: 45", "Treatment": TREATMENT_METHODS[param]})

        # 6. Fluoride Check (Acceptable: 1.0, Permissible: 1.5)
        elif param == "Fluoride_mg_L":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Max 1.5 (Permissible Exceeded)", "Treatment": TREATMENT_METHODS[param]})
            elif value > limit["max_acc"]:
                warnings_list.append({"Parameter": param, "Value": value, "Limit_Text": "Acceptable: 1.0 | Permissible: 1.5", "Treatment": TREATMENT_METHODS[param]})

        # 7. Iron Check (Strict Maximum: 0.3. No relaxation.)
        elif param == "Iron_mg_L":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Strict Max: 0.3", "Treatment": TREATMENT_METHODS[param]})

        # 8. Total Coliform Check (Strict Maximum: 0. No bacteria allowed.)
        elif param == "Total_Coliform_MPN":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Strict Max: 0", "Treatment": TREATMENT_METHODS[param]})

        # 9. E. coli Check (Strict Maximum: 0. No bacteria allowed.)
        elif param == "Ecoli_MPN":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Strict Max: 0", "Treatment": TREATMENT_METHODS[param]})

        # 10. Lead Check (Strict Maximum: 10 ppb. No relaxation.)
        elif param == "Lead_ppb":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Strict Max: 10", "Treatment": TREATMENT_METHODS[param]})

        # 11. Arsenic Check (Acceptable: 10 ppb, Permissible: 50 ppb)
        elif param == "Arsenic_ppb":
            if value > limit["max_perm"]:
                critical_list.append({"Parameter": param, "Value": value, "Limit_Text": "Max 50 (Permissible Exceeded)", "Treatment": TREATMENT_METHODS[param]})
            elif value > limit["max_acc"]:
                warnings_list.append({"Parameter": param, "Value": value, "Limit_Text": "Acceptable: 10 | Permissible: 50", "Treatment": TREATMENT_METHODS[param]})

    return warnings_list, critical_list




# 'rf_model.pkl', 'svm_model.pkl', or 'xgb_model.pkl'
MODEL_FILE = 'rf_model.pkl' 

try:
    loaded_model = joblib.load(MODEL_FILE)
    print(f"Successfully loaded {MODEL_FILE}\n")
except FileNotFoundError:
    print(f"Error: Could not find {MODEL_FILE}. Make sure you ran the training script first!")
    exit()


print("--- Water Quality Predictor ---")
print("Please enter the following water parameters:")

try:
    user_ph = float(input("pH level: "))
    user_tds = float(input("TDS (mg/L): "))
    user_turbidity = float(input("Turbidity (NTU): "))
    user_hardness = float(input("Hardness (mg/L): "))
    user_nitrates = float(input("Nitrates (mg/L): "))
    user_fluoride = float(input("Fluoride (mg/L): "))
    user_iron = float(input("Iron (mg/L): "))
    user_coliform = float(input("Total Coliform (MPN): "))
    user_ecoli = float(input("E. coli (MPN): "))
    user_lead = float(input("Lead (ppb): "))
    user_arsenic = float(input("Arsenic (ppb): "))
except ValueError:
    print("\nError: Please enter numbers only. Try running the script again.")
    exit()


new_water_sample = pd.DataFrame({
    "pH": [user_ph],
    "TDS_mg_L": [user_tds],
    "Turbidity_NTU": [user_turbidity],
    "Hardness_mg_L": [user_hardness],
    "Nitrates_mg_L": [user_nitrates],
    "Fluoride_mg_L": [user_fluoride],
    "Iron_mg_L": [user_iron],
    "Total_Coliform_MPN": [user_coliform],
    "Ecoli_MPN": [user_ecoli],
    "Lead_ppb": [user_lead],
    "Arsenic_ppb": [user_arsenic]
})




print("\nAnalyzing sample...")
ml_prediction = loaded_model.predict(new_water_sample)[0]

# Execute Treatment Analysis
sample_dict = new_water_sample.iloc[0].to_dict()
warnings_list, critical_list = analyze_water_sample(sample_dict)


final_status_is_potable = False
override_triggered = False

if len(critical_list) > 0:
    
    final_status_is_potable = False
    if ml_prediction == 1:
        override_triggered = True 
elif ml_prediction == 1:
    
    final_status_is_potable = True
else:
    
    final_status_is_potable = False


print("\n" + "="*50)
if final_status_is_potable:
    if len(warnings_list) > 0:
         print(" RESULT: POTABLE (With Warnings - Requires Attention)")
    else:
         print(" RESULT: POTABLE (Excellent Quality)")
else:
    if override_triggered:
        print(" RESULT: NOT POTABLE (Unsafe)")
    else:
        print(" RESULT: NOT POTABLE (Unsafe)")
print("="*50)

if hasattr(loaded_model, "predict_proba"):
    probabilities = loaded_model.predict_proba(new_water_sample)
    # if override_triggered:
        # print(f"ML Model's Original Prediction: POTABLE ({probabilities[0][1] * 100:.2f}% confidence)")
        # print("-> Status changed to NOT POTABLE due to strict IS 10500 critical parameter violations.")
    conf = probabilities[0][1] * 100 if ml_prediction == 1 else probabilities[0][0] * 100
    status_text = "Potable" if ml_prediction == 1 else "Not Potable"
    print(f"Model Confidence (Probability of being {status_text}): {conf:.2f}%")


print("\n--- TREATMENT ANALYSIS ---")
if len(critical_list) == 0 and len(warnings_list) == 0:
    print("[+] All individual parameters are within strict IS 10500 acceptable limits.")

if warnings_list:
    print("\n[WARNINGS] - Within permissible limits, but treatment recommended:")
    for w in warnings_list:
        print(f"  ⚠️ {w['Parameter']}")
        print(f"     - Your Value: {w['Value']} (Standard: {w['Limit_Text']})")
        print(f"     - Action: {w['Treatment']}")

if critical_list:
    print("\n[CRITICALS] - Exceeds maximum permissible limits. Immediate treatment required:")
    for c in critical_list:
        print(f"  🛑 {c['Parameter']}")
        print(f"     - Your Value: {c['Value']} (Standard: {c['Limit_Text']})")
        print(f"     - Action: {c['Treatment']}")



# SHAP ANALYSIS

print("\n--- SHAP ANALYSIS ---")
try:
    # Extract the preprocessor and model from the pipeline
    preprocessor = loaded_model.named_steps['preprocess']
    rf_model = loaded_model.named_steps['model']

    # Transform the single user input to match what the model expects
    sample_transformed = preprocessor.transform(new_water_sample)

    # Initialize SHAP and calculate values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(sample_transformed)

    # Extract feature names
    feature_names = preprocessor.get_feature_names_out()

    # Handle SHAP array formatting 
    if isinstance(shap_values, list):
        contributions = shap_values[1][0] # Class 1 (Potable) contributions
    elif len(shap_values.shape) == 3:
        contributions = shap_values[0, :, 1]
    else:
        contributions = shap_values[0]

    # Combine names and contributions, sort by absolute impact
    shap_data = list(zip(feature_names, contributions))
    shap_data.sort(key=lambda x: abs(x[1]), reverse=True)

    
    for feature, impact in shap_data:
        
        clean_feature = feature.replace('num__', '').replace('cat__', '')
        print(f"{clean_feature:25} | Impact: {impact:+.4f}")
        
except Exception as e:
    print(f"Could not generate SHAP explanation. Error: {e}")

print("\nAnalysis Complete.")