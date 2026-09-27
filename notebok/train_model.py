#%% Cell 1: Library
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
#%% Cell 2: Load data
df = pd.read_csv('../data/stunting_jeneponto_preprocessed.csv')

def age_band(a):
    if a <= 11: return '0-11'
    elif a <= 23: return '12-23'
    elif a <= 35: return '24-35'
    elif a <= 47: return '36-47'
    else: return '48-59'

df['Age_Band'] = df['Age_Month'].apply(age_band)

RAW_FEATURES = ['Gender_Code', 'Age_Band', 'Height_cm']  # dipakai utk hitung Height_std, bukan fitur final ke model
LABEL = 'Stunting_Label'

X = df[RAW_FEATURES]
y = df[LABEL]

print(df[RAW_FEATURES + [LABEL]].describe(include='all'))
#%% Cell 3: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print('Train:', X_train.shape, ' Test:', X_test.shape)
print('Proporsi kelas train:\n', y_train.value_counts(normalize=True))
print('Proporsi kelas test:\n', y_test.value_counts(normalize=True))

#%% Cell 4: Standarisasi data
age_band_stats = X_train.groupby('Age_Band')['Height_cm'].agg(['mean', 'std'])
print(age_band_stats)

def add_height_std(df_part, stats):
    merged = df_part.merge(stats, left_on='Age_Band', right_index=True, how='left')
    return (merged['Height_cm'] - merged['mean']) / merged['std']

X_train_final = pd.DataFrame({
    'Gender_Code': X_train['Gender_Code'].values,
    'Height_std': add_height_std(X_train, age_band_stats).values
})
X_test_final = pd.DataFrame({
    'Gender_Code': X_test['Gender_Code'].values,
    'Height_std': add_height_std(X_test, age_band_stats).values
})

FEATURES = ['Gender_Code', 'Height_std']
#%% Cell 5: Training model
model = GaussianNB()
model.fit(X_train_final, y_train)

y_pred = model.predict(X_test_final)
y_proba = model.predict_proba(X_test_final)
#%% Cell 6: Evaluasi model
acc = accuracy_score(y_test, y_pred)
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

print(f'Accuracy: {acc:.4f}')
print()
print('Classification report:')
print(classification_report(y_test, y_pred, target_names=['Not Stunted', 'Stunted']))
print()
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(pd.DataFrame(cm,
                    index=['Actual: Not Stunted', 'Actual: Stunted'],
                    columns=['Pred: Not Stunted', 'Pred: Stunted']))
#%% Cell 7: Baseline pembanding
majority_class = y_train.mode()[0]
y_baseline = np.full_like(y_test, fill_value=majority_class)

baseline_acc = accuracy_score(y_test, y_baseline)
print(f'Baseline accuracy (selalu prediksi kelas {majority_class}): {baseline_acc:.4f}')
print(f'Naive Bayes accuracy: {acc:.4f}')
print(f'Selisih terhadap baseline: {acc - baseline_acc:+.4f}')
#%% Cell 8: Save model & Encoding map
import os
MODEL_DIR = os.path.join('..', 'model')
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, 'stunting_nb_model.pkl'))

encoding_map = {
    'gender': {'F': 0, 'M': 1},
    'label': {0: 'Not Stunted', 1: 'Stunted'},
    'features_order': FEATURES,
    'age_bands': {
        band: {'mean': row['mean'], 'std': row['std']}
        for band, row in age_band_stats.iterrows()
    }
}

with open(os.path.join(MODEL_DIR, 'encoding_map.json'), 'w') as f:
    json.dump(encoding_map, f, indent=2)

print('Model dan encoding map tersimpan.')
# %%
