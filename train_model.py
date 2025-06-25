import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt

# --- Utwórz katalog na obrazy, jeśli nie istnieje ---
os.makedirs('static/images', exist_ok=True)

# --- 1. Wczytaj dane ---
df = pd.read_csv('dane_rekomendacyjne_500.csv')

# --- 2. Usuń outliery (3-sigma) ---
numerical = ['wiek','wzrost','waga','ciśnienie skurczowe','ciśnienie rozkurczowe',
             'glukoza','cholesterol','HDL','LDL','triglicerydy','stres','BMI']
df = df[(np.abs((df[numerical] - df[numerical].mean()) / df[numerical].std()) < 3).all(axis=1)]

# --- 3. Kodowanie zmiennych kategorycznych ---
df['płeć'] = df['płeć'].map({'Mężczyzna':1, 'Kobieta':0})
df['palenie'] = df['palenie'].map({'tak':1, 'nie':0})
df['aktywność'] = df['aktywność'].map({'brak':0, 'umiarkowana':1, 'duża':2})
df['poważna choroba w przeszłości'] = df['poważna choroba w przeszłości'].map({'tak':1, 'nie':0})

# --- 4. Przygotuj macierz cech i etykiet ---
features = ['wiek','płeć','wzrost','waga','ciśnienie skurczowe','ciśnienie rozkurczowe',
            'glukoza','cholesterol','HDL','LDL','triglicerydy','palenie',
            'aktywność','stres','BMI','poważna choroba w przeszłości']
labels = ['cukrzyca','nadciśnienie','otyłość','choroba serca','choroba nerek']
X = df[features]
y = df[labels]

# --- 5. Skalowanie cech ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 6. Trening modelu Random Forest ---
base_clf = RandomForestClassifier(n_estimators=100, random_state=42)
model = MultiOutputClassifier(base_clf)
model.fit(X_scaled, y)

# --- 7. Zapis modeli ---
with open('scaler.pkl', 'wb') as sf:
    pickle.dump(scaler, sf)
with open('model.pkl', 'wb') as mf:
    pickle.dump(model, mf)
print("Zapisano: scaler.pkl i model.pkl")

# --- 8. Rysowanie macierzy korelacji ---
corr = df[features + labels].corr()
plt.figure(figsize=(12, 10))
plt.imshow(corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.tight_layout()
plt.savefig('static/images/corr.png')