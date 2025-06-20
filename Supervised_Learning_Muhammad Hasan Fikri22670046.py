from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load dan siapkan data saat aplikasi dijalankan
df = pd.read_csv('heart.csv')
df = pd.get_dummies(df, drop_first=True)

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Simpan kolom fitur untuk input user
input_columns = X.columns.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    prediction = None
    if request.method == 'POST':
        try:
            # Ambil data mentah dari form (yang pakai nama umum)
            form_data = request.form.to_dict()

            # Konversi nilai kategorikal ke bentuk kolom dummy
            user_input = {col: 0 for col in input_columns}

            # Langsung input untuk nilai numerik/biner
            numeric_fields = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Sex_M', 'ExerciseAngina_Y']
            for field in numeric_fields:
                user_input[field] = float(form_data.get(field, 0))

            # Mapping kategori ke kolom dummy (ChestPainType)
            chest_pain = form_data.get("ChestPainType", "")
            if chest_pain != "ASY":
                user_input[f"ChestPainType_{chest_pain}"] = 1

            # RestingECG
            ecg = form_data.get("RestingECG", "")
            if ecg != "LVH":
                user_input[f"RestingECG_{ecg}"] = 1

            # ST_Slope
            st_slope = form_data.get("ST_Slope", "")
            if st_slope != "Down":
                user_input[f"ST_Slope_{st_slope}"] = 1

            # Susun sesuai urutan kolom model
            input_data = [user_input[col] for col in input_columns]

            # Prediksi
            input_scaled = scaler.transform([input_data])
            result = model.predict(input_scaled)[0]
            prediction = 'Positif (Penyakit Jantung)' if result == 1 else 'Negatif (Sehat)'
        except Exception as e:
            prediction = f'Error: {str(e)}'


    return render_template('form.html', columns=input_columns, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
