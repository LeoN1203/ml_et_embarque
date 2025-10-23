import joblib
import numpy as np

model = joblib.load('regression.joblib')

# Utiliser les mêmes données que dans le main() du C
sample_data = np.array([[100.0, 3.0, 1.0]])

prediction_py = model.predict(sample_data)

print(f"Prédiction du modèle Python pour l'exemple : {prediction_py[0]}")