import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

try:
    df = pd.read_csv('atp_tennis.csv')
    print("Dataset cargado correctamente.")
    print(df.head()) # Mostrar las primeras filas del DataFrame
    print(df.info()) # Mostrar información sobre las columnas y tipos de datos
except FileNotFoundError:
    print("Error: No se encontró el archivo 'tennis_matches.csv'. Asegúrate de que el archivo esté en la ruta correcta.")
    exit()

# 1. (Victoria para Player_1):
df['Player1_Win'] = df['Player_1'] == df['Winner']

# 2. (Duración en Sets):
def get_num_sets(score):
    if isinstance(score, str):
        return len(score.split())
    return None

df['Num_Sets'] = df['Score'].apply(get_num_sets)

# 3. Ranking Diferencial:
df['Ranking_Difference'] = df['Rank_1'] - df['Rank_2']

# 4. Puntos de Ranking Diferencial:
df['Points_Difference'] = df['Pts_1'] - df['Pts_2']

# 5. Convertir la superficie a categórica
df['Surface'] = df['Surface'].astype('category')

# 1. One-hot encoding de la columna 'Surface':
df = pd.get_dummies(df, columns=['Surface'], prefix='Surface')

# 2. Selección de características para el modelo de probabilidad de victoria:
features_win = ['Ranking_Difference'] + [col for col in df.columns if col.startswith('Surface_')]
target_win = 'Player1_Win'

# 3. Selección de características para el modelo de duración en sets:
features_sets = ['Ranking_Difference'] + [col for col in df.columns if col.startswith('Surface_')]
target_sets = 'Num_Sets'

# 4. Manejo de valores faltantes (si los hubiera)
df_win = df[features_win + [target_win]].dropna()
df_sets = df[features_sets + [target_sets]].dropna()

# 5. División de los datos en conjuntos de entrenamiento y prueba:
X_win = df_win[features_win]
y_win = df_win[target_win]
X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(X_win, y_win, test_size=0.2, random_state=42)

X_sets = df_sets[features_sets]
y_sets = df_sets[target_sets]
X_train_sets, X_test_sets, y_train_sets, y_test_sets = train_test_split(X_sets, y_sets, test_size=0.2, random_state=42)

# Inicializar y entrenar el modelo de Regresión Logística
model_win = LogisticRegression(solver='liblinear', random_state=42)
model_win.fit(X_train_win, y_train_win)

# Realizar predicciones en el conjunto de prueba
y_pred_win = model_win.predict(X_test_win)
y_prob_win = model_win.predict_proba(X_test_win)[:, 1] # Probabilidad de la clase positiva (Player 1 gana)

# Evaluar el modelo
accuracy_win = accuracy_score(y_test_win, y_pred_win)
report_win = classification_report(y_test_win, y_pred_win)
roc_auc = roc_auc_score(y_test_win, y_prob_win)

# Inicializar y entrenar el modelo de Random Forest para la predicción del número de sets
model_sets = RandomForestClassifier(random_state=42)
model_sets.fit(X_train_sets, y_train_sets)

# Realizar predicciones en el conjunto de prueba
y_pred_sets = model_sets.predict(X_test_sets)

# Evaluar el modelo
accuracy_sets = accuracy_score(y_test_sets, y_pred_sets)
report_sets = classification_report(y_test_sets, y_pred_sets)

print("\nResultados del modelo de Random Forest (Predicción del Número de Sets):")
print(f"Accuracy: {accuracy_sets:.4f}")
print("\nClassification Report:")
print(report_sets)

print("Resultados del modelo de Regresión Logística (Predicción de Victoria):")
print(f"Accuracy: {accuracy_win:.4f}")
print(f"AUC: {roc_auc:.4f}")
print("\nClassification Report:")
print(report_win)

# Visualizar la curva ROC
fpr, tpr, thresholds = roc_curve(y_test_win, y_prob_win)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Predicción de Victoria')
plt.legend(loc='lower right')
plt.show()
