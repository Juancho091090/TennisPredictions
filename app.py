from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Cargar el dataset y entrenar el modelo.
try:
    df = pd.read_csv('atp_tennis.csv')
    df['Player1_Win'] = df['Player_1'] == df['Winner']
    df['Ranking_Difference'] = df['Rank_1'] - df['Rank_2']
    df = pd.get_dummies(df, columns=['Surface'], prefix='Surface')
    features_win = ['Ranking_Difference'] + [col for col in df.columns if col.startswith('Surface_')]
    target_win = 'Player1_Win'
    df_win = df[features_win + [target_win]].dropna()
    X_win = df_win[features_win]
    y_win = df_win[target_win]
    model_win = LogisticRegression(solver='liblinear', random_state=42)
    model_win.fit(X_win, y_win)
except FileNotFoundError:
    print("Error: No se encontró el archivo 'atp_tennis.csv'.")
    exit()

@app.route('/')
def index():
    title = "Tennis Predictions"
    objective = "El objetivo principal mostrar la probabilidad de vistoria de un jugador de tenis."
    description = "TNNS."
    return render_template('index.html', title=title, objective=objective, description=description)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        player1_rank = int(request.form['rank1'])
        player2_rank = int(request.form['rank2'])
        surface = request.form['surface']

        ranking_difference = player1_rank - player2_rank

        prediction_data = pd.DataFrame({'Ranking_Difference': [ranking_difference]})

        prediction_data['Surface_Carpet'] = 0
        prediction_data['Surface_Clay'] = 0
        prediction_data['Surface_Grass'] = 0
        prediction_data['Surface_Hard'] = 0
        prediction_data[f'Surface_{surface.capitalize()}'] = 1

        prediction_data = prediction_data[features_win]

        probability = model_win.predict_proba(prediction_data)[:, 1][0]
        prediction_text = f"La probabilidad de victoria para el Jugador 1 es: {probability:.2f}"

        # Generar gráfico de barras de probabilidades
        players = ['Jugador 1', 'Jugador 2']
        probabilities = [probability, 1 - probability]
        plt.figure(figsize=(6, 4))
        plt.bar(players, probabilities, color=['blue', 'red'])
        plt.ylabel('Probabilidad')
        plt.title('Probabilidad de Victoria')
        plt.ylim([0, 1])
        # Guardar el gráfico en memoria
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

        return render_template('index.html', prediction_result=prediction_text, plot_url=plot_url,
                               title="Tennis Predictions",
                               objective="El objetivo principal mostrar la probabilidad de vistoria de un jugador de tenis.",
                               description="TNNS.")

if __name__ == '__main__':
    app.run(debug=True)