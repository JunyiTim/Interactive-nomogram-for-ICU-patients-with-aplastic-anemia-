import numpy as np
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from sklearn.preprocessing import StandardScaler

# Load the logistic regression models (assumed to be trained previously)
thresholds = [7, 14, 28]
logistic_models = {}
feature_names = ['APSIII', 'Anion Gap', 'Base Excess', 'Bicarbonate', 'Fibrinogen, Functional', 'Lactate Dehydrogenase (LD)', 'Platelet Count']

for threshold in thresholds:
    logistic_models[threshold] = pd.read_csv(f'lr_coefficients_{threshold}_days.csv')

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Interactive Nomogram for ICU Mortality Prediction"),
    html.P("Enter patient values to predict mortality risk."),
    
    # Feature input fields
    *[html.Div([
        html.Label(f"{feature}"),
        dcc.Input(id=f"input-{feature}", type="number", value=0, step=0.1)
    ]) for feature in feature_names],
    
    # Threshold selection
    html.Label("Select Prediction Timeframe (Days)"),
    dcc.RadioItems(
        id="threshold-selector",
        options=[{"label": f"{t}-day", "value": t} for t in thresholds],
        value=7,
        inline=True
    ),
    
    html.Button("Predict", id="predict-button", n_clicks=0),
    
    html.H2("Predicted Mortality Probability"),
    html.Div(id="probability-output", style={"fontSize": "24px", "fontWeight": "bold"})
])

# Callback for updating the prediction
@app.callback(
    Output("probability-output", "children"),
    [Input("predict-button", "n_clicks")],
    [Input(f"input-{feature}", "value") for feature in feature_names] + [Input("threshold-selector", "value")]
)
def update_prediction(n_clicks, *inputs):
    threshold = inputs[-1]  # Last input is threshold selection
    feature_values = np.array(inputs[:-1])  # All other inputs are feature values
    
    # Get the corresponding model coefficients
    coef_df = logistic_models[threshold]
    intercept = coef_df[coef_df['feature'] == 'intercept']['coef'].values[0]
    coefs = coef_df[coef_df['feature'] != 'intercept']['coef'].values
    
    # Calculate log-odds and probability
    log_odds = intercept + np.dot(coefs, feature_values)
    probability = 1 / (1 + np.exp(-log_odds))
    
    return f"{threshold}-day Mortality Probability: {probability:.4f}"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)