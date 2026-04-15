import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from agents import build_agents
from uncertainty import prediction_uncertainty
from conflict_graph import build_conflict_graph, conflict_score
from aggregation import majority_voting


# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("../data/raw/synthetic_energy.csv")

X = df[["temperature", "wind_speed", "solar_irradiance"]]
y = df["energy_output"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ===============================
# Build Agents
# ===============================
agents = build_agents()

predictions = []

for agent in agents:
    agent.fit(X_train, y_train)
    pred = agent.predict(X_test)
    predictions.append(pred)

# ===============================
# Conflict Graph
# ===============================
G = build_conflict_graph(predictions)

print("Conflict edges:", conflict_score(G))

# ===============================
# Final Aggregation
# ===============================
final_pred = majority_voting(predictions)

mse = mean_squared_error(y_test, final_pred)

print("Final MSE:", round(mse, 4))