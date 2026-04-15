import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
class BaseAgent:
    def __init__(self, name):
        self.name = name

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


class LinearAgent(BaseAgent):
    def __init__(self):
        super().__init__("LinearAgent")
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class TreeAgent(BaseAgent):
    def __init__(self):
        super().__init__("TreeAgent")
        self.model = DecisionTreeRegressor(max_depth=5)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ForestAgent(BaseAgent):
    def __init__(self):
        super().__init__("ForestAgent")
        self.model = RandomForestRegressor(
            n_estimators=50,
            random_state=42
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def build_agents():
    return [
        LinearAgent(),
        TreeAgent(),
        ForestAgent()
    ]