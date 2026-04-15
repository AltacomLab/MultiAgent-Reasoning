# MultiAgent-Reasoning
Multi-Agent Neuro-Symbolic Reasoning for Reliable Decision-Making under Uncertainty/

⚙️ Installation
Recommended Python version:

Python 3.10+
Install dependencies:

pip install pandas numpy scikit-learn networkx
Run Experiment

python main.py

Framework Components
agents.py
Defines heterogeneous agents:

Linear Regression Agent
Decision Tree Agent
Random Forest Agent
uncertainty.py
Implements uncertainty estimation using prediction variance.

conflict_graph.py
Constructs pairwise disagreement graph between agents.

aggregation.py
Implements:

Majority aggregation
Weighted aggregation
Uncertainty-aware aggregation
main.py
Main pipeline:

Load dataset
Train agents
Generate predictions
Build conflict graph
Aggregate outputs
Evaluate MSE

 Research Motivation
Traditional ensemble methods often assume all agents are equally reliable.
Our framework instead:

detects conflicts,
estimates uncertainty,
prioritizes reliable agents.
This is particularly useful in:

smart grids
autonomous systems
distributed AI
sensor networks

 Reproducibility
All experiments can be reproduced directly from the included CSV files and Python scripts.

No external APIs or hidden data are required.

 Citation

 Author
Thi Hong Khanh Nguyen
Electric Power University
Hanoi, Vietnam
