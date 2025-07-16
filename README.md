# NRL Tipping & Prediction Model

This repository contains the code for an end-to-end machine learning system that predicts NRL match outcomes and simulates betting strategies. It features a complete MLOps pipeline, from feature engineering to automated weekly predictions.

**Live Deployment:** Version 1.0 of this model is currently deployed and making live, ongoing predictions at [**nrltipping.vercel.app**](https://nrltipping.vercel.app/).



---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [MLOps Practices](#mlops-practices)
- [Local Setup & Usage](#local-setup--usage)
- [Future Improvements](#future-improvements)
- [Acknowledgements](#acknowledgements)

---

## Tech Stack

- **Backend & Data Processing:** Python
- **Core Libraries:** Pandas, NumPy, Scikit-learn, Joblib
- **Web App:** Next.js, React
- **Hosting:** Vercel
- **Data Scraping:** Python (`requests`) for weather data and odds, Python (`BeautifulSoup4`) for match related data.

---

## Project Structure

The project is organised into a modular structure to separate concerns, following best practices for maintainability and scalability.

├── data/
│ ├── nrl_core_features_list.txt
│ ├── nrl_matches_final_model_ready.csv     # the final feature-rich data ready for training (89 features)
│ ├── nrl_team_stats_final_complete.csv
│ └── nrlBaselineData.csv                   # initial raw data
│
├── model/
│ ├── nrl_baseline_logistic_model_v1.pkl    # trained Sckit-learn model object
│ ├── nrl_feature_importance_baseline.csv   # report on model's feature coefficients 
│ └── nrl_feature_scaler_v1.pkl             # scaler fitted on training data, required for transform new data
│
├── notebook/
│ └── feature_engineering.ipynb             # notebook to transfrom baseline data to model ready data
│
├── scrapers/
│ ├── weather_cache.json
│ ├── weather_cache_openmeteo.json
│ ├── weather_scraper_log.txt
│ └── weather_scraper.py
│
├── scripts/
│ ├── config.py
│ ├── evaluate_model.py                     # loads the trained model and evaluates classification performance, and betting simulation.
│ └── train_model.py                        # script to train the model on the model ready dataset
│
├── LICENSE
└── README.md

---

## MLOps Practices

This project implements foundational MLOps principles to ensure a robust and reliable system.

1.  **Modular Pipeline:** The workflow is separated into distinct, reusable scripts for training (`train_model.py`) and evaluation (`evaluate_model.py`), which is a core tenet of MLOps.

2.  **Automated Inference Pipeline:** A weekly automated job is deployed, which:
    - Ingests new match data.
    - Engineers features.
    - Loads the production model to generate predictions.
    - Executes bets based on a probability threshold and updates the live website.

3.  **Performance Monitoring:** A live dashboard tracks key business metrics like P&L, bankroll, and win rate, providing real-time insight into the model's production performance.

4.  **Leakage-Free Validation:** The training script uses `TimeSeriesSplit` with scaling performed *inside each fold* to provide a reliable, unbiased estimate of model performance before training the final version.

---

## Local Setup & Usage

Follow these steps to set up and run the project on your local machine.

### 1. Setup

Clone the repository and install the required dependencies.

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

---

## Usage

The project can be run locally by following these stages. It's recommended to run them in order to reproduce the full pipeline.

### Stage 0: Feature Engineering (Optional)

If you want to regenerate the model-ready dataset from the raw `nrlBaselineData.csv`, you can run the feature engineering notebook. This is useful for understanding the data transformation process or modifying it.

1.  Open the project in VS Code or start Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Navigate to the `/notebook` directory and open `feature_engineering.ipynb`.
3.  Run the cells sequentially from top to bottom. This will create `nrl_matches_final_model_ready.csv` in the `/data` directory.

---

### Stage 1: Train the Model

This script takes the processed data, performs cross-validation to estimate performance, trains a final model on all data, and saves the model, scaler, and feature importance artifacts to the `/model` directory.

```bash
python scripts/train_model.py

### Stage 2: Evaluate the Model

After a model has been trained and saved, this script loads the artifacts and runs a full evaluation, including statistical reports and a betting simulation.

`python scripts/evaluate_model.py`

---

## Future Improvements

While Version 1.0 is successfully deployed, the project is designed for continuous improvement. The roadmap is divided into two key areas: enhancing the model's predictive power and automating the operational lifecycle.

### Iterating and Improving

These steps focus on building a more accurate and robust model.

-   **Realistic Backtesting:** Implement an out-of-sample (OOF) backtesting script to get a ground-truth measure of profitability, which is a more reliable benchmark than in-sample simulations.
-   **Advanced Models:** Experiment with more complex gradient-boosting models like XGBoost or LightGBM, which are often better at capturing non-linear relationships in the data.
-   **Hyperparameter Tuning:** Use automated tools like Optuna or Scikit-learn's `GridSearchCV` to systematically find the optimal settings for the chosen model.
-   **Enhanced Feature Engineering:** Incorporate new, potentially predictive features such as:
    -   Head-to-head (H2H) historical records between the two competing teams.
    -   Statistics related to the assigned match referee.

### Automating the Lifecycle

These steps focus on evolving the project into a mature, self-improving MLOps system.

-   **Continuous Training (CT):** Automate the model retraining process to run on a regular schedule (e.g., weekly), ensuring the model always learns from the most recent data.
-   **Champion/Challenger Deployment:** Implement an automated workflow that validates new "challenger" models against the live "champion" and promotes them to production only if they demonstrate superior performance on a hold-out dataset.
-   **Data Drift Monitoring:** Add automated checks to monitor for significant changes in the statistical properties of incoming data. This can provide early warnings that the model's performance may degrade due to shifts in game dynamics.

---

## Acknowledgements 
-   Historical (2009-2025) data sourced from https://www.aussportsbetting.com/data/historical-nrl-results-and-odds-data/ (this data was cleaned and features dropped)
-   NRL venue, city and future match data from https://www.nrl.com/draw
-   weather data provided by Open-Meteo API


