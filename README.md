# 🏏 T20 International Match Winner Predictor

A Machine Learning web application that predicts the outcome of T20 International cricket matches based on historical data, team form, and match conditions.

## 📊 Project Overview
Predicting the winner of a T20 match is complex due to the high impact of the toss, venue conditions, and recent team momentum. This project uses a **Random Forest Classifier** to analyze these variables and provide a win probability for either team.

### Key Features
* **Binary Classification:** Predicts if "Team 1" will win (1) or lose (0).
* **Feature Engineering:** Includes rolling statistics (wins/runs in the last 10 matches) to capture current team form.
* **Interactive UI:** Built with Streamlit, allowing users to input specific match scenarios.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Deployment:** Streamlit
* **Model:** Random Forest Classifier (Accuracy: ~64.7%)

## 📂 Dataset
The model was trained on a comprehensive dataset of T20 International matches, including:
- Match Venue & City
- Toss Winner & Decision
- Team Lineups
- Historical performance metrics

## 🧠 Model Performance
The model achieves an accuracy of 64.73%. Given the unpredictable nature of T20 cricket, this performance is significantly better than a baseline (50%) and provides a statistically significant edge in prediction.

Top Features Driving Predictions:
-Team Name (Strength)
-Recent Run Rate (Last 10 matches)
-Venue/Country Advantage
-Recent Win Rate
