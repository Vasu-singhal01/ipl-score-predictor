from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and features
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)
with open("models/all_results.pkl", "rb") as f:
    all_results = pickle.load(f)

# 8 consistent IPL teams
TEAMS = [
    "Chennai Super Kings",
    "Delhi Daredevils",
    "Kings XI Punjab",
    "Kolkata Knight Riders",
    "Mumbai Indians",
    "Rajasthan Royals",
    "Royal Challengers Bangalore",
    "Sunrisers Hyderabad"
]

TEAM_SHORT = {
    "Chennai Super Kings":         "CSK",
    "Delhi Daredevils":            "DD",
    "Kings XI Punjab":             "KXIP",
    "Kolkata Knight Riders":       "KKR",
    "Mumbai Indians":              "MI",
    "Rajasthan Royals":            "RR",
    "Royal Challengers Bangalore": "RCB",
    "Sunrisers Hyderabad":         "SRH",
}

TEAM_COLORS = {
    "Chennai Super Kings":         "#F5A623",
    "Delhi Daredevils":            "#0078BC",
    "Kings XI Punjab":             "#ED1B24",
    "Kolkata Knight Riders":       "#3A225D",
    "Mumbai Indians":              "#004BA0",
    "Rajasthan Royals":            "#254AA5",
    "Royal Challengers Bangalore": "#EC1C24",
    "Sunrisers Hyderabad":         "#F7A721",
}

TEAM_LOGOS = {
    "Chennai Super Kings":         "🦁",
    "Delhi Daredevils":            "🦅",
    "Kings XI Punjab":             "🦁",
    "Kolkata Knight Riders":       "⚡",
    "Mumbai Indians":              "🌊",
    "Rajasthan Royals":            "👑",
    "Royal Challengers Bangalore": "🔥",
    "Sunrisers Hyderabad":         "☀️",
}


def predict_score(bat_team, bowl_team, overs, runs, wickets, runs_last_5):
    """Build feature vector and predict score"""
    row = {
        'runs':         runs,
        'wickets':      wickets,
        'overs':        overs,
        'runs_last_5':  runs_last_5,
        'wickets_last_5': min(wickets, 2),
    }
    # One-hot encode teams
    for team in TEAMS:
        row[f'bat_team_{team}']  = 1 if bat_team  == team else 0
        row[f'bowl_team_{team}'] = 1 if bowl_team == team else 0

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    predicted = int(model.predict(df)[0])
    predicted = max(runs + 10, min(predicted, 260))
    return predicted


@app.route("/")
def index():
    return render_template("index.html",
                           teams=TEAMS,
                           team_short=TEAM_SHORT,
                           team_colors=TEAM_COLORS,
                           team_logos=TEAM_LOGOS)


@app.route("/predict", methods=["POST"])
def predict():
    bat_team    = request.form.get("bat_team")
    bowl_team   = request.form.get("bowl_team")
    overs       = float(request.form.get("overs", 10))
    runs        = int(request.form.get("runs", 80))
    wickets     = int(request.form.get("wickets", 2))
    runs_last_5 = int(request.form.get("runs_last_5", 40))

    if bat_team == bowl_team:
        return render_template("index.html",
                               teams=TEAMS,
                               team_short=TEAM_SHORT,
                               team_colors=TEAM_COLORS,
                               team_logos=TEAM_LOGOS,
                               error="Batting and bowling teams cannot be the same!")

    predicted   = predict_score(bat_team, bowl_team, overs, runs, wickets, runs_last_5)
    range_low   = predicted - 10
    range_high  = predicted + 10
    crr         = round(runs / overs, 2) if overs > 0 else 0
    rrr         = round((predicted - runs) / ((20 - overs)), 2) if overs < 20 else 0

    # Model comparison for display
    model_comparison = [
        {"name": "Linear Regression", "mae": all_results['Linear Regression']['mae'], "rmse": all_results['Linear Regression']['rmse']},
        {"name": "Lasso Regression",  "mae": all_results['Lasso Regression']['mae'],  "rmse": all_results['Lasso Regression']['rmse']},
        {"name": "Ridge Regression",  "mae": all_results['Ridge Regression']['mae'],  "rmse": all_results['Ridge Regression']['rmse']},
    ]

    return render_template("result.html",
                           bat_team=bat_team,
                           bowl_team=bowl_team,
                           overs=overs,
                           runs=runs,
                           wickets=wickets,
                           runs_last_5=runs_last_5,
                           predicted=predicted,
                           range_low=range_low,
                           range_high=range_high,
                           crr=crr,
                           rrr=rrr,
                           bat_color=TEAM_COLORS.get(bat_team, "#004BA0"),
                           bowl_color=TEAM_COLORS.get(bowl_team, "#EC1C24"),
                           bat_short=TEAM_SHORT.get(bat_team, "BAT"),
                           bowl_short=TEAM_SHORT.get(bowl_team, "BOWL"),
                           bat_logo=TEAM_LOGOS.get(bat_team, "🏏"),
                           bowl_logo=TEAM_LOGOS.get(bowl_team, "🎯"),
                           model_comparison=model_comparison,
                           teams=TEAMS)


if __name__ == "__main__":
    print("=" * 55)
    print("  IPL Score Predictor - Starting...")
    print("  Open http://localhost:5000")
    print("  Press CTRL+C to stop")
    print("=" * 55)
    app.run(debug=True)
