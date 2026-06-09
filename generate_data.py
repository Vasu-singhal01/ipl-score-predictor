import pandas as pd
import numpy as np
import os

np.random.seed(42)

# 8 consistent IPL teams (as per PPT)
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

# Team batting & bowling strength (based on historical IPL performance)
TEAM_STRENGTH = {
    "Chennai Super Kings":         {"bat": 0.87, "bowl": 0.83},
    "Delhi Daredevils":            {"bat": 0.76, "bowl": 0.75},
    "Kings XI Punjab":             {"bat": 0.80, "bowl": 0.74},
    "Kolkata Knight Riders":       {"bat": 0.82, "bowl": 0.80},
    "Mumbai Indians":              {"bat": 0.88, "bowl": 0.85},
    "Rajasthan Royals":            {"bat": 0.79, "bowl": 0.78},
    "Royal Challengers Bangalore": {"bat": 0.86, "bowl": 0.72},
    "Sunrisers Hyderabad":         {"bat": 0.78, "bowl": 0.83},
}

def generate_ipl_dataset(n_matches=500):
    """
    Generate realistic IPL dataset mimicking Kaggle IPL dataset structure.
    Each match has over-by-over records from over 5 to 20.
    """
    records = []
    match_id = 1

    for _ in range(n_matches):
        bat_team  = np.random.choice(TEAMS)
        bowl_team = np.random.choice([t for t in TEAMS if t != bat_team])
        year      = np.random.choice(range(2008, 2020),
                                     p=[0.06,0.07,0.08,0.09,0.09,0.10,0.10,0.10,0.11,0.10,0.06,0.04])

        bat_str  = TEAM_STRENGTH[bat_team]["bat"]
        bowl_str = TEAM_STRENGTH[bowl_team]["bowl"]

        # Simulate realistic innings progression
        total_runs = int(np.random.normal(
            loc=160 * bat_str * (1.1 - bowl_str * 0.3),
            scale=18
        ))
        total_runs = max(100, min(total_runs, 250))

        # Generate over-by-over data
        runs_so_far = 0
        wickets     = 0

        for over in range(1, 21):
            # Runs per over based on phase
            if over <= 6:       # Powerplay
                runs_this_over = int(np.random.normal(loc=8.5 * bat_str, scale=2.5))
            elif over <= 15:    # Middle overs
                runs_this_over = int(np.random.normal(loc=7.5 * bat_str * (1 - bowl_str*0.2), scale=2.2))
            else:               # Death overs
                runs_this_over = int(np.random.normal(loc=10.5 * bat_str * (1 - bowl_str*0.15), scale=3.0))

            runs_this_over = max(0, min(runs_this_over, 30))
            runs_so_far   += runs_this_over

            # Wickets
            if np.random.random() < 0.12 * (1 - bat_str + 0.1) and wickets < 9:
                wickets += 1

            # Runs in last 5 overs
            if over >= 5:
                runs_last5 = min(runs_so_far, int(np.random.normal(
                    loc=42 * bat_str * (1 - bowl_str * 0.15), scale=8
                )))
                runs_last5 = max(10, min(runs_last5, 80))
            else:
                runs_last5 = runs_so_far

            if over >= 5:  # Only record from over 5 onwards (as per PPT)
                records.append({
                    "mid":          match_id,
                    "date":         f"{year}-04-{np.random.randint(1,30):02d}",
                    "venue":        "Various",
                    "bat_team":     bat_team,
                    "bowl_team":    bowl_team,
                    "batsman":      "Player A",
                    "bowler":       "Player B",
                    "runs":         runs_so_far,
                    "wickets":      wickets,
                    "overs":        over,
                    "runs_last_5":  runs_last5,
                    "wickets_last_5": min(wickets, np.random.randint(0, 3)),
                    "striker":      np.random.randint(0, 80),
                    "non_striker":  np.random.randint(0, 60),
                    "total":        total_runs
                })

        match_id += 1

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    print("Generating IPL dataset (mimicking Kaggle IPL data)...")
    df = generate_ipl_dataset(600)
    df.to_csv("data/ipl_data.csv", index=False)
    print(f"Dataset created: {len(df)} rows, {len(df.columns)} columns")
    print(f"Years covered: {df['date'].str[:4].unique().tolist()}")
    print(f"Teams: {df['bat_team'].unique().tolist()}")
    print(f"\nSample data:")
    print(df.head())
