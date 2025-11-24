import numpy as np
import pandas as pd
import joblib

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
from itertools import permutations

# =========================================
# CONFIG / FILE PATHS
# =========================================

TEAM_MODEL_PATH = "team_model.pkl"
TEAM_SCALER_PATH = "team_scaler.pkl"
PLAYER_BASELINES_PATH = "player_role_baselines.csv"
TEAM_LEVEL_DATASET_PATH = "team_level_dataset.csv"  # for enemy baselines (optional)
FLEX_STATS_PATH = "flex_game_stats.csv"        # for sample sizes + overall winrates

ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# Team-level model features (must match how you trained team_model.pkl)
TEAM_FEATURES = [
    "goldEarned",                     # team total
    "kills",                          # team total
    "deaths",                         # team total
    "assists",                        # team total
    "cs",                             # team total
    "visionScore",                    # team total
    "totalDamageDealtToChampions",    # team total
    "totalDamageDealt",               # team total
    "teamDragons",
    "teamBarons",
    "teamTowers",
    "enemyDragons",
    "enemyBarons",
    "enemyTowers",
]

# Synergy configuration (additive bump to win probability)
SYNERGY_BONUSES = {
    frozenset({"WT BrokenSword#EUW", "WT GigaChad#1070"}): 0.03,
    frozenset({"Billy Bongo#EUW", "WT Final Stand#EUW"}): 0.02,
}

# Sample size → confidence saturation (games needed for confidence ~1)
CONF_N0 = 30
# Minimum games in a role for that player to be considered valid in that role
MIN_ROLE_GAMES = 10


# =========================================
# LOAD MODELS & BASELINES
# =========================================

team_model = joblib.load(TEAM_MODEL_PATH)
team_scaler = joblib.load(TEAM_SCALER_PATH)

player_role_baselines = pd.read_csv(PLAYER_BASELINES_PATH)

# Ensure we have the predictors we need from baselines
player_feature_cols = [
    c for c in TEAM_FEATURES
    if c in player_role_baselines.columns and not c.startswith("enemy")
]

# Overall per-player baselines (any role)
overall_baselines = (
    player_role_baselines.groupby("summoner")[player_feature_cols]
    .mean()
    .reset_index()
)

all_players = sorted(overall_baselines["summoner"].unique())

# Load team-level dataset to get typical enemy stats, or fall back to zeros
try:
    team_level_df = pd.read_csv(TEAM_LEVEL_DATASET_PATH)
    enemy_baseline = {
        "enemyDragons": team_level_df["enemyDragons"].mean(),
        "enemyBarons": team_level_df["enemyBarons"].mean(),
        "enemyTowers": team_level_df["enemyTowers"].mean(),
    }
    print("[INFO] Loaded enemy baselines from team_level_dataset.csv")
except Exception as e:
    print(f"[WARN] Could not load team_level_dataset.csv ({e}). Using zeros for enemy stats.")
    enemy_baseline = {"enemyDragons": 0.0, "enemyBarons": 0.0, "enemyTowers": 0.0}

# =========================================
# SAMPLE SIZE / OVERALL WINRATE DATA
# =========================================

player_role_counts = None     # per (summoner, role) games
player_total_stats = None     # per summoner: games, winrate

try:
    flex_stats = pd.read_csv(FLEX_STATS_PATH)
    print(f"[INFO] Loaded {len(flex_stats)} rows from {FLEX_STATS_PATH}")
    print("[INFO] Columns in flex_game_stats.csv:", flex_stats.columns.tolist())

    required_cols = {"summoner", "individualPosition", "win"}
    if required_cols.issubset(set(flex_stats.columns)):
        player_role_counts = (
            flex_stats.groupby(["summoner", "individualPosition"])
            .size()
            .reset_index(name="games")
        )
        player_total_stats = (
            flex_stats.groupby("summoner")
            .agg(
                games=("win", "size"),
                winrate=("win", "mean"),
            )
            .reset_index()
        )
        print("[INFO] Built per-role and total game counts + winrates for players.")
    else:
        print(
            "[WARN] flex_game_stats.csv is missing one of "
            f"{required_cols} - confidence and baseline winrates will fall back to defaults."
        )
        player_role_counts = None
        player_total_stats = None

except FileNotFoundError as e:
    print(f"[WARN] {FLEX_STATS_PATH} not found: {e}. Confidence will fall back to default.")
except Exception as e:
    print(f"[WARN] Could not load or process flex_game_stats.csv ({e}). "
          "Confidence and baseline winrates will fall back to default.")
    player_role_counts = None
    player_total_stats = None

# =========================================
# VALID PLAYERS PER ROLE (>= MIN_ROLE_GAMES)
# =========================================

valid_players_by_role = {role: [] for role in ROLES}

if player_role_counts is not None:
    enough_games = player_role_counts[player_role_counts["games"] >= MIN_ROLE_GAMES]
    for role in ROLES:
        subset = enough_games[enough_games["individualPosition"] == role]
        players_for_role = sorted(
            p for p in subset["summoner"].unique() if p in all_players
        )
        valid_players_by_role[role] = players_for_role

    print("[INFO] Valid players per role (>= "
          f"{MIN_ROLE_GAMES} games):")
    for role in ROLES:
        print(f"  {role}: {len(valid_players_by_role[role])} players")
else:
    print("[WARN] No per-role counts; allowing all players for all roles.")
    for role in ROLES:
        valid_players_by_role[role] = list(all_players)


def get_player_role_baseline(summoner, role):
    """
    Return a Series of baseline stats for a given (summoner, role).
    Falls back to per-player overall baseline, then global means.
    """
    subset = player_role_baselines[
        (player_role_baselines["summoner"] == summoner)
        & (player_role_baselines["individualPosition"] == role)
    ]
    if not subset.empty:
        return subset.iloc[0]

    subset2 = overall_baselines[overall_baselines["summoner"] == summoner]
    if not subset2.empty:
        return subset2.iloc[0]

    return player_role_baselines[player_feature_cols].mean()


def get_player_role_games(summoner, role):
    if player_role_counts is None:
        return None

    row = player_role_counts[
        (player_role_counts["summoner"] == summoner)
        & (player_role_counts["individualPosition"] == role)
    ]
    if not row.empty:
        return int(row["games"].iloc[0])

    if player_total_stats is not None:
        row2 = player_total_stats[player_total_stats["summoner"] == summoner]
        if not row2.empty:
            return int(row2["games"].iloc[0])

    return None


def get_player_overall_winrate(summoner):
    if player_total_stats is None:
        return None
    row = player_total_stats[player_total_stats["summoner"] == summoner]
    if not row.empty:
        return float(row["winrate"].iloc[0])
    return None


def get_player_role_confidence(summoner, role):
    games = get_player_role_games(summoner, role)
    if games is None:
        return 0.5, None
    conf = min(1.0, float(games) / CONF_N0)
    return conf, games


def compute_synergy_bonus(selected_players):
    bonus = 0.0
    for i in range(len(selected_players)):
        for j in range(i + 1, len(selected_players)):
            pair = frozenset({selected_players[i], selected_players[j]})
            bonus += SYNERGY_BONUSES.get(pair, 0.0)
    return max(min(bonus, 0.10), -0.10)


def aggregate_team_vector(player_rows):
    team_vector = {f: 0.0 for f in TEAM_FEATURES}
    for f in TEAM_FEATURES:
        if f.startswith("enemy"):
            team_vector[f] = float(enemy_baseline.get(f, 0.0))
        elif f in ["teamDragons", "teamBarons", "teamTowers"]:
            team_vector[f] = float(np.mean([row.get(f, 0.0) for row in player_rows]))
        else:
            team_vector[f] = float(sum(row.get(f, 0.0) for row in player_rows))
    return team_vector


def compute_baseline_team_prob(selected_players):
    probs = []
    for p in selected_players:
        wr = get_player_overall_winrate(p)
        if wr is None:
            wr = 0.5
        probs.append(wr)
    if not probs:
        return 0.5
    return float(np.mean(probs))


def compute_team_prob_for_lineup(lineup_by_role):
    """
    No sliders here (baseline only) – used by 'best team' search.
    """
    player_rows = []
    selected_players = []
    confs = []

    for role, player in lineup_by_role.items():
        selected_players.append(player)
        base = get_player_role_baseline(player, role)

        feat = {f: 0.0 for f in TEAM_FEATURES}
        for col in player_feature_cols:
            if col in feat:
                feat[col] = float(base[col])

        player_rows.append(feat)

        conf, _games = get_player_role_confidence(player, role)
        confs.append(conf)

    lineup_conf = float(np.mean(confs)) if confs else 1.0
    baseline_team_prob = compute_baseline_team_prob(selected_players)

    team_vector = aggregate_team_vector(player_rows)
    X_team = np.array([[team_vector[f] for f in TEAM_FEATURES]])
    X_scaled = team_scaler.transform(X_team)
    base_prob = float(team_model.predict_proba(X_scaled)[0, 1])

    synergy_bonus = compute_synergy_bonus(selected_players)
    prob_with_synergy = max(min(base_prob + synergy_bonus, 1.0), 0.0)

    prob_adj = baseline_team_prob + lineup_conf * (prob_with_synergy - baseline_team_prob)
    prob_adj = max(min(prob_adj, 1.0), 0.0)

    return prob_adj, lineup_conf


# =========================================
# BUILD DASH APP
# =========================================

app = Dash(__name__)


def role_block(role_name, default_player=None):
    """
    One dropdown + one KDA slider per role.
    KDA multiplier meaning:
      - >1.0: more kills/assists/gold, fewer deaths
      - <1.0: fewer kills/assists/gold, more deaths
    """
    role_id = role_name.lower()
    options_players = valid_players_by_role.get(role_name, [])
    return html.Div(
        [
            html.H4(role_name),
            dcc.Dropdown(
                id=f"{role_id}-player",
                options=[{"label": p, "value": p} for p in options_players],
                value=default_player if default_player in options_players else None,
                clearable=False,
                placeholder=f"Select player (≥{MIN_ROLE_GAMES} games in this role)",
                style={"marginBottom": "0.5rem"},
            ),
            html.Div("KDA / performance multiplier", style={"fontSize": "0.85rem"}),
            dcc.Slider(
                id=f"{role_id}-kda-mult",
                min=0.5,
                max=1.5,
                step=0.05,
                value=1.0,
                marks={0.5: "Low", 1.0: "Normal", 1.5: "High"},
            ),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "0.75rem",
            "flex": "1 1 0",
            "margin": "0.25rem",
        },
    )


# choose defaults from valid players per role (if available)
default_players_for_roles = {}
for role in ROLES:
    players_for_role = valid_players_by_role.get(role, [])
    default_players_for_roles[role] = players_for_role[0] if players_for_role else None

app.layout = html.Div(
    [
        html.H2("Flex Team Win Likelihood Simulator", style={"textAlign": "center"}),
        html.P(
            "Pick a team, adjust their KDA/impact, and simulate the win likelihood "
            "using a team-level regression model with player-winrate-based shrinkage.\n"
            f"Only players with ≥{MIN_ROLE_GAMES} games in a role are eligible for that role.",
            style={"textAlign": "center", "whiteSpace": "pre-line"},
        ),
        html.Div(
            [
           

 role_block(role, default_players_for_roles.get(role))
                for role in ROLES
            ],
            style={"display": "flex", "flexWrap": "wrap", "justifyContent": "space-between"},
        ),
        html.Div(
            [
                html.Button(
                    "Find best team composition",
                    id="best-team-button",
                    n_clicks=0,
                    style={"marginRight": "0.5rem"},
                ),
                html.Div(
                    id="best-team-output",
                    style={"marginTop": "0.5rem", "fontSize": "0.9rem"},
                ),
            ],
            style={"margin": "1rem 0", "textAlign": "center"},
        ),
        html.Hr(),
        html.Div(
            [
                html.Div(
                    [
                        html.H4("Simulation settings"),
                        html.Div("Number of simulated games:", style={"fontSize": "0.9rem"}),
                        dcc.Slider(
                            id="sim-runs",
                            min=50,
                            max=1000,
                            step=50,
                            value=300,
                            marks={50: "50", 300: "300", 1000: "1000"},
                        ),
                        html.Div(
                            id="team-win-summary",
                            style={"marginTop": "1rem", "fontSize": "1.2rem", "fontWeight": "bold"},
                        ),
                        html.Div(
                            id="team-win-detail",
                            style={"marginTop": "0.5rem", "fontSize": "0.9rem", "whiteSpace": "pre-line"},
                        ),
                    ],
                    style={"flex": "0 0 30%", "padding": "0.5rem"},
                ),
                html.Div(
                    [
                        dcc.Graph(
                            id="sim-histogram",
                            figure=go.Figure(),
                            style={"height": "350px"},
                        )
                    ],
                    style={"flex": "0 0 70%", "padding": "0.5rem"},
                ),
            ],
            style={"display": "flex", "flexWrap": "wrap"},
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "fontFamily": "Arial, sans-serif"},
)


# =========================================
# CALLBACK: ENFORCE UNIQUE PLAYER SELECTION (WITH ROLE FILTERING)
# =========================================

@app.callback(
    [
        Output("top-player", "options"),
        Output("jungle-player", "options"),
        Output("middle-player", "options"),
        Output("bottom-player", "options"),
        Output("utility-player", "options"),
    ],
    [
        Input("top-player", "value"),
        Input("jungle-player", "value"),
        Input("middle-player", "value"),
        Input("bottom-player", "value"),
        Input("utility-player", "value"),
    ],
)
def update_player_options(top_val, jungle_val, mid_val, bot_val, util_val):
    selected = {
        "TOP": top_val,
        "JUNGLE": jungle_val,
        "MIDDLE": mid_val,
        "BOTTOM": bot_val,
        "UTILITY": util_val,
    }

    outputs = []
    for role in ROLES:
        base_list = valid_players_by_role.get(role, [])
        other_selected = {v for r, v in selected.items() if r != role and v is not None}
        allowed = [p for p in base_list if p not in other_selected]
        options = [{"label": p, "value": p} for p in allowed]
        outputs.append(options)

    return outputs


# =========================================
# CALLBACK: MAIN SIMULATION
# =========================================

inputs = []
for role in ROLES:
    rid = role.lower()
    inputs.extend(
        [
            Input(f"{rid}-player", "value"),
            Input(f"{rid}-kda-mult", "value"),
        ]
    )
inputs.append(Input("sim-runs", "value"))


@app.callback(
    Output("team-win-summary", "children"),
    Output("team-win-detail", "children"),
    Output("sim-histogram", "figure"),
    inputs,
)
def update_simulation(*values):
    sim_runs = int(values[-1]) if values[-1] is not None else 300
    role_values = values[:-1]

    selected_players = []
    for i, role in enumerate(ROLES):
        player = role_values[i * 2]
        selected_players.append(player)
    if any(p is None for p in selected_players):
        msg = (
            "Please select a player (with ≥"
            f"{MIN_ROLE_GAMES} games) for each role to run the simulation."
        )
        empty_fig = go.Figure()
        return msg, "", empty_fig

    player_rows = []
    player_confs = []
    player_games = []
    player_winrates = []

    for i, role in enumerate(ROLES):
        offset = i * 2
        player = role_values[offset]
        kda_mult = role_values[offset + 1] or 1.0

        base = get_player_role_baseline(player, role)

        feat = {f: 0.0 for f in TEAM_FEATURES}
        for col in player_feature_cols:
            if col in feat:
                feat[col] = float(base[col])

        # --- Apply KDA multiplier ---
        # Higher KDA multiplier → more kills/assists/gold, fewer deaths
        if "kills" in feat:
            feat["kills"] *= kda_mult
        if "assists" in feat:
            feat["assists"] *= kda_mult
        if "goldEarned" in feat:
            feat["goldEarned"] *= kda_mult
        if "deaths" in feat:
            safe_mult = max(kda_mult, 0.1)
            feat["deaths"] /= safe_mult
        # ---------------------------

        player_rows.append(feat)

        conf, games = get_player_role_confidence(player, role)
        player_confs.append(conf)
        player_games.append(games if games is not None else "unknown")

        wr = get_player_overall_winrate(player)
        player_winrates.append(wr if wr is not None else 0.5)

    lineup_conf = float(np.mean(player_confs)) if player_confs else 1.0
    baseline_team_prob = float(np.mean(player_winrates)) if player_winrates else 0.5

    team_vector = aggregate_team_vector(player_rows)
    X_team = np.array([[team_vector[f] for f in TEAM_FEATURES]])
    X_scaled = team_scaler.transform(X_team)
    base_prob = float(team_model.predict_proba(X_scaled)[0, 1])

    synergy_bonus = compute_synergy_bonus(selected_players)
    prob_with_synergy = max(min(base_prob + synergy_bonus, 1.0), 0.0)

    team_prob = baseline_team_prob + lineup_conf * (prob_with_synergy - baseline_team_prob)
    team_prob = max(min(team_prob, 1.0), 0.0)

    sim_probs = []
    for _ in range(sim_runs):
        noisy_vec = []
        for idx, f in enumerate(TEAM_FEATURES):
            val = X_team[0, idx]
            if f.startswith("enemy"):
                noisy_val = val
            else:
                sd = abs(val) * 0.1
                noisy_val = np.random.normal(val, sd) if sd > 1e-6 else val
                noisy_val = max(noisy_val, 0.0)
            noisy_vec.append(noisy_val)
        noisy_vec = np.array(noisy_vec).reshape(1, -1)
        noisy_scaled = team_scaler.transform(noisy_vec)
        p = float(team_model.predict_proba(noisy_scaled)[0, 1])
        p = max(min(p + synergy_bonus, 1.0), 0.0)
        p = baseline_team_prob + lineup_conf * (p - baseline_team_prob)
        p = max(min(p, 1.0), 0.0)
        sim_probs.append(p)

    sim_probs = np.array(sim_probs)
    mean_p = sim_probs.mean()
    p10 = np.percentile(sim_probs, 10)
    p90 = np.percentile(sim_probs, 90)

    summary_text = (
        "Estimated team win probability (confidence & player-winrate adjusted): "
        f"{team_prob*100:.1f}%"
    )

    detail_lines = [
        f"Based on {sim_runs} simulated games.",
        f"Mean simulated winrate: {mean_p*100:.1f}% (10th–90th percentile: {p10*100:.1f}%–{p90*100:.1f}%).",
        f"Synergy bonus applied: {synergy_bonus*100:.1f} percentage points.",
        "",
        f"Baseline team winrate from players' overall winrates: {baseline_team_prob*100:.1f}%.",
        f"Lineup data confidence (0–1): {lineup_conf:.2f} "
        f"(higher = more games in these roles).",
        "Per-role sample sizes and overall player winrates:",
    ]
    for role, player, games, conf, wr in zip(
        ROLES, selected_players, player_games, player_confs, player_winrates
    ):
        wr_pct = wr * 100 if wr is not None else 50.0
        detail_lines.append(
            f"- {role}: {player} — games={games}, role_conf={conf:.2f}, overall winrate≈{wr_pct:.1f}%"
        )

    detail_text = "\n".join(detail_lines)

    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=sim_probs * 100,
            nbinsx=20,
            name="Simulated winrates",
            opacity=0.8,
        )
    )
    hist_fig.update_layout(
        xaxis_title="Simulated winrate (%)",
        yaxis_title="Frequency",
        bargap=0.05,
        title="Distribution of simulated winrates for this composition",
    )

    return summary_text, detail_text, hist_fig


# =========================================
# CALLBACK: FIND BEST TEAM COMPOSITION
# =========================================

@app.callback(
    Output("best-team-output", "children"),
    Input("best-team-button", "n_clicks"),
    prevent_initial_call=True,
)
def find_best_team(n_clicks):
    if not n_clicks:
        return ""

    for role in ROLES:
        if not valid_players_by_role.get(role):
            return (
                f"No players have ≥{MIN_ROLE_GAMES} games in the {role} role. "
                "Cannot compute a best lineup under this constraint."
            )

    if len(all_players) < len(ROLES):
        return "Not enough distinct players to form a full team."

    best_prob = -1.0
    best_lineup = None
    best_conf = None

    for perm in permutations(all_players, len(ROLES)):
        ok = True
        for i, role in enumerate(ROLES):
            player = perm[i]
            if player not in valid_players_by_role.get(role, []):
                ok = False
                break
        if not ok:
            continue

        lineup_by_role = {role: player for role, player in zip(ROLES, perm)}
        prob, conf = compute_team_prob_for_lineup(lineup_by_role)
        if prob > best_prob:
            best_prob = prob
            best_lineup = lineup_by_role
            best_conf = conf

    if best_lineup is None:
        return (
            f"No valid lineup found where each player has ≥{MIN_ROLE_GAMES} games "
            "in their assigned role."
        )

    lines = [
        html.Div(
            "Best estimated baseline win probability "
            f"(confidence & player-winrate adjusted): {best_prob*100:.1f}%",
            style={"fontWeight": "bold", "marginBottom": "0.25rem"},
        ),
        html.Div(
            f"Lineup data confidence: {best_conf:.2f}",
            style={"marginBottom": "0.5rem"},
        ),
    ]
    for role in ROLES:
        lines.append(html.Div(f"{role}: {best_lineup[role]}"))

    return lines


if __name__ == "__main__":
    app.run(debug=True)
