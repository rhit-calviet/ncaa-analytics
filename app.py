import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="NCAA March Madness Analytics",
    layout="wide",
    page_icon="🏀"
)

# GLOBAL STYLES
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'DM Sans', sans-serif;
  }
  h1, h2, h3, .big-title {
      font-family: 'Bebas Neue', sans-serif;
      letter-spacing: 0.04em;
  }
  .metric-card {
      background: linear-gradient(135deg, #0d1b2a 0%, #1b2e45 100%);
      border: 1px solid rgba(255,165,0,0.25);
      border-radius: 12px;
      padding: 20px 24px;
      text-align: center;
  }
  .metric-card .label {
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #f0a500;
      margin-bottom: 6px;
  }
  .metric-card .value {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 2.4rem;
      color: #ffffff;
      line-height: 1;
  }
  .metric-card .sub {
      font-size: 12px;
      color: #7a99b8;
      margin-top: 4px;
  }
  .team-banner {
      background: linear-gradient(90deg, #f0a500 0%, #e07b00 100%);
      border-radius: 10px;
      padding: 14px 20px;
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
  }
  .team-banner .tname {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.4rem;
      color: #0d1b2a;
      letter-spacing: 0.05em;
  }
  .team-banner .tprob {
      font-family: 'Bebas Neue', sans-serif;
      font-size: 1.6rem;
      color: #0d1b2a;
  }
  .seed-badge {
      background: rgba(255,255,255,0.15);
      border-radius: 6px;
      padding: 2px 8px;
      font-size: 11px;
      font-weight: 600;
      color: #0d1b2a;
      letter-spacing: 0.08em;
  }
  [data-testid="stSidebar"] {
      background: #0d1b2a !important;
  }
  [data-testid="stSidebar"] * {
      color: #d0dce8 !important;
  }
  .stRadio label { color: #d0dce8 !important; }
  div[data-testid="metric-container"] {
      background: #0d1b2a;
      border: 1px solid #1e3a55;
      border-radius: 10px;
      padding: 12px 16px;
  }
</style>
""", unsafe_allow_html=True)


# ASSET LOADING
@st.cache_resource
def load_assets():
    assets = {}

    # Page 1 — Selection & Seeding model
    try:
        assets['pkg'] = joblib.load('champ_models/ncaa_v3_package.joblib')
    except Exception:
        assets['pkg'] = None

    # Page 2 — Elite 8 v2 (PCA-based)
    try:
        assets['e8_pre']   = joblib.load('champ_models/elite8_preprocessor.joblib')
        assets['e8_model'] = joblib.load('champ_models/elite8_xgb_calibrated_v2.joblib')
        assets['e8_feats'] = json.load(open('champ_models/elite8_features_v2.json'))['feature_cols']
    except Exception as ex:
        assets['e8_pre']   = None
        assets['e8_model'] = None
        assets['e8_feats'] = None
        st.warning(f"Elite 8 model not found: {ex}")

    # Page 2 — Tournament Wins regressor (raw-feature XGBoost, separate preprocessing)
    try:
        assets['win_model'] = joblib.load('champ_models/tourney_wins_model_v2.joblib')
        assets['win_imp']   = joblib.load('champ_models/median_imputer.joblib')
        assets['win_cols']  = joblib.load('champ_models/model_columns.joblib')
    except Exception as ex:
        assets['win_model'] = None
        assets['win_imp']   = None
        assets['win_cols']  = None

    try:
        assets['h2h_model'] = joblib.load('champ_models/h2h_model.joblib')
    except Exception:
        assets['h2h_model'] = None

    return assets


all_assets = load_assets()


# SHARED HELPERS
MONTH_MAP = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}

def parse_wl(value):
    s = str(value).strip()
    if '-' not in s:
        return 0, 0
    
    left, right = s.split('-', 1)
    
    try:
        w = int(left)
    except ValueError:
        w = MONTH_MAP.get(left.lower(), 0)
        
    try:
        l = int(right)
    except ValueError:
        l = MONTH_MAP.get(right.lower(), 0)
        
    return w, l


def engineer_features_p1(df, pkg):
    df = df.copy()
    
    df = df.rename(columns={'NET Rank': 'NETRank'})
    
    for col in ['WL', 'Quadrant1', 'Quadrant2', 'Quadrant3', 'Quadrant4']:
        if col in df.columns:
            parsed = df[col].apply(parse_wl).apply(pd.Series)
            df[f'{col}_W'], df[f'{col}_L'] = parsed[0], parsed[1]
    
    df['Win_Pct'] = df['WL_W'] / (df['WL_W'] + df['WL_L'] + 1e-6)
    df['Quadrant1_W'] = df['Quadrant1_W'] # renaming for S1 consistency
    df['Q1_Win_Pct'] = df['Quadrant1_W'] / (df['Quadrant1_W'] + df['Quadrant1_L'] + 1e-6)
    
    df['Quality_Wins'] = df['Quadrant1_W'] + df['Quadrant2_W']
    df['Bad_Losses'] = df['Quadrant3_L'] + df['Quadrant4_L']
    
    df['PrevNET'] = pd.to_numeric(df['PrevNET'], errors='coerce').replace(0, np.nan)
    
    df['PrevNET'] = df['PrevNET'].fillna(
        df['Conference'].map(pkg['conf_med'])
    ).fillna(pkg['global_med'])
    
    known = set(pkg['le'].classes_)
    fallback_id = 0
    
    df['Conf_ID'] = df['Conference'].apply(
        lambda x: pkg['le'].transform([x])[0] if x in known else fallback_id
    )
    
    return df

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("## 🏀 NCAA Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["Selection & Seeding", "Championship-Caliber Insights"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Final Four Analytics · 2026")


# PAGE 1 — SELECTION & SEEDING
if page == "Selection & Seeding":
    st.markdown("<h1 style='margin-bottom:0'>Selection & Seeding Analysis</h1>", unsafe_allow_html=True)
    st.caption("Predicting the 2026 NCAA Tournament Field via Multi-Stage GBM")

    # Tableau dashboard
    with st.expander("Selection & Seeding Dashboard", expanded=True):
        tableau_url = (
            "https://public.tableau.com/views/FinalDashboardsNCAA_v2025_3/Dashboard1"
            "?:embed=y&:showVizHome=no"
        )
        components.html(f"""
            <div id="container" style="width:100%;overflow:hidden;">
                <iframe id="tableau-viz" src="{tableau_url}"
                    style="width:2300px;height:1800px;border:none;transform-origin:0 0;">
                </iframe>
            </div>
            <script>
                function resize() {{
                    var c = document.getElementById('container');
                    var f = document.getElementById('tableau-viz');
                    f.style.transform = 'scale(' + (c.offsetWidth/2300) + ')';
                }}
                window.addEventListener('resize', resize); resize();
            </script>
        """, height=1200)

    st.divider()

    pkg = all_assets.get('pkg')
    st.subheader("Upload 2026 Season Data")
    st.info("Upload your season CSV to generate the predicted tournament field.")
    use_default = st.checkbox("Use NCAA 2026 dataset")
    if use_default:
        df_raw = pd.read_csv("data/NCAA_Seed_Test_Set_2026_20260315.csv")
        st.success("Default dataset loaded.")
        data_ready = True
    else:
        upload = st.file_uploader("Upload CSV File", type="csv", key="p1_upload")
        if upload is not None:
            df_raw = pd.read_csv(upload)
            data_ready = True
        else:
            data_ready = False

    if data_ready and pkg:
        df = engineer_features_p1(df_raw, pkg)
        df['Is_AQ'] = (df.get('Bid Type', '') == 'AQ').astype(bool)

        st.markdown("### 🏆 Step 1 — Mark Automatic Qualifiers")
        edited_df = st.data_editor(
            df[['Team', 'Conference', 'NETRank', 'WL', 'Is_AQ']],
            column_config={"Is_AQ": st.column_config.CheckboxColumn("AQ?", default=False)},
            disabled=["Team", "Conference", "NETRank", "WL"],
            hide_index=True, use_container_width=True
        )
        df['Is_AQ'] = edited_df['Is_AQ'].astype(int)

        if st.button("Generate Predicted Field", type="primary"):
            n_aqs       = int(df['Is_AQ'].sum())
            n_al_spots  = 68 - n_aqs
            aq_df       = df[df['Is_AQ'] == 1].copy()
            aq_df['Selection_Status'] = 'Automatic Qualifier'
            aq_df['Sel_Prob'] = 1.0
            nonaq = df[df['Is_AQ'] == 0].copy()
            nonaq['Sel_Prob'] = pkg['clf'].predict_proba(nonaq[pkg['feats_s1']])[:, 1]
            selected_al = nonaq.nlargest(n_al_spots, 'Sel_Prob')
            selected_al['Selection_Status'] = 'At-Large'
            field = pd.concat([aq_df, selected_al])
            field['Seed_Score'] = pkg['reg'].predict(field[pkg['feats_s2']])
            field = field.sort_values('Seed_Score').reset_index(drop=True)
            field['Predicted_Seed'] = range(1, len(field) + 1)
            field['Seed_Line'] = ((field['Predicted_Seed'] - 1) // 4) + 1

            st.success("Field Predicted Successfully!")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Field", "68 Teams")
            c2.metric("Auto Qualifiers", str(n_aqs))
            c3.metric("At-Large Bids", str(n_al_spots))

            st.markdown("### Projected #1 Seeds")
            cols = st.columns(4)
            for i in range(min(4, len(field))):
                cols[i].metric(f"Region {i+1}", field.iloc[i]['Team'])

            st.markdown("### Full Predicted Field")
            display_field = field[['Predicted_Seed','Seed_Line','Team','Conference',
                                   'Selection_Status','NETRank']].copy()
            display_field.columns = ['Seed','Seed Line','School','Conference','Status','NET']
            st.dataframe(display_field.sort_values('Seed'), use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### Selection Drivers")
            imp = pd.DataFrame({
                'Feature': pkg['feats_s1'],
                'Importance': pkg['clf'].feature_importances_
            }).sort_values('Importance')
            fig = px.bar(imp, x='Importance', y='Feature', orientation='h',
                         title="Top Drivers for At-Large Selection",
                         color='Importance', color_continuous_scale='Oranges')
            fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                              paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    elif not pkg:
        st.error("Selection model not found. Please ensure `ncaa_v3_package.joblib` is in `champ_models/`.")


# PAGE 2 — CHAMPIONSHIP-CALIBER INSIGHTS
elif page == "Championship-Caliber Insights":
    st.markdown("<h1 style='margin-bottom:0'>Championship-Caliber Insights</h1>", unsafe_allow_html=True)
    st.caption("What separates Elite 8 teams from first-round exits?")

    e8_ready = all_assets['e8_pre'] is not None and all_assets['e8_model'] is not None

    if not e8_ready:
        st.error("Elite 8 model artifacts not found. Ensure `elite8_preprocessor.joblib` and "
                 "`elite8_xgb_calibrated_v2.joblib` are in `champ_models/`.")
        st.stop()

    with st.expander("Championship Trends Dashboard", expanded=True):
        tableau_url2 = (
            "https://public.tableau.com/views/NCAAFinals-formatteo/Dashboard4"
            "?:embed=y&:showVizHome=no"
        )
        components.html(f"""
            <div id="container2" style="width:100%;overflow:hidden;">
                <iframe id="tableau-viz2" src="{tableau_url2}"
                    style="width:2300px;height:1800px;border:none;transform-origin:0 0;">
                </iframe>
            </div>
            <script>
                function resize2() {{
                    var c = document.getElementById('container2');
                    var f = document.getElementById('tableau-viz2');
                    f.style.transform = 'scale(' + (c.offsetWidth/2300) + ')';
                }}
                window.addEventListener('resize', resize2); resize2();
            </script>
        """, height=1200)

    st.divider()

    st.text("NOTE: This is a sample of what the website could look like. " \
    "This shows what our models have already run on the current 2026 March Madness tournament. " \
    "The next step is to allow users to upload their own season data and run these analyses on demand.")


    # HEAD-TO-HEAD PREDICTOR

    @st.cache_data
    def load_h2h_data():
        base = os.path.dirname(__file__)

        close_games = pd.read_csv(
            os.path.join(base, "March_Madness_Close_Games_2026_with_Names.csv")
        )
        seed_gap = pd.read_csv(
            os.path.join(base, "Tableau_2026_Seed_Gap_Analysis.csv")
        )
        return close_games, seed_gap


    close_games_df, seed_gap_df = load_h2h_data()

    def _normalize(name: str) -> str:
        """Lower-case, strip punctuation, collapse spaces."""
        import re
        return re.sub(r"[^a-z0-9 ]", "", str(name).lower()).strip()


    _seed_lookup: dict[str, pd.Series] = {}
    for _, row in seed_gap_df.iterrows():
        _seed_lookup[_normalize(row["Team Name"])] = row

    _all_game_teams: set[str] = set(
        close_games_df["Fav_Team"].dropna().tolist()
        + close_games_df["Und_Team"].dropna().tolist()
    )

    _canonical_teams = sorted(seed_gap_df["Team Name"].dropna().unique().tolist())


    def get_seed_profile(display_name: str) -> pd.Series | None:
        key = _normalize(display_name)
        if key in _seed_lookup:
            return _seed_lookup[key]
        for k, v in _seed_lookup.items():
            if key in k or k in key:
                return v
        return None


    def get_game_row(team_a: str, team_b: str) -> pd.Series | None:

        na, nb = _normalize(team_a), _normalize(team_b)

        def _partial_match(cell: str, target: str) -> bool:
            c = _normalize(str(cell))
            # check both directions + word-level subset
            return target in c or c in target or any(
                w in c for w in target.split() if len(w) > 3
            )

        for _, row in close_games_df.iterrows():
            fav_n = str(row.get("Fav_Team", ""))
            und_n = str(row.get("Und_Team", ""))
            if _partial_match(fav_n, na) and _partial_match(und_n, nb):
                return row
            if _partial_match(fav_n, nb) and _partial_match(und_n, na):
                return row
        return None


    def predict_matchup(
        team_a: str,
        team_b: str,
        profile_a: pd.Series,
        profile_b: pd.Series,
        game_row: pd.Series | None,
    ) -> dict:
        
        score_a = 0.0
        score_b = 0.0
        factors = []

        jug_a = float(profile_a.get("Championship Caliber", 0) or 0)
        jug_b = float(profile_b.get("Championship Caliber", 0) or 0)
        jug_diff = jug_a - jug_b
        jug_weight = 3.0
        score_a += jug_diff * jug_weight if jug_diff > 0 else 0
        score_b -= jug_diff * jug_weight if jug_diff < 0 else 0
        factors.append({
            "Factor": "Championship Caliber",
            "Team A": f"{jug_a:.4f}",
            "Team B": f"{jug_b:.4f}",
            "Edge": team_a if jug_diff > 0 else (team_b if jug_diff < 0 else "Even"),
        })

        ts_a = float(profile_a.get("Model Seed", 8) or 8)
        ts_b = float(profile_b.get("Model Seed", 8) or 8)
        ts_diff = ts_b - ts_a
        ts_weight = 1.5
        score_a += ts_diff * ts_weight if ts_diff > 0 else 0
        score_b -= ts_diff * ts_weight if ts_diff < 0 else 0
        factors.append({
            "Factor": "Model Seed (lower = better)",
            "Team A": f"{int(ts_a)}",
            "Team B": f"{int(ts_b)}",
            "Edge": team_a if ts_diff > 0 else (team_b if ts_diff < 0 else "Even"),
        })

        sd_a = float(profile_a.get("Seed Delta", 0) or 0)
        sd_b = float(profile_b.get("Seed Delta", 0) or 0)
        sd_diff = sd_a - sd_b
        sd_weight = 0.8
        score_a += sd_diff * sd_weight if sd_diff > 0 else 0
        score_b -= sd_diff * sd_weight if sd_diff < 0 else 0
        factors.append({
            "Factor": "Seed Delta (hidden strength)",
            "Team A": f"{sd_a:+.1f}",
            "Team B": f"{sd_b:+.1f}",
            "Edge": team_a if sd_diff > 0 else (team_b if sd_diff < 0 else "Even"),
        })

        pred_diff_raw: float | None = None
        model_fav: str | None = None
        close_game_flag = False

        if game_row is not None:
            pred_raw = game_row.get("Predicted_Score_Diff", None)
            if pred_raw is not None and not (
                isinstance(pred_raw, float) and np.isnan(pred_raw)
            ):
                pred_diff_raw = float(pred_raw)
                fav_team_name = str(game_row.get("Fav_Team", ""))
                und_team_name = str(game_row.get("Und_Team", ""))
                model_fav = fav_team_name


                na = _normalize(team_a)
                if _normalize(fav_team_name) in na or na in _normalize(fav_team_name):
                    model_boost = abs(pred_diff_raw) * 0.5
                    if pred_diff_raw >= 0:
                        score_a += model_boost
                    else:
                        score_b += model_boost
                else:
                    model_boost = abs(pred_diff_raw) * 0.5
                    if pred_diff_raw >= 0:
                        score_b += model_boost
                    else:
                        score_a += model_boost

                close_game_flag = abs(pred_diff_raw) <= 5

                factors.append({
                    "Factor": "ML Predicted Score Diff",
                    "Team A": f"{pred_diff_raw:+.1f} (fav={fav_team_name})",
                    "Team B": "—",
                    "Edge": fav_team_name if pred_diff_raw >= 0 else und_team_name,
                })

            ht_raw = game_row.get("Half time Point Diff", None)
            eg_raw = game_row.get("End Game Point Diff", None)
            if ht_raw is not None and not (isinstance(ht_raw, float) and np.isnan(ht_raw)):
                ht = float(ht_raw)
                fav_n = _normalize(str(game_row.get("Fav_Team", "")))
                a_is_fav = _normalize(team_a) in fav_n or fav_n in _normalize(team_a)
                ht_edge = (ht > 0 and a_is_fav) or (ht < 0 and not a_is_fav)
                factors.append({
                    "Factor": "Half-Time Point Diff",
                    "Team A": f"{ht:+.0f}" if a_is_fav else f"{-ht:+.0f}",
                    "Team B": f"{-ht:+.0f}" if a_is_fav else f"{ht:+.0f}",
                    "Edge": team_a if ht_edge else team_b,
                })

        total = score_a + score_b
        win_pct_a = score_a / total if total > 0 else 0.5
        win_pct_b = 1 - win_pct_a

        if abs(score_a - score_b) < 0.5:
            winner = "Too Close to Call"
            win_pct_a = win_pct_b = 0.5
            conf_label = "Coin flip"
            colour = "#f0a500"
            close_game_flag = True
        else:
            winner = team_a if score_a > score_b else team_b
            loser  = team_b if score_a > score_b else team_a
            gap = abs(win_pct_a - 0.5) * 200   # 0-100

            if gap < 10:
                conf_label, colour = "Extremely close — potential upset", "#e84545"
                close_game_flag = True
            elif gap < 20:
                conf_label, colour = "Close game — could go either way", "#f0a500"
                close_game_flag = True
            elif gap < 35:
                conf_label, colour = "Moderate advantage", "#5bb8f5"
            else:
                conf_label, colour = "Clear favourite", "#4caf7d"

        spread_est = round(abs(jug_a - jug_b) * 60 + abs(ts_a - ts_b) * 1.2, 1)
        if pred_diff_raw is not None:
            spread_est = round((spread_est + abs(pred_diff_raw)) / 2, 1)
        spread_est = min(spread_est, 30)

        return {
            "winner": winner,
            "win_pct_a": round(win_pct_a * 100, 1),
            "win_pct_b": round(win_pct_b * 100, 1),
            "conf_label": conf_label,
            "colour": colour,
            "close_game": close_game_flag,
            "spread_est": spread_est,
            "factors": factors,
            "pred_diff_raw": pred_diff_raw,
        }

    st.divider()

    st.markdown(
        "<h2 style='font-family:Bebas Neue,sans-serif;letter-spacing:0.05em'>"
        "Head-to-Head Predictor</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        **How it works.** This tool combines the two model predictions (exported in csv files) to simulate any
        first- or second-round matchup from the 2026 NCAA Tournament field.

        - **Close-Game Model** — ML model trained on historical tournament data that
        predicts the raw score differential for each matchup, flagging games likely
        to be close or upsets.
        - **Seed Gap Analysis** — re-ranks every team based on advanced metrics
        (efficiency margins, offensive and defensive strengths, roster talent, coaching tenure,
        and more) to produce a *Model Seed* and a *Championship Caliber metric* that capture
        which teams are over- or under-seeded by the selection committee.

        The below Head-to-Head predictor weighs these two model outputs to project a winner and an estimated
        point spread, so you can see not just *who* is predicted to win, but *how convincingly* and whether an upset is lurking.
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_a, col_vs, col_b = st.columns([5, 1, 5])

    with col_a:
        team_a = st.selectbox(
            "Select Team A",
            _canonical_teams,
            index=_canonical_teams.index("Duke Blue Devils")
            if "Duke Blue Devils" in _canonical_teams
            else 0,
            key="h2h_team_a",
        )

    with col_vs:
        st.markdown(
            "<div style='text-align:center;padding-top:2rem;"
            "font-family:Bebas Neue,sans-serif;font-size:1.8rem;color:#f0a500'>VS</div>",
            unsafe_allow_html=True,
        )

    with col_b:
        default_b_idx = 1 if len(_canonical_teams) > 1 else 0
        if "Florida Gators" in _canonical_teams:
            default_b_idx = _canonical_teams.index("Florida Gators")
        team_b = st.selectbox(
            "Select Team B",
            _canonical_teams,
            index=default_b_idx,
            key="h2h_team_b",
        )

    run_btn = st.button("Predict Matchup", type="primary", use_container_width=True)

    if run_btn:
        if team_a == team_b:
            st.warning("Please select two different teams.")
            st.stop()

        # Fetch profiles
        profile_a = get_seed_profile(team_a)
        profile_b = get_seed_profile(team_b)

        if profile_a is None or profile_b is None:
            st.error("Could not find profile data for one or both teams.")
            st.stop()

        game_row = get_game_row(team_a, team_b)

        result = predict_matchup(team_a, team_b, profile_a, profile_b, game_row)

        # ── Result banner ──────────────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)

        if result["winner"] == "Too Close to Call":
            banner_html = f"""
            <div style="background:linear-gradient(135deg,#1b2e45,#0d1b2a);
                        border:2px solid {result['colour']};border-radius:14px;
                        padding:24px 32px;text-align:center;margin-bottom:16px;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:2.2rem;
                        color:{result['colour']};letter-spacing:0.06em;">
                TOO CLOSE TO CALL
            </div>
            <div style="color:#d0dce8;font-size:1rem;margin-top:6px;">
                {result['conf_label']}
            </div>
            </div>
            """
        else:
            loser_name = team_b if result["winner"] == team_a else team_a
            banner_html = f"""
            <div style="background:linear-gradient(135deg,#1b2e45,#0d1b2a);
                        border:2px solid {result['colour']};border-radius:14px;
                        padding:24px 32px;text-align:center;margin-bottom:16px;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:1rem;
                        color:#7a99b8;letter-spacing:0.1em;text-transform:uppercase;">
                Predicted Winner
            </div>
            <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;
                        color:{result['colour']};letter-spacing:0.06em;margin:4px 0;">
                {result['winner']}
            </div>
            <div style="color:#d0dce8;font-size:1rem;margin-top:2px;">
                {result['conf_label']} &nbsp;|&nbsp;
                Est. margin: <strong>~{result['spread_est']} pts</strong>
            </div>
            {"<div style='margin-top:10px;background:#e8454522;border-radius:8px;padding:6px 14px;display:inline-block;color:#e84545;font-weight:600;font-size:0.9rem;'>🚨 Upset Alert — this game could flip</div>" if result["close_game"] and result["winner"] != "Too Close to Call" else ""}
            </div>
            """

        st.markdown(banner_html, unsafe_allow_html=True)

        st.markdown("#### Team Profiles")
        c1, c2 = st.columns(2)

        def _stat_card(col, team_name, profile, colour):
            actual_seed = int(profile.get("Actual Seed", "?") or 0)
            true_seed   = int(profile.get("Model Seed", "?") or 0)
            delta       = float(profile.get("Seed Delta", 0) or 0)
            jug         = float(profile.get("Championship Caliber", 0) or 0)
            delta_str   = f"{delta:+.0f}" if delta != 0 else "±0 (correctly seeded)"
            delta_color = "#4caf7d" if delta > 0 else ("#e84545" if delta < 0 else "#7a99b8")

            col.markdown(
                f"""
                <div style="background:linear-gradient(135deg,#0d1b2a,#1b2e45);
                            border:1.5px solid {colour};border-radius:12px;
                            padding:18px 22px;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;
                            color:{colour};letter-spacing:0.05em;margin-bottom:12px;">
                    {team_name}
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:0.9rem;color:#d0dce8;">
                    <tr><td style="padding:4px 0;color:#7a99b8">Actual Seed</td>
                        <td style="text-align:right;font-weight:600">#{actual_seed}</td></tr>
                    <tr><td style="padding:4px 0;color:#7a99b8">True Seed</td>
                        <td style="text-align:right;font-weight:600">#{true_seed}</td></tr>
                    <tr><td style="padding:4px 0;color:#7a99b8">Seed Delta</td>
                        <td style="text-align:right;font-weight:600;color:{delta_color}">
                        {delta_str}</td></tr>
                    <tr><td style="padding:4px 0;color:#7a99b8">Championship Caliber</td>
                        <td style="text-align:right;font-weight:600">
                        {jug:.4f}</td></tr>
                </table>
                </div>
                """,
                unsafe_allow_html=True,
            )

        _stat_card(c1, team_a, profile_a, "#f0a500")
        _stat_card(c2, team_b, profile_b, "#5bb8f5")

        if game_row is not None:
            st.markdown("#### ML Model Data for This Matchup")
            pred = result["pred_diff_raw"]
            ht   = game_row.get("Half time Point Diff", None)
            eg   = game_row.get("End Game Point Diff", None)

            mcols = st.columns(3)
            mcols[0].metric(
                "Predicted Score Diff",
                f"{pred:+.1f} pts" if pred is not None else "N/A",
                help="Positive = Favourite wins; from the close-game ML model",
            )
        else:
            st.info(
                "ℹThis specific matchup isn't in the ML dataset — "
                "the prediction is powered purely by the Seed Gap Analysis scores.",
                icon="📌",
            )