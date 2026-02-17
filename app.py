"""
DFS Optimizer Pro - Updated Streamlit Application
Supports native DraftKings CSV format + all new features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from backend.data_validator import DataValidator, PlayerPool
from backend.game_modes import GameModes
from backend.optimizer import OptimizerEngine
from backend.rule_engine import RuleEngine, Rule
from backend.monte_carlo import MonteCarloSimulator
from backend.exposure_manager import ExposureManager

st.set_page_config(
    page_title="DFS Optimizer Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

DK_COLUMN_MAP = {
    "Golfer":          "Player",
    "DK Salary":       "Salary",
    "DK Points":       "Projection",
    "Large Field Own": "Ownership",
    "Small Field Own": "SmallOwn",
    "DK Ceiling":      "Ceiling",
    "Make Cut Odds":   "MakeCut",
    "DK Value":        "Value",
    "Volatility":      "Volatility",
    "id":              "ID",
}

def normalize_dk_csv(df):
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff").strip('"') for c in df.columns]
    df = df.rename(columns={k: v for k, v in DK_COLUMN_MAP.items() if k in df.columns})
    if "Position" not in df.columns:
        df["Position"] = "G"
    if "Salary" in df.columns:
        df["Salary"] = (df["Salary"].astype(str)
                        .str.replace("$", "", regex=False)
                        .str.replace(",", "", regex=False)
                        .str.strip())
    for col in ["Ownership", "SmallOwn", "MakeCut"]:
        if col in df.columns:
            df[col] = (df[col].astype(str)
                       .str.replace("%", "", regex=False).str.strip())
            df[col] = pd.to_numeric(df[col], errors="coerce") / 100
    for col in ["Salary", "Projection", "Ceiling", "Value", "Volatility"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ID" not in df.columns:
        df["ID"] = range(1, len(df) + 1)
    else:
        df["ID"] = df["ID"].astype(str)
    for col in ["Team", "Game"]:
        if col not in df.columns:
            df[col] = ""
    return df


def initialize_session_state():
    defaults = {
        "player_pool": None,
        "validated_data": None,
        "generated_lineups": None,
        "simulation_results": None,
        "exposure_manager": ExposureManager(),
        "rule_engine": RuleEngine(),
        "game_mode": "Golf Classic",
        "excluded_players": set(),
        "player_min_own": {},
        "player_max_own": {},
        "custom_rules": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_file_upload():
    st.header("1️⃣ Data Import")
    uploaded_file = st.file_uploader(
        "Upload DraftKings CSV (native DK export or custom)",
        type=["csv"],
        help="Supports native DraftKings PGA export format automatically"
    )
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = normalize_dk_csv(raw_df)
            validator = DataValidator()
            validated_df, stats = validator.validate_and_process(df)
            st.session_state.validated_data = validated_df
            st.session_state.player_pool = PlayerPool(validated_df)
            st.session_state.excluded_players = set()
            st.success(f"✅ Loaded {stats['total_players']} players")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Players", stats["total_players"])
            col2.metric("Salary Range", f"${stats['salary_min']:,.0f} – ${stats['salary_max']:,.0f}")
            col3.metric("Proj Range", f"{stats['projection_min']:.1f} – {stats['projection_max']:.1f}")
            col4.metric("Own Range", f"{stats['ownership_min']:.1%} – {stats['ownership_max']:.1%}")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback; st.code(traceback.format_exc())


def render_player_pool():
    st.header("2️⃣ Player Pool Manager")
    if st.session_state.player_pool is None:
        st.info("Upload data first.")
        return
    df = st.session_state.player_pool.get_player_data()
    excluded = st.session_state.excluded_players

    base_cols = ["Player", "Salary", "Projection", "Ownership"]
    extra_cols = [c for c in ["MakeCut", "Value", "Ceiling", "SmallOwn", "Volatility"] if c in df.columns]
    show_cols = base_cols + extra_cols

    display_df = df[show_cols].copy()
    display_df.insert(0, "Exclude", display_df.index.map(lambda i: df.loc[i, "Player"] in excluded))

    st.markdown("**Check Exclude to remove a player from all lineups.**")

    # Convert decimals to readable strings BEFORE displaying
    display_df["Ownership"] = display_df["Ownership"].apply(
        lambda x: f"{x:.1%}" if pd.notnull(x) else ""
    )
    if "MakeCut" in display_df.columns:
        display_df["MakeCut"] = display_df["MakeCut"].apply(
            lambda x: f"{x:.0%}" if pd.notnull(x) else ""
        )
    if "SmallOwn" in display_df.columns:
        display_df["SmallOwn"] = display_df["SmallOwn"].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else ""
        )

    col_cfg = {
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude"),
        "Salary":  st.column_config.NumberColumn("Salary", format="$%d"),
    }

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        column_config=col_cfg,
        disabled=[c for c in display_df.columns if c != "Exclude"],
        hide_index=True,
        key="pool_editor"
    )

    new_excluded = set()
    for i, row in edited.iterrows():
        if row["Exclude"]:
            new_excluded.add(df.loc[i, "Player"])
    st.session_state.excluded_players = new_excluded

    if new_excluded:
        st.warning(f"⛔ {len(new_excluded)} players excluded: {', '.join(sorted(new_excluded))}")

    st.subheader("Individual Ownership Bounds")
    player_names = df["Player"].tolist()
    sel = st.selectbox("Player", player_names, key="own_override_sel")
    if sel:
        pid = str(df[df["Player"] == sel].iloc[0]["ID"])
        cur_min = int(st.session_state.player_min_own.get(pid, 0) * 100)
        cur_max = int(st.session_state.player_max_own.get(pid, 1) * 100)
        c1, c2 = st.columns(2)
        new_min = c1.number_input("Min Ownership %", 0, 100, cur_min, key=f"min_{pid}")
        new_max = c2.number_input("Max Ownership %", 0, 100, cur_max, key=f"max_{pid}")
        if st.button("Save Bounds"):
            st.session_state.player_min_own[pid] = new_min / 100
            st.session_state.player_max_own[pid] = new_max / 100
            st.success(f"Set {sel}: min {new_min}%, max {new_max}%")

    if st.session_state.player_min_own or st.session_state.player_max_own:
        all_pids = set(list(st.session_state.player_min_own) + list(st.session_state.player_max_own))
        rows = []
        for pid in all_pids:
            match = df[df["ID"].astype(str) == str(pid)]
            if match.empty:
                continue
            rows.append({
                "Player": match.iloc[0]["Player"],
                "Min Own": f"{st.session_state.player_min_own.get(pid, 0):.0%}",
                "Max Own": f"{st.session_state.player_max_own.get(pid, 1):.0%}",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if st.button("Clear All Ownership Bounds"):
                st.session_state.player_min_own = {}
                st.session_state.player_max_own = {}
                st.rerun()


def render_game_mode_selector():
    st.header("3️⃣ Game Mode")
    modes = GameModes.get_all_modes()
    mode_names = list(modes.keys())
    selected = st.selectbox("Contest Type", mode_names,
                            index=mode_names.index(st.session_state.game_mode))
    st.session_state.game_mode = selected
    mode = modes[selected]
    st.info(f"**{mode.description}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Roster Size", mode.roster_size)
    c2.metric("Salary Cap", f"${mode.salary_cap:,}")
    if mode.has_captain:
        c3.metric("Captain Multiplier", f"{mode.captain_multiplier}x")


def render_optimization_settings():
    st.header("4️⃣ Optimization Settings")
    opt_mode = st.radio("Mode", ["Cash Game", "Tournament"])
    c1, c2 = st.columns(2)
    num_lineups = c1.slider("Number of Lineups", 1, 150, 20 if opt_mode == "Tournament" else 3)
    min_salary = c1.number_input("Min Salary Floor", 0, 50000, 0, 1000)
    unique_players = c1.selectbox(
        "Minimum Unique Players Per Lineup",
        options=[1, 2, 3, 4, 5, 6], index=0,
        help="Each lineup must differ from prior lineups by at least this many players"
    )
    variance_pct = c2.slider("Variance %", 0.0, 50.0, 15.0 if opt_mode == "Tournament" else 0.0, 1.0) / 100
    if opt_mode == "Tournament":
        proj_weight = c2.slider("Projection Weight %", 0, 100, 70) / 100
        own_weight  = c2.slider("Ownership Leverage %", 0, 100, 30) / 100
        own_penalty = c2.slider("High-Own Penalty Threshold %", 0, 50, 15) / 100
    else:
        proj_weight, own_weight, own_penalty = 1.0, 0.0, 1.0
    return {
        "mode": opt_mode,
        "num_lineups": num_lineups,
        "min_salary": min_salary if min_salary > 0 else None,
        "variance_pct": variance_pct,
        "projection_weight": proj_weight,
        "ownership_weight": own_weight,
        "ownership_penalty_threshold": own_penalty,
        "unique_players": unique_players,
    }


def render_exposure_controls():
    st.header("5️⃣ Exposure & Boost Controls")
    if st.session_state.player_pool is None:
        st.info("Upload data first.")
        return
    df = st.session_state.player_pool.get_player_data()
    player_names = df["Player"].tolist()
    t1, t2, t3 = st.tabs(["🔒 Lock Players", "⚡ Boost / Dock", "📊 Exposure Limits"])
    with t1:
        locked = st.multiselect("Lock into every lineup", player_names)
        if st.button("Apply Locks"):
            st.session_state.exposure_manager.clear_all()
            for name in locked:
                pid = df[df["Player"] == name].iloc[0]["ID"]
                st.session_state.exposure_manager.set_lock(pid, True)
            st.success(f"Locked {len(locked)} players")
    with t2:
        sel = st.selectbox("Player", player_names, key="boost_sel")
        boost = st.slider("Boost / Dock %", -50, 100, 0)
        if st.button("Apply Boost"):
            pid = df[df["Player"] == sel].iloc[0]["ID"]
            st.session_state.exposure_manager.set_projection_boost(pid, boost)
            st.success(f"{sel}: {boost:+d}%")
    with t3:
        sel2 = st.selectbox("Player", player_names, key="exp_sel")
        max_exp = st.slider("Max Exposure %", 0, 100, 100) / 100
        if st.button("Set Limit"):
            pid = df[df["Player"] == sel2].iloc[0]["ID"]
            st.session_state.exposure_manager.set_max_exposure(pid, max_exp)
            st.success(f"{sel2}: max {max_exp:.0%}")


def render_rule_engine():
    st.header("6️⃣ Rules Engine")
    if st.session_state.player_pool is None:
        st.info("Upload data first.")
        return
    df = st.session_state.player_pool.get_player_data()
    player_names = df["Player"].tolist()

    st.subheader("Structural Rules")
    preset_rules = st.session_state.rule_engine.get_rules_summary()
    if preset_rules:
        for r in preset_rules:
            st.write(f"• {r}")
        if st.button("Clear Structural Rules"):
            st.session_state.rule_engine.clear_rules()
            st.rerun()

    rule_type = st.selectbox("Add Structural Rule", [
        "At least one of (players)", "Max players from team",
        "Min salary threshold", "Max salary on expensive players",
    ])
    if rule_type == "At least one of (players)":
        picks = st.multiselect("Players", player_names, key="r1")
        if st.button("Add", key="add_r1") and picks:
            st.session_state.rule_engine.add_rule(Rule(
                rule_type="at_least_one_of",
                description=f"At least one of: {', '.join(picks)}",
                params={"players": picks}
            ))
            st.rerun()
    elif rule_type == "Max players from team":
        teams = [t for t in df["Team"].unique().tolist() if t]
        if teams:
            team = st.selectbox("Team", teams)
            mx = st.number_input("Max", 0, 6, 2)
            if st.button("Add", key="add_r2"):
                st.session_state.rule_engine.add_rule(Rule(
                    rule_type="max_from_team",
                    description=f"Max {mx} from {team}",
                    params={"team": team, "max_count": mx}
                ))
                st.rerun()
        else:
            st.info("No team data in your CSV.")
    elif rule_type == "Min salary threshold":
        mn = st.number_input("Min Total Salary", 0, 50000, 48000, 500)
        if st.button("Add", key="add_r3"):
            st.session_state.rule_engine.add_rule(Rule(
                rule_type="min_salary_threshold",
                description=f"Min salary ${mn:,}",
                params={"min_salary": mn}
            ))
            st.rerun()
    elif rule_type == "Max salary on expensive players":
        thr = st.number_input("Expensive threshold $", 5000, 15000, 10000, 500)
        mx_sal = st.number_input("Max $ on expensive", 0, 50000, 20000, 500)
        if st.button("Add", key="add_r4"):
            st.session_state.rule_engine.add_rule(Rule(
                rule_type="max_salary_on_expensive",
                description=f"Max ${mx_sal:,} on players >${thr:,}",
                params={"threshold": thr, "max_salary": mx_sal}
            ))
            st.rerun()

    st.divider()
    st.subheader("Conditional Rules (projection adjustments)")
    st.caption("Adjust a player's projection based on whether another player is in the lineup.")

    with st.expander("➕ Create Conditional Rule"):
        trigger = st.selectbox("IF I use…", player_names, key="trig")
        action  = st.selectbox("THEN…", [
            "Boost another player's projection",
            "Dock another player's projection",
            "Force % exposure of another player across lineups",
        ], key="act")
        target  = st.selectbox("…apply to:", player_names, key="tgt")

        if action == "Force % exposure of another player across lineups":
            amount = st.slider(
                "Use target player in this % of lineups where trigger is used",
                1, 100, 50, key="cond_amt",
                help="e.g. 75 means: whenever trigger appears, target appears in 75% of those lineups"
            )
            label = st.text_input("Label (optional)", key="cond_note",
                                  placeholder="e.g. If Scheffler, use Rory 75% of the time")
            if st.button("Save Conditional Rule"):
                st.session_state.custom_rules.append({
                    "trigger": trigger,
                    "target": target,
                    "direction": "exposure",
                    "amount": amount,
                    "label": label or f"If {trigger} → use {target} {amount}% of the time"
                })
                st.success("Rule saved!")
                st.rerun()
        else:
            amount = st.slider("By % amount", 1, 100, 10, key="cond_amt")
            label  = st.text_input("Label (optional)", key="cond_note",
                                   placeholder="e.g. Scheffler + Rory stack")
            if st.button("Save Conditional Rule"):
                direction = "boost" if "Boost" in action else "dock"
                st.session_state.custom_rules.append({
                    "trigger": trigger,
                    "target": target,
                    "direction": direction,
                    "amount": amount,
                    "label": label or f"If {trigger} → {direction} {target} {amount}%"
                })
                st.success("Rule saved!")
                st.rerun()

    if st.session_state.custom_rules:
        st.markdown("**Active Conditional Rules:**")
        for i, r in enumerate(st.session_state.custom_rules):
            c1, c2 = st.columns([5, 1])
            c1.write(f"• {r['label']}")
            if c2.button("🗑️", key=f"del_cr_{i}"):
                st.session_state.custom_rules.pop(i)
                st.rerun()


def apply_conditional_rules_to_pool(df, lineup_so_far):
    """Apply boost/dock conditional projection rules based on current lineup."""
    df = df.copy()
    lineup_names = {p["Player"] for p in lineup_so_far}
    for rule in st.session_state.custom_rules:
        # Only apply projection rules here; exposure rules handled post-generation
        if rule["direction"] in ("boost", "dock") and rule["trigger"] in lineup_names:
            mask = df["Player"] == rule["target"]
            if mask.any():
                mult = (1 + rule["amount"] / 100 if rule["direction"] == "boost"
                        else 1 - rule["amount"] / 100)
                df.loc[mask, "Projection"] *= mult
    return df


def enforce_exposure_conditional_rules(lineups, player_pool):
    """
    Post-generation: for each 'exposure' conditional rule, scan all lineups
    that contain the trigger player and enforce that the target appears in
    exactly `amount`% of those lineups.

    Strategy:
    - Find lineups WITH the trigger but WITHOUT the target.
    - Calculate how many of those need the target swapped in.
    - Swap out the lowest-projection non-locked player to fit the target.
    """
    if not st.session_state.custom_rules:
        return lineups

    exposure_rules = [r for r in st.session_state.custom_rules if r["direction"] == "exposure"]
    if not exposure_rules:
        return lineups

    lineups = [list(lu) for lu in lineups]  # deep-ish copy

    for rule in exposure_rules:
        trigger_name = rule["trigger"]
        target_name  = rule["target"]
        pct          = rule["amount"] / 100.0

        # Find target player data
        target_data = player_pool[player_pool["Player"] == target_name]
        if target_data.empty:
            continue
        target_player = target_data.iloc[0].to_dict()

        # Lineups containing the trigger
        trigger_lineups_idx = [
            i for i, lu in enumerate(lineups)
            if any(p["Player"] == trigger_name for p in lu)
        ]
        if not trigger_lineups_idx:
            continue

        needed = max(0, round(len(trigger_lineups_idx) * pct))

        # Of those, which already have the target?
        already_have = [
            i for i in trigger_lineups_idx
            if any(p["Player"] == target_name for p in lineups[i])
        ]

        if len(already_have) >= needed:
            continue  # already satisfied

        # Lineups that need the target added
        missing = [i for i in trigger_lineups_idx if i not in already_have]
        to_add  = needed - len(already_have)
        slots   = missing[:to_add]

        for idx in slots:
            lu = lineups[idx]
            # Calculate current salary without target
            current_salary = sum(p["Salary"] for p in lu)
            cap = 50000
            salary_room = cap - current_salary + min(p["Salary"] for p in lu)

            # Find the lowest-projection player that isn't the trigger or locked
            locked_names = set()
            candidates = [
                p for p in lu
                if p["Player"] != trigger_name
                and p["Player"] not in locked_names
                and p["Player"] != target_name
            ]
            if not candidates:
                continue

            candidates.sort(key=lambda x: x["Projection"])
            swap_out = candidates[0]

            new_salary = current_salary - swap_out["Salary"] + target_player.get("Salary", 0)
            if new_salary <= cap:
                lineups[idx] = [p for p in lu if p["Player"] != swap_out["Player"]]
                lineups[idx].append({
                    "ID":         target_player.get("ID", ""),
                    "Player":     target_name,
                    "Position":   target_player.get("Position", "G"),
                    "Salary":     target_player.get("Salary", 0),
                    "Projection": target_player.get("Projection", 0),
                    "Ownership":  target_player.get("Ownership", 0),
                    "Team":       target_player.get("Team", ""),
                })

    return lineups


def get_active_pool(df):
    excluded = st.session_state.excluded_players
    if excluded:
        df = df[~df["Player"].isin(excluded)].copy()
    return df.reset_index(drop=True)


def render_optimization_button(settings):
    st.header("7️⃣ Generate Lineups")
    if st.session_state.player_pool is None:
        st.warning("Upload data first.")
        return

    df_full = st.session_state.player_pool.get_player_data()
    df = get_active_pool(df_full)

    if df.empty:
        st.error("No players left after exclusions!")
        return

    rules_valid, errors = st.session_state.rule_engine.validate_all(df)
    if not rules_valid:
        for e in errors:
            st.error(e)
        return

    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        with st.spinner("Optimizing…"):
            try:
                adjusted = st.session_state.exposure_manager.apply_projection_adjustments(df)
                # Apply conditional rules (use empty lineup as starting context)
                adjusted = apply_conditional_rules_to_pool(adjusted, [])
                game_mode = GameModes.get_mode(st.session_state.game_mode)
                locked = st.session_state.exposure_manager.get_locked_players()
                optimizer = OptimizerEngine(
                    player_pool=adjusted,
                    game_mode=game_mode,
                    rule_engine=st.session_state.rule_engine,
                    min_unique_players=settings["unique_players"],
                    player_min_own=st.session_state.player_min_own,
                    player_max_own=st.session_state.player_max_own,
                )
                if settings["mode"] == "Cash Game":
                    lineups = optimizer.optimize_cash(
                        num_lineups=settings["num_lineups"],
                        min_salary=settings["min_salary"],
                        locked_players=locked,
                        variance_pct=settings["variance_pct"],
                    )
                else:
                    lineups = optimizer.optimize_tournament(
                        num_lineups=settings["num_lineups"],
                        projection_weight=settings["projection_weight"],
                        ownership_weight=settings["ownership_weight"],
                        ownership_penalty_threshold=settings["ownership_penalty_threshold"],
                        min_salary=settings["min_salary"],
                        locked_players=locked,
                        variance_pct=settings["variance_pct"],
                    )
                if not lineups:
                    st.error("No feasible lineups found. Try relaxing constraints.")
                    return

                # Apply post-generation exposure conditional rules
                lineups = enforce_exposure_conditional_rules(lineups, adjusted)

                st.session_state.generated_lineups = lineups
                st.success(f"✅ Generated {len(lineups)} unique lineups!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback; st.code(traceback.format_exc())


def render_lineup_results():
    if not st.session_state.generated_lineups:
        return

    st.header("8️⃣ Generated Lineups")
    lineups = st.session_state.generated_lineups

    # ── Sort controls ──
    c1, c2 = st.columns([2, 2])
    sort_by = c1.selectbox("Sort by", ["Projection", "Salary", "Avg Ownership", "Combinatorial Ownership"])
    show_lineups = c2.toggle("Show All Lineup Tables", value=True,
                             help="Toggle off to hide individual lineup tables and save space")

    # Build stats
    stats_list = []
    for i, lu in enumerate(lineups):
        s = OptimizerEngine.calculate_lineup_stats(lu)
        s["lineup_num"] = i + 1
        stats_list.append(s)

    key_map = {
        "Projection":             ("total_projection",      True),
        "Salary":                 ("total_salary",          True),
        "Avg Ownership":          ("avg_ownership",         False),
        "Combinatorial Ownership":("combinatorial_ownership",False),
    }
    sk, rev = key_map[sort_by]
    stats_list.sort(key=lambda x: x[sk], reverse=rev)

    # ── Summary table (always visible) ──
    st.subheader("Summary")
    summary = pd.DataFrame([{
        "Lineup":     s["lineup_num"],
        "Projection": f"{s['total_projection']:.2f}",
        "Salary":     f"${s['total_salary']:,}",
        "Avg Own":    f"{s['avg_ownership']:.1%}",
        "Comb Own":   f"{s['combinatorial_ownership']:.2%}",
    } for s in stats_list])
    st.dataframe(summary, use_container_width=True)

    # ── Individual lineup tables (toggleable) ──
    if show_lineups:
        st.subheader("Individual Lineups")

        # Respect sort order
        sorted_lineups = [lineups[s["lineup_num"] - 1] for s in stats_list]

        for rank, (stats, lu) in enumerate(zip(stats_list, sorted_lineups), start=1):
            lu_num = stats["lineup_num"]

            with st.expander(
                f"Lineup #{lu_num}  |  "
                f"Proj: {stats['total_projection']:.1f}  |  "
                f"Salary: ${stats['total_salary']:,}  |  "
                f"Avg Own: {stats['avg_ownership']:.1%}",
                expanded=True
            ):
                lu_df = pd.DataFrame(lu)
                disp_cols = [c for c in ["Player", "Position", "Salary", "Projection", "Ownership"]
                             if c in lu_df.columns]
                lu_df_disp = lu_df[disp_cols].copy()

                # Format ownership as %
                if "Ownership" in lu_df_disp.columns:
                    lu_df_disp["Ownership"] = lu_df_disp["Ownership"].apply(lambda x: f"{x:.1%}")

                # ── TOTALS ROW ──
                totals = {}
                for col in disp_cols:
                    if col == "Player":
                        totals[col] = "⚡ TOTAL"
                    elif col == "Position":
                        totals[col] = ""
                    elif col == "Salary":
                        totals[col] = f"${stats['total_salary']:,}"
                    elif col == "Projection":
                        totals[col] = f"{stats['total_projection']:.2f}"
                    elif col == "Ownership":
                        totals[col] = f"{stats['avg_ownership']:.1%} avg"

                totals_df = pd.DataFrame([totals])
                combined = pd.concat([lu_df_disp, totals_df], ignore_index=True)

                st.dataframe(combined, use_container_width=True, hide_index=True)

    # ── Export ──
    st.subheader("Export")

    c1, c2 = st.columns(2)

    # Standard CSV (internal use)
    rows = []
    for i, lu in enumerate(lineups):
        for p in lu:
            rows.append({
                "Lineup":     i + 1,
                "Player":     p["Player"],
                "Position":   p["Position"],
                "Salary":     p["Salary"],
                "Projection": p["Projection"],
                "Ownership":  f"{p['Ownership']:.1%}",
            })
    csv_standard = pd.DataFrame(rows).to_csv(index=False)
    c1.download_button(
        "⬇️ Download Summary CSV",
        csv_standard,
        "dfs_lineups_summary.csv",
        "text/csv"
    )

    # ── DraftKings Upload CSV ──
    # Format: Entry ID, Contest Name, Contest ID, Entry Fee, G, G, G, G, G, G
    # Player cells: "Player Name (PlayerID)"
    # We need the DK player ID — stored in the 'ID' column from the original CSV
    dk_rows = []
    for i, lu in enumerate(lineups):
        # Sort lineup: put highest-projection player first (cosmetic only)
        sorted_lu = sorted(lu, key=lambda x: x["Projection"], reverse=True)

        # Build player cells: "Name (ID)" — pad to 6 golfer slots
        player_cells = []
        for p in sorted_lu:
            raw_id = str(p.get("ID", "")).replace("_CPT", "")
            name   = p["Player"]
            player_cells.append(f"{name} ({raw_id})")

        # Pad to exactly 6 slots if somehow short
        while len(player_cells) < 6:
            player_cells.append("")

        dk_rows.append({
            "Entry ID":     "",
            "Contest Name": "",
            "Contest ID":   "",
            "Entry Fee":    "",
            "G":            player_cells[0],
            "G.1":          player_cells[1],
            "G.2":          player_cells[2],
            "G.3":          player_cells[3],
            "G.4":          player_cells[4],
            "G.5":          player_cells[5],
        })

    dk_df = pd.DataFrame(dk_rows)

    # Rename columns to match DK header exactly
    dk_df.columns = ["Entry ID", "Contest Name", "Contest ID", "Entry Fee",
                     "G", "G", "G", "G", "G", "G"]

    dk_csv = dk_df.to_csv(index=False)

    c2.download_button(
        "🏌️ Download DraftKings Upload CSV",
        dk_csv,
        "dk_upload.csv",
        "text/csv",
        help="Upload this file directly to DraftKings"
    )

    st.caption(
        "💡 **DraftKings Upload:** Go to your contest → My Entries → Upload Lineups → select `dk_upload.csv`. "
        "Make sure the Player IDs in your projections CSV match your DK contest's player pool."
    )


def render_exposure_analysis():
    if not st.session_state.generated_lineups:
        return
    st.header("9️⃣ Exposure Analysis")
    exp_df = st.session_state.exposure_manager.calculate_exposure(st.session_state.generated_lineups)
    if exp_df.empty:
        return
    disp = [c for c in ["Player", "Exposure", "Count", "Salary", "Projection", "Ownership"] if c in exp_df.columns]
    st.dataframe(exp_df[disp], use_container_width=True)
    fig = px.bar(exp_df.head(20), x="Player", y="Exposure",
                 color="Exposure", color_continuous_scale="Viridis",
                 title="Top 20 Players by Exposure")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def render_monte_carlo():
    if not st.session_state.generated_lineups:
        return
    st.header("🔟 Monte Carlo Simulation")
    c1, c2 = st.columns(2)
    n_sims   = c1.slider("Simulations", 1000, 20000, 10000, 1000)
    var_pct  = c2.slider("Variance %", 5, 50, 20) / 100
    field_sz = st.number_input("Field Size", 10, 10000, 100)
    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Simulating…"):
            sim = MonteCarloSimulator(variance_pct=var_pct)
            results = sim.simulate_lineups(st.session_state.generated_lineups,
                                           num_simulations=n_sims, field_size=field_sz)
            st.session_state.simulation_results = results
            st.success("✅ Done!")
    if st.session_state.simulation_results:
        results = st.session_state.simulation_results
        rows = [{"Lineup": r["lineup_idx"]+1, "Mean": f"{r['mean_score']:.2f}",
                 "Std": f"{r['std_score']:.2f}", "P10": f"{r['percentile_10']:.2f}",
                 "P90": f"{r['percentile_90']:.2f}", "Win%": f"{r['win_probability']:.2%}",
                 "Top1%": f"{r['top1_probability']:.2%}"} for r in results]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        sel = st.selectbox("View Distribution", [r["lineup_idx"]+1 for r in results])
        res = results[sel-1]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=res["simulated_scores"], nbinsx=50))
        fig.add_vline(x=res["mean_score"], line_dash="dash", line_color="red", annotation_text="Mean")
        fig.add_vline(x=res["percentile_10"], line_dash="dot", line_color="orange", annotation_text="P10")
        fig.add_vline(x=res["percentile_90"], line_dash="dot", line_color="green", annotation_text="P90")
        fig.update_layout(title=f"Lineup {sel} Distribution", xaxis_title="Score", yaxis_title="Freq")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(res["player_stats"]), use_container_width=True)


def main():
    initialize_session_state()
    st.title("⚡ DFS Optimizer Pro")
    st.markdown("**DraftKings Golf Optimizer** | MILP · Monte Carlo · Smart Ownership")
    st.markdown("---")
    with st.sidebar:
        st.header("Steps")
        st.markdown("""
1. Upload DK CSV  
2. Manage Player Pool  
3. Select Game Mode  
4. Set Optimization  
5. Exposure Controls  
6. Rules Engine  
7. Generate Lineups  
8. Analyze Results  
9. Monte Carlo  
        """)
    render_file_upload()
    if st.session_state.validated_data is not None:
        render_player_pool()
        render_game_mode_selector()
        settings = render_optimization_settings()
        render_exposure_controls()
        render_rule_engine()
        render_optimization_button(settings)
        render_lineup_results()
        render_exposure_analysis()
        render_monte_carlo()

if __name__ == "__main__":
    main()
