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
    # Classic format columns
    "Golfer":          "Player",
    "DK Salary":       "Salary",
    "DK Points":       "Projection",
    "Large Field Own": "Ownership",
    "Small Field Own": "SmallOwn",
    "DK Ceiling":      "Ceiling",
    "Make Cut Odds":   "MakeCut",
    "DK Value":        "Value",
    "Volatility":      "Volatility",
    # Showdown format columns
    "Points":          "Projection",  # Showdown uses "Points" not "DK Points"
    "Tee Time":        "TeeTime",
    # Common
    "Salary":          "Salary",      # Showdown already has "Salary" not "DK Salary"
    "Ownership":       "Ownership",   # Showdown already has "Ownership"
    "Value":           "Value",       # Showdown already has "Value"
    "id":              "ID",
}

def normalize_dk_csv(df):
    df = df.copy()
    df.columns = [c.strip().lstrip("\ufeff").strip('"') for c in df.columns]
    df = df.rename(columns={k: v for k, v in DK_COLUMN_MAP.items() if k in df.columns})
    
    # Check for required columns after mapping
    required_cols = ["Player", "Salary", "Projection"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalization: {', '.join(missing)}. "
            f"Your CSV must have 'Golfer', 'Salary' (or 'DK Salary'), and 'Points' (or 'DK Points')."
        )
    
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
    
    # Fill NaN values in Salary and Projection with 0 to avoid issues
    if "Salary" in df.columns:
        df["Salary"] = df["Salary"].fillna(0).astype(int)
    if "Projection" in df.columns:
        df["Projection"] = df["Projection"].fillna(0)
    
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
        "current_settings": {},  # Stores optimization settings for persistence
        "uploaded_filename": None,  # Track uploaded file name
        "tee_time_labels": {},  # Store tee time labels: {player_id: "AM/PM" or "PM/AM"}
        "golfer_pairing_limits": [],  # Store pairing limits: [{"golfer_x": name, "golfer_y": name, "max_pct": 0.3}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_file_upload():
    st.header("1️⃣ Data Import")
    
    # Show currently loaded data if it exists
    if st.session_state.validated_data is not None and st.session_state.uploaded_filename:
        st.success(f"✅ Data loaded: **{st.session_state.uploaded_filename}**")
        validator = DataValidator()
        stats = {
            'total_players': len(st.session_state.validated_data),
            'salary_min': st.session_state.validated_data['Salary'].min(),
            'salary_max': st.session_state.validated_data['Salary'].max(),
            'projection_min': st.session_state.validated_data['Projection'].min(),
            'projection_max': st.session_state.validated_data['Projection'].max(),
            'ownership_min': st.session_state.validated_data['Ownership'].min(),
            'ownership_max': st.session_state.validated_data['Ownership'].max(),
        }
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Players", stats["total_players"])
        col2.metric("Salary Range", f"${stats['salary_min']:,.0f} – ${stats['salary_max']:,.0f}")
        col3.metric("Proj Range", f"{stats['projection_min']:.1f} – {stats['projection_max']:.1f}")
        col4.metric("Own Range", f"{stats['ownership_min']:.1%} – {stats['ownership_max']:.1%}")
        st.info("💡 **Data persists during your browser session.** Upload new file to replace, or use Reset button to clear. (Note: Closing browser tab will clear data)")
    
    uploaded_file = st.file_uploader(
        "Upload DraftKings CSV (Classic or Showdown format)",
        type=["csv"],
        help="Supports DraftKings Classic (full tournament) and Showdown (single round) formats"
    )
    
    if uploaded_file:
        try:
            raw_df = pd.read_csv(uploaded_file)
            df = normalize_dk_csv(raw_df)
            validator = DataValidator()
            validated_df, stats = validator.validate_and_process(df)
            st.session_state.validated_data = validated_df
            st.session_state.player_pool = PlayerPool(validated_df)
            st.session_state.uploaded_filename = uploaded_file.name
            st.session_state.excluded_players = set()
            st.success(f"✅ Loaded {stats['total_players']} players from **{uploaded_file.name}**")
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

    # Don't convert to strings - use column config for formatting instead
    # This allows proper numerical sorting
    col_cfg = {
        "Exclude": st.column_config.CheckboxColumn("❌ Exclude"),
        "Salary":  st.column_config.NumberColumn("Salary", format="$%d"),
        "Ownership": st.column_config.NumberColumn("Ownership", format="%.1f%%"),
        "Projection": st.column_config.NumberColumn("Projection", format="%.1f"),
    }
    
    # Add percentage formatting for optional columns if they exist
    if "MakeCut" in display_df.columns:
        col_cfg["MakeCut"] = st.column_config.NumberColumn("MakeCut", format="%.0f%%")
    
    if "SmallOwn" in display_df.columns:
        col_cfg["SmallOwn"] = st.column_config.NumberColumn("SmallOwn", format="%.1f%%")
    
    if "Value" in display_df.columns:
        col_cfg["Value"] = st.column_config.NumberColumn("Value", format="%.2f")
    
    if "Ceiling" in display_df.columns:
        col_cfg["Ceiling"] = st.column_config.NumberColumn("Ceiling", format="%.1f")
    
    if "Volatility" in display_df.columns:
        col_cfg["Volatility"] = st.column_config.NumberColumn("Volatility", format="%.1f")
    
    # Convert ownership columns from decimal to percentage (multiply by 100)
    # This way they display as percentages but sort numerically
    if "Ownership" in display_df.columns:
        display_df["Ownership"] = display_df["Ownership"] * 100
    if "MakeCut" in display_df.columns:
        display_df["MakeCut"] = display_df["MakeCut"] * 100
    if "SmallOwn" in display_df.columns:
        display_df["SmallOwn"] = display_df["SmallOwn"] * 100

    edited = st.data_editor(
        display_df,
        use_container_width=True,
        column_config=col_cfg,
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


def parse_tee_time_text(text_input, player_pool_df):
    """
    Parse pasted tee time data from PGA Tour website or manual format.
    
    Handles formats like:
    PGA Tour format:
      "7:24 a.m. - Scottie Scheffler, Xander Schauffele, Ludvig Aberg"
      "12:09 p.m. - Rory McIlroy, Patrick Cantlay, Viktor Hovland"
    
    Or simple format:
      "Scottie Scheffler - AM/PM"
      "Rory McIlroy - PM/AM"
    
    Returns dict of {player_id: label}
    """
    import re
    
    assignments = {}
    player_names = player_pool_df["Player"].tolist()
    
    # Track player tee times for Round 1 and Round 2
    player_round1 = {}  # {player_name: "AM" or "PM"}
    player_round2 = {}  # {player_name: "AM" or "PM"}
    
    current_round = None  # "Round 1" or "Round 2"
    
    lines = text_input.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Detect round headers
        if 'round 1' in line.lower() or 'thursday' in line.lower():
            current_round = "Round 1"
            continue
        elif 'round 2' in line.lower() or 'friday' in line.lower():
            current_round = "Round 2"
            continue
        
        # Skip tee headers
        if line.lower().startswith('tee '):
            continue
        
        # Try to parse times-first format: "12:54 p.m., 8:04 a.m.: Player1, Player2, Player3"
        # Times come BEFORE the colon, players AFTER
        times_first_match = re.match(
            r'(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)\s*,\s*(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)\s*:\s*(.+)',
            line,
            re.IGNORECASE
        )
        
        if times_first_match:
            thu_time = times_first_match.group(1)
            thu_am_pm = times_first_match.group(2).replace('.', '').upper()
            fri_time = times_first_match.group(3)
            fri_am_pm = times_first_match.group(4).replace('.', '').upper()
            players_str = times_first_match.group(5)
            
            # Split players by comma
            players_in_group = [p.strip() for p in players_str.split(',')]
            
            for player_str in players_in_group:
                matched_player = match_player_name(player_str, player_names)
                if matched_player:
                    player_round1[matched_player] = thu_am_pm
                    player_round2[matched_player] = fri_am_pm
            continue
        
        # Try to parse single-line format: "Player Name - 7:24 a.m., 12:09 p.m."
        # First time = Thursday, Second time = Friday
        single_line_match = re.match(
            r'(.+?)\s*[-–]\s*(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)\s*,\s*(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)',
            line,
            re.IGNORECASE
        )
        
        if single_line_match:
            player_str = single_line_match.group(1).strip()
            thu_time = single_line_match.group(2)
            thu_am_pm = single_line_match.group(3).replace('.', '').upper()
            fri_time = single_line_match.group(4)
            fri_am_pm = single_line_match.group(5).replace('.', '').upper()
            
            # Match player name
            matched_player = match_player_name(player_str, player_names)
            
            if matched_player:
                player_round1[matched_player] = thu_am_pm
                player_round2[matched_player] = fri_am_pm
            continue
        
        # Try to parse PGA Tour format: "7:24 a.m. - Player1, Player2, Player3"
        time_match = re.match(r'(\d{1,2}:\d{2})\s*(a\.?m\.?|p\.?m\.?)\s*[-–]\s*(.+)', line, re.IGNORECASE)
        
        if time_match:
            time_str = time_match.group(1)
            am_pm = time_match.group(2).replace('.', '').upper()  # "AM" or "PM"
            players_str = time_match.group(3)
            
            # Split players by comma
            players_in_group = [p.strip() for p in players_str.split(',')]
            
            for player_str in players_in_group:
                # Try to match to player pool
                matched_player = match_player_name(player_str, player_names)
                
                if matched_player and current_round:
                    if current_round == "Round 1":
                        player_round1[matched_player] = am_pm
                    elif current_round == "Round 2":
                        player_round2[matched_player] = am_pm
        
        # Also handle simple format: "Player Name - AM/PM" or "Player Name - PM/AM"
        elif 'AM/PM' in line.upper():
            label = "AM/PM"
            player_part = re.sub(r'[-,|\t]?\s*AM/PM', '', line, flags=re.IGNORECASE).strip()
            matched_player = match_player_name(player_part, player_names)
            
            if matched_player:
                player_row = player_pool_df[player_pool_df["Player"] == matched_player]
                if not player_row.empty:
                    player_id = str(player_row.iloc[0]["ID"])
                    assignments[player_id] = label
        
        elif 'PM/AM' in line.upper():
            label = "PM/AM"
            player_part = re.sub(r'[-,|\t]?\s*PM/AM', '', line, flags=re.IGNORECASE).strip()
            matched_player = match_player_name(player_part, player_names)
            
            if matched_player:
                player_row = player_pool_df[player_pool_df["Player"] == matched_player]
                if not player_row.empty:
                    player_id = str(player_row.iloc[0]["ID"])
                    assignments[player_id] = label
    
    # Now assign labels based on Round 1 and Round 2 times
    for player in player_round1.keys():
        if player in player_round2:
            r1_time = player_round1[player]
            r2_time = player_round2[player]
            
            # Determine label
            if r1_time == "AM" and r2_time == "PM":
                label = "AM/PM"
            elif r1_time == "PM" and r2_time == "AM":
                label = "PM/AM"
            else:
                # Same time both rounds - skip
                continue
            
            # Assign to player
            player_row = player_pool_df[player_pool_df["Player"] == player]
            if not player_row.empty:
                player_id = str(player_row.iloc[0]["ID"])
                assignments[player_id] = label
    
    return assignments


def match_player_name(input_name, player_pool_names):
    """
    Match an input name to the player pool, handling variations.
    Returns matched name or None.
    """
    input_clean = input_name.lower().strip()
    
    # First try exact match
    for pool_player in player_pool_names:
        if pool_player.lower() == input_clean:
            return pool_player
    
    # Try matching last name (common in player pool)
    # e.g., "McIlroy" matches "Rory McIlroy"
    input_parts = input_clean.split()
    if input_parts:
        last_name = input_parts[-1]
        for pool_player in player_pool_names:
            pool_parts = pool_player.lower().split()
            if pool_parts and pool_parts[-1] == last_name:
                # Check if first names match or one is initial
                if len(input_parts) > 1 and len(pool_parts) > 1:
                    # Both have first names, check if they match
                    if input_parts[0][0] == pool_parts[0][0]:  # Same first initial
                        return pool_player
                else:
                    return pool_player
    
    # Try substring match
    for pool_player in player_pool_names:
        if input_clean in pool_player.lower() or pool_player.lower() in input_clean:
            return pool_player
    
    return None


def render_tee_time_manager():
    """Manage tee time labels for players (AM/PM or PM/AM based on Thu/Fri tee times)"""
    st.header("2.5️⃣ Tee Time Labels")
    
    if st.session_state.player_pool is None:
        st.info("Upload data first.")
        return
    
    df = st.session_state.player_pool.get_player_data()
    
    with st.expander("ℹ️ About Tee Time Labels", expanded=False):
        st.markdown("""
        **Tee Time Labels Help:**
        - **AM/PM**: Player tees off Thursday morning, Friday afternoon
        - **PM/AM**: Player tees off Thursday afternoon, Friday morning
        
        **Why it matters:** Weather conditions, course conditions, and scoring opportunities 
        often differ between morning/afternoon rounds. Use this to balance your lineups.
        
        **How to use:**
        1. Find tee times on PGA Tour website or DraftKings
        2. Use quick import (paste data), OR manually assign labels
        3. Use the Rules Engine to set min/max players per label
        """)
    
    # Quick import section
    st.subheader("📋 Quick Import from Tee Times")
    
    with st.expander("How to import from PGA Tour website", expanded=False):
        st.markdown("""
        **The app automatically detects these formats:**
        
        **Format 1: Times First, Players After Colon** (Most Common)
        ```
        12:54 p.m., 8:04 a.m.: Chad Ramey, Alex Smalley, Pierceson Coody
        1:06 p.m., 8:16 a.m.: Kurt Kitayama, Harry Hall, Stephan Jaeger
        1:18 p.m., 8:28 a.m.: Keegan Bradley, Ryan Fox, Chris Kirk
        ```
        *(First time = Thursday, Second time = Friday)*
        
        **Format 2: Player Name First**
        ```
        Scottie Scheffler - 7:24 a.m., 12:09 p.m.
        Rory McIlroy - 12:09 p.m., 7:24 a.m.
        ```
        
        **Format 3: Grouped by Rounds**
        ```
        Round 1 - Thursday
        7:24 a.m. - Scottie Scheffler, Xander Schauffele
        
        Round 2 - Friday
        12:09 p.m. - Scottie Scheffler, Xander Schauffele
        ```
        
        **Format 4: Manual**
        ```
        Scottie Scheffler - AM/PM
        Rory McIlroy - PM/AM
        ```
        
        ---
        
        **Just copy and paste from PGA Tour - the app figures out the format!**
        """)
    
    tee_time_text = st.text_area(
        "Paste tee time data",
        height=200,
        placeholder="Example:\n12:54 p.m., 8:04 a.m.: Player1, Player2\n1:06 p.m., 8:16 a.m.: Player3, Player4\n\nJust paste from PGA Tour website!",
        help="Works with any PGA Tour tee time format - paste and click Import!"
    )
    
    col1, col2 = st.columns([1, 4])
    if col1.button("📥 Import Tee Times", type="primary", disabled=not tee_time_text):
        assignments = parse_tee_time_text(tee_time_text, df)
        
        if assignments:
            # Apply assignments
            for player_id, label in assignments.items():
                st.session_state.tee_time_labels[player_id] = label
            
            # Show detailed summary
            am_pm_count = sum(1 for label in assignments.values() if label == "AM/PM")
            pm_am_count = sum(1 for label in assignments.values() if label == "PM/AM")
            
            st.success(f"✅ Successfully imported {len(assignments)} players!")
            
            # Show breakdown
            col_a, col_b = st.columns(2)
            col_a.metric("AM/PM (Thu AM, Fri PM)", am_pm_count)
            col_b.metric("PM/AM (Thu PM, Fri AM)", pm_am_count)
            
            # Show matched players in expander
            with st.expander("View imported players"):
                am_pm_players = [
                    df[df["ID"].astype(str) == pid].iloc[0]["Player"]
                    for pid, label in assignments.items() if label == "AM/PM"
                ]
                pm_am_players = [
                    df[df["ID"].astype(str) == pid].iloc[0]["Player"]
                    for pid, label in assignments.items() if label == "PM/AM"
                ]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**AM/PM Players:**")
                    for p in sorted(am_pm_players):
                        st.write(f"- {p}")
                
                with col2:
                    st.markdown("**PM/AM Players:**")
                    for p in sorted(pm_am_players):
                        st.write(f"- {p}")
            
            st.rerun()
        else:
            st.warning("⚠️ No players matched. Make sure you copied the tee times correctly from PGA Tour.")
            st.info("💡 **Tip:** Copy the entire tee times section including 'Round 1' and 'Round 2' headers")
    
    st.divider()
    
    # Quick assign section
    st.subheader("Manual Label Assignment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Assign AM/PM Label**")
        am_pm_players = st.multiselect(
            "Select players with AM/PM tee times",
            options=df["Player"].tolist(),
            default=[p for p, label in st.session_state.tee_time_labels.items() 
                    if label == "AM/PM" and p in df["Player"].values],
            key="am_pm_select"
        )
        if st.button("✅ Set as AM/PM", key="set_ampm"):
            for player in am_pm_players:
                player_id = str(df[df["Player"] == player].iloc[0]["ID"])
                st.session_state.tee_time_labels[player_id] = "AM/PM"
            st.success(f"Set {len(am_pm_players)} players to AM/PM")
            st.rerun()
    
    with col2:
        st.markdown("**Assign PM/AM Label**")
        pm_am_players = st.multiselect(
            "Select players with PM/AM tee times",
            options=df["Player"].tolist(),
            default=[p for p, label in st.session_state.tee_time_labels.items() 
                    if label == "PM/AM" and p in df["Player"].values],
            key="pm_am_select"
        )
        if st.button("✅ Set as PM/AM", key="set_pmam"):
            for player in pm_am_players:
                player_id = str(df[df["Player"] == player].iloc[0]["ID"])
                st.session_state.tee_time_labels[player_id] = "PM/AM"
            st.success(f"Set {len(pm_am_players)} players to PM/AM")
            st.rerun()
    
    # Clear labels button
    if st.button("🗑️ Clear All Tee Time Labels", key="clear_tee_labels"):
        st.session_state.tee_time_labels = {}
        st.success("All tee time labels cleared!")
        st.rerun()
    
    # Show current label summary
    if st.session_state.tee_time_labels:
        st.subheader("Current Label Summary")
        am_pm_count = sum(1 for label in st.session_state.tee_time_labels.values() if label == "AM/PM")
        pm_am_count = sum(1 for label in st.session_state.tee_time_labels.values() if label == "PM/AM")
        unlabeled_count = len(df) - am_pm_count - pm_am_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("AM/PM Players", am_pm_count)
        col2.metric("PM/AM Players", pm_am_count)
        col3.metric("Unlabeled", unlabeled_count)
        
        # Clickable sections to view each group
        st.markdown("---")
        
        # AM/PM Players
        with st.expander(f"👁️ View AM/PM Players ({am_pm_count})", expanded=False):
            am_pm_players = []
            for player_id, label in st.session_state.tee_time_labels.items():
                if label == "AM/PM":
                    player_row = df[df["ID"].astype(str) == player_id]
                    if not player_row.empty:
                        am_pm_players.append({
                            "Player": player_row.iloc[0]["Player"],
                            "Salary": player_row.iloc[0]["Salary"],
                            "Projection": player_row.iloc[0]["Projection"]
                        })
            
            if am_pm_players:
                st.dataframe(pd.DataFrame(am_pm_players), use_container_width=True, hide_index=True)
            else:
                st.info("No AM/PM players assigned yet.")
        
        # PM/AM Players
        with st.expander(f"👁️ View PM/AM Players ({pm_am_count})", expanded=False):
            pm_am_players = []
            for player_id, label in st.session_state.tee_time_labels.items():
                if label == "PM/AM":
                    player_row = df[df["ID"].astype(str) == player_id]
                    if not player_row.empty:
                        pm_am_players.append({
                            "Player": player_row.iloc[0]["Player"],
                            "Salary": player_row.iloc[0]["Salary"],
                            "Projection": player_row.iloc[0]["Projection"]
                        })
            
            if pm_am_players:
                st.dataframe(pd.DataFrame(pm_am_players), use_container_width=True, hide_index=True)
            else:
                st.info("No PM/AM players assigned yet.")
        
        # Unlabeled Players - with quick labeling interface
        with st.expander(f"⚠️ View & Label Unlabeled Players ({unlabeled_count})", expanded=unlabeled_count > 0):
            labeled_player_ids = set(st.session_state.tee_time_labels.keys())
            unlabeled_players = []
            
            for idx, row in df.iterrows():
                player_id = str(row["ID"])
                if player_id not in labeled_player_ids:
                    unlabeled_players.append({
                        "Player": row["Player"],
                        "ID": player_id,
                        "Salary": row["Salary"],
                        "Projection": row["Projection"]
                    })
            
            if unlabeled_players:
                st.warning(f"⚠️ {len(unlabeled_players)} players don't have tee time labels yet")
                
                # Show the unlabeled players
                unlabeled_df = pd.DataFrame(unlabeled_players)
                st.dataframe(unlabeled_df[["Player", "Salary", "Projection"]], use_container_width=True, hide_index=True)
                
                st.markdown("**Quick Label Assignment:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Assign as AM/PM**")
                    am_pm_quick = st.multiselect(
                        "Select unlabeled players to assign AM/PM",
                        options=[p["Player"] for p in unlabeled_players],
                        key="unlabeled_am_pm_select"
                    )
                    if st.button("✅ Assign AM/PM", key="assign_unlabeled_ampm") and am_pm_quick:
                        for player_name in am_pm_quick:
                            # Find player ID
                            player_data = [p for p in unlabeled_players if p["Player"] == player_name]
                            if player_data:
                                st.session_state.tee_time_labels[player_data[0]["ID"]] = "AM/PM"
                        st.success(f"✅ Assigned {len(am_pm_quick)} players to AM/PM")
                        st.rerun()
                
                with col2:
                    st.markdown("**Assign as PM/AM**")
                    pm_am_quick = st.multiselect(
                        "Select unlabeled players to assign PM/AM",
                        options=[p["Player"] for p in unlabeled_players],
                        key="unlabeled_pm_am_select"
                    )
                    if st.button("✅ Assign PM/AM", key="assign_unlabeled_pmam") and pm_am_quick:
                        for player_name in pm_am_quick:
                            # Find player ID
                            player_data = [p for p in unlabeled_players if p["Player"] == player_name]
                            if player_data:
                                st.session_state.tee_time_labels[player_data[0]["ID"]] = "PM/AM"
                        st.success(f"✅ Assigned {len(pm_am_quick)} players to PM/AM")
                        st.rerun()
            else:
                st.success("✅ All players have been labeled!")


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
    
    # Load previous settings if available
    prev = st.session_state.get("current_settings", {})
    
    # Mode selection
    default_mode = prev.get("mode", "Tournament")
    mode_index = 0 if default_mode == "Cash Game" else 1
    opt_mode = st.radio("Mode", ["Cash Game", "Tournament"], index=mode_index)
    
    c1, c2 = st.columns(2)
    
    # Get default based on mode
    default_lineups = 20 if opt_mode == "Tournament" else 3
    num_lineups = c1.slider(
        "Number of Lineups", 
        min_value=1, 
        max_value=150, 
        value=prev.get("num_lineups", default_lineups)
    )
    
    min_salary = c1.number_input(
        "Min Salary Floor", 
        min_value=0, 
        max_value=50000, 
        value=prev.get("min_salary_input", 0), 
        step=1000
    )
    
    unique_players = c1.selectbox(
        "Minimum Unique Players Per Lineup",
        options=[1, 2, 3, 4, 5, 6], 
        index=prev.get("unique_players", 1) - 1,
        help="Each lineup must differ from prior lineups by at least this many players"
    )
    
    global_max_own = c1.slider(
        "Global Max Ownership % (applies to all players)",
        min_value=0, 
        max_value=100, 
        value=prev.get("global_max_own_pct", 100),
        help="No player can appear in more than this % of lineups (overrides individual settings)"
    )
    
    st.markdown("**Combinatorial Ownership Bounds**")
    ccol1, ccol2 = st.columns(2)
    min_comb_own = ccol1.slider(
        "Min Combinatorial Ownership %", 
        min_value=0, 
        max_value=100, 
        value=prev.get("min_comb_own_pct", 0),
        help="Sum of all 6 golfers' ownership percentages (e.g., 6 golfers at 10% each = 60%)"
    )
    max_comb_own = ccol2.slider(
        "Max Combinatorial Ownership %", 
        min_value=5, 
        max_value=200, 
        value=prev.get("max_comb_own_pct", 100),
        help="Sum of all 6 golfers' ownership percentages (e.g., 6 golfers at 20% each = 120%)"
    )
    
    default_var = 15.0 if opt_mode == "Tournament" else 0.0
    variance_pct = c2.slider(
        "Variance %", 
        min_value=0.0, 
        max_value=50.0, 
        value=prev.get("variance_pct_input", default_var), 
        step=1.0
    ) / 100
    
    if opt_mode == "Tournament":
        proj_weight = c2.slider(
            "Projection Weight %", 
            min_value=0, 
            max_value=100, 
            value=prev.get("proj_weight_pct", 70)
        ) / 100
        own_weight = c2.slider(
            "Ownership Leverage %", 
            min_value=0, 
            max_value=100, 
            value=prev.get("own_weight_pct", 30)
        ) / 100
        own_penalty = c2.slider(
            "High-Own Penalty Threshold %", 
            min_value=0, 
            max_value=50, 
            value=prev.get("own_penalty_pct", 15)
        ) / 100
    else:
        proj_weight, own_weight, own_penalty = 1.0, 0.0, 1.0
    
    settings = {
        "mode": opt_mode,
        "num_lineups": num_lineups,
        "min_salary": min_salary if min_salary > 0 else None,
        "min_salary_input": min_salary,
        "variance_pct": variance_pct,
        "variance_pct_input": variance_pct * 100,
        "projection_weight": proj_weight,
        "ownership_weight": own_weight,
        "ownership_penalty_threshold": own_penalty,
        "unique_players": unique_players,
        "global_max_ownership": global_max_own / 100,
        "min_combinatorial_own": min_comb_own / 100,
        "max_combinatorial_own": max_comb_own / 100,
        # Store percentage versions for persistence
        "global_max_own_pct": global_max_own,
        "min_comb_own_pct": min_comb_own,
        "max_comb_own_pct": max_comb_own,
        "proj_weight_pct": int(proj_weight * 100),
        "own_weight_pct": int(own_weight * 100),
        "own_penalty_pct": int(own_penalty * 100),
    }
    
    # Save settings for persistence
    st.session_state.current_settings = settings
    
    return settings


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
        "At least one of (players)",
        "Pick X of group (1-6 players from list)",
        "Max players from team",
        "Min salary threshold",
        "Max salary on expensive players",
        "Tee time constraints (AM/PM or PM/AM)",
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
    
    elif rule_type == "Pick X of group (1-6 players from list)":
        picks = st.multiselect("Select player group", player_names, key="r_group")
        c1, c2 = st.columns(2)
        min_count = c1.number_input("Minimum from group", 0, 6, 1, key="r_group_min")
        max_count = c2.number_input("Maximum from group", 0, 6, 6, key="r_group_max")
        if st.button("Add", key="add_r_group") and picks and min_count <= max_count:
            st.session_state.rule_engine.add_rule(Rule(
                rule_type="pick_x_of_group",
                description=f"Pick {min_count}-{max_count} of: {', '.join(picks)}",
                params={"players": picks, "min_count": min_count, "max_count": max_count}
            ))
            st.rerun()
    
    elif rule_type == "Tee time constraints (AM/PM or PM/AM)":
        if not st.session_state.tee_time_labels:
            st.warning("⚠️ No tee time labels assigned yet! Go to section 2.5 to assign labels first.")
        else:
            tee_label = st.radio("Constraint for:", ["AM/PM", "PM/AM"], key="tee_label_select")
            c1, c2 = st.columns(2)
            min_count = c1.number_input(f"Min {tee_label} players", 0, 6, 0, key="tee_min")
            max_count = c2.number_input(f"Max {tee_label} players", 0, 6, 6, key="tee_max")
            if st.button("Add Tee Time Rule", key="add_tee_rule") and min_count <= max_count:
                st.session_state.rule_engine.add_rule(Rule(
                    rule_type="tee_time_constraint",
                    description=f"{min_count}-{max_count} players with {tee_label} tee times",
                    params={
                        "label": tee_label, 
                        "min_count": min_count, 
                        "max_count": max_count,
                        "tee_time_labels": dict(st.session_state.tee_time_labels)
                    }
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
            "Boost projection of one or more players",
            "Dock projection of one or more players",
            "Force % exposure of another player across lineups",
        ], key="act")
        
        if action == "Force % exposure of another player across lineups":
            # Single target for exposure rules
            target = st.selectbox("…apply to:", player_names, key="tgt")
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
                    "targets": [target],  # Store as list for consistency
                    "direction": "exposure",
                    "amount": amount,
                    "label": label or f"If {trigger} → use {target} {amount}% of the time"
                })
                st.success("Rule saved!")
                st.rerun()
        else:
            # Multiple targets for boost/dock
            targets = st.multiselect(
                "…apply to these players:",
                options=[p for p in player_names if p != trigger],
                key="tgt_multi",
                help="Select one or more players to boost/dock when the trigger player is used"
            )
            amount = st.slider("By % amount", 1, 100, 10, key="cond_amt")
            label  = st.text_input("Label (optional)", key="cond_note",
                                   placeholder="e.g. Scheffler stack")
            if st.button("Save Conditional Rule") and targets:
                direction = "boost" if "Boost" in action else "dock"
                target_list = ", ".join(targets)
                st.session_state.custom_rules.append({
                    "trigger": trigger,
                    "targets": targets,  # Store as list
                    "direction": direction,
                    "amount": amount,
                    "label": label or f"If {trigger} → {direction} {target_list} by {amount}%"
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
    
    # ── GOLFER PAIRING LIMITS ──
    st.divider()
    st.subheader("⛓️ Golfer Pairing Limits")
    st.caption("Limit how often Golfer Y can appear in lineups that contain Golfer X")
    
    with st.expander("➕ Add Pairing Limit", expanded=False):
        st.markdown("**Example:** If Scottie is in a lineup, limit Collin to appearing in only 30% of those Scottie lineups")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            golfer_x = st.selectbox(
                "If lineup has...",
                player_names,
                key="pairing_x",
                help="Primary golfer (e.g., Scottie Scheffler)"
            )
        
        with col2:
            golfer_y = st.selectbox(
                "Then limit...",
                player_names,
                key="pairing_y",
                help="Golfer to limit (e.g., Collin Morikawa)"
            )
        
        with col3:
            max_pct = st.number_input(
                "Max % allowed",
                min_value=0,
                max_value=100,
                value=30,
                step=5,
                key="pairing_pct",
                help="Maximum % of Golfer X lineups that can also contain Golfer Y"
            )
        
        if st.button("Add Pairing Limit", type="primary", key="add_pairing"):
            if golfer_x == golfer_y:
                st.error("❌ Golfer X and Golfer Y cannot be the same")
            else:
                # Check if this pairing already exists
                existing = [p for p in st.session_state.golfer_pairing_limits 
                           if p["golfer_x"] == golfer_x and p["golfer_y"] == golfer_y]
                if existing:
                    st.warning(f"⚠️ A limit for {golfer_x} → {golfer_y} already exists. Delete it first to update.")
                else:
                    st.session_state.golfer_pairing_limits.append({
                        "golfer_x": golfer_x,
                        "golfer_y": golfer_y,
                        "max_pct": max_pct / 100  # Store as decimal
                    })
                    st.success(f"✅ Added: If {golfer_x} is in lineup → {golfer_y} in max {max_pct}% of those lineups")
                    st.rerun()
    
    # Show active pairing limits
    if st.session_state.golfer_pairing_limits:
        st.markdown("**Active Pairing Limits:**")
        for i, limit in enumerate(st.session_state.golfer_pairing_limits):
            c1, c2 = st.columns([5, 1])
            pct_display = int(limit["max_pct"] * 100)
            c1.write(f"• If **{limit['golfer_x']}** in lineup → **{limit['golfer_y']}** in max **{pct_display}%** of those lineups")
            if c2.button("🗑️", key=f"del_pairing_{i}"):
                st.session_state.golfer_pairing_limits.pop(i)
                st.rerun()
    else:
        st.info("💡 No pairing limits set. Add one above to prevent specific golfer combinations from appearing too frequently together.")


def apply_conditional_rules_to_pool(df, lineup_so_far):
    """Apply boost/dock conditional projection rules based on current lineup."""
    df = df.copy()
    lineup_names = {p["Player"] for p in lineup_so_far}
    for rule in st.session_state.custom_rules:
        # Only apply projection rules here; exposure rules handled post-generation
        if rule["direction"] in ("boost", "dock") and rule["trigger"] in lineup_names:
            # Handle both new format (targets list) and old format (target string)
            targets = rule.get("targets", [rule.get("target")]) if "targets" in rule or "target" in rule else []
            
            mult = (1 + rule["amount"] / 100 if rule["direction"] == "boost"
                    else 1 - rule["amount"] / 100)
            
            for target_name in targets:
                if target_name:
                    mask = df["Player"] == target_name
                    if mask.any():
                        df.loc[mask, "Projection"] *= mult
    return df


def enforce_global_max_ownership(lineups, max_ownership_pct):
    """
    Remove lineups to enforce that no player appears in more than max_ownership_pct of lineups.
    """
    if max_ownership_pct >= 1.0:
        return lineups
    
    max_count = int(len(lineups) * max_ownership_pct)
    if max_count <= 0:
        return lineups[:1]  # Keep at least one
    
    # Count player appearances
    player_counts = {}
    for lu in lineups:
        for p in lu:
            name = p["Player"]
            player_counts[name] = player_counts.get(name, 0) + 1
    
    # Find players over the limit
    over_limit = {name: count for name, count in player_counts.items() if count > max_count}
    
    if not over_limit:
        return lineups
    
    # Remove lineups to bring players under the limit
    # Strategy: sort lineups by projection (ascending), remove worst lineups containing over-limit players
    lineups_with_proj = [
        (i, lu, sum(p["Projection"] for p in lu))
        for i, lu in enumerate(lineups)
    ]
    lineups_with_proj.sort(key=lambda x: x[2])  # Sort by projection ascending
    
    to_remove = set()
    for name, count in over_limit.items():
        need_to_remove = count - max_count
        removed = 0
        for i, lu, proj in lineups_with_proj:
            if i in to_remove:
                continue
            if any(p["Player"] == name for p in lu):
                to_remove.add(i)
                removed += 1
                if removed >= need_to_remove:
                    break
    
    kept_lineups = [lu for i, lu in enumerate(lineups) if i not in to_remove]
    return kept_lineups if kept_lineups else lineups[:1]


def enforce_golfer_pairing_limits(lineups):
    """
    Enforce golfer pairing limits: If Golfer X is in a lineup, limit how often Golfer Y can appear.
    
    For each pairing limit:
    - Find all lineups containing Golfer X
    - Count how many also contain Golfer Y
    - If exceeds limit, remove lineups containing both until within limit
    """
    if not st.session_state.golfer_pairing_limits:
        return lineups
    
    filtered_lineups = lineups.copy()
    removal_log = []
    
    for limit in st.session_state.golfer_pairing_limits:
        golfer_x = limit["golfer_x"]
        golfer_y = limit["golfer_y"]
        max_pct = limit["max_pct"]
        
        # Find lineups containing Golfer X
        lineups_with_x = []
        for lu in filtered_lineups:
            lineup_players = [p["Player"] for p in lu]
            if golfer_x in lineup_players:
                lineups_with_x.append(lu)
        
        if not lineups_with_x:
            continue  # No lineups with Golfer X, skip this limit
        
        # Find lineups with BOTH X and Y
        lineups_with_both = []
        for lu in lineups_with_x:
            lineup_players = [p["Player"] for p in lu]
            if golfer_y in lineup_players:
                lineups_with_both.append(lu)
        
        # Calculate current percentage
        current_count = len(lineups_with_both)
        total_x_lineups = len(lineups_with_x)
        current_pct = current_count / total_x_lineups if total_x_lineups > 0 else 0
        
        # If over limit, remove excess lineups
        if current_pct > max_pct:
            allowed_count = int(total_x_lineups * max_pct)
            excess_count = current_count - allowed_count
            
            if excess_count > 0:
                # Remove excess lineups (remove from the end to preserve highest-value lineups)
                lineups_to_remove = lineups_with_both[-excess_count:]
                
                for lu in lineups_to_remove:
                    if lu in filtered_lineups:
                        filtered_lineups.remove(lu)
                
                removal_log.append({
                    "golfer_x": golfer_x,
                    "golfer_y": golfer_y,
                    "removed": excess_count,
                    "was_pct": int(current_pct * 100),
                    "now_pct": int(max_pct * 100)
                })
    
    # Store removal log in session state for display
    if removal_log:
        st.session_state.pairing_limit_removals = removal_log
    
    return filtered_lineups




def enforce_combinatorial_ownership_bounds(lineups, min_own, max_own):
    """
    Filter lineups based on combinatorial ownership bounds.
    """
    if min_own <= 0 and max_own >= 1:
        return lineups
    
    filtered = []
    for lu in lineups:
        comb_own = OptimizerEngine.calculate_combinatorial_ownership(lu)
        if min_own <= comb_own <= max_own:
            filtered.append(lu)
    
    return filtered if filtered else lineups  # Return all if none match


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
        # Handle both new format (targets list) and old format (target string)
        targets = rule.get("targets", [rule.get("target")]) if "targets" in rule or "target" in rule else []
        if not targets or not targets[0]:
            continue
        target_name = targets[0]  # Exposure rules only support single target
        pct = rule["amount"] / 100.0

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
            # Calculate current salary - ensure int conversion
            current_salary = sum(int(p.get("Salary", 0)) for p in lu)
            cap = 50000

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

            # Ensure salaries are integers
            swap_out_salary = int(swap_out.get("Salary", 0))
            target_salary = int(target_player.get("Salary", 0))
            new_salary = current_salary - swap_out_salary + target_salary
            
            if new_salary <= cap:
                lineups[idx] = [p for p in lu if p["Player"] != swap_out["Player"]]
                lineups[idx].append({
                    "ID":         str(target_player.get("ID", "")),
                    "Player":     target_name,
                    "Position":   target_player.get("Position", "G"),
                    "Salary":     target_salary,
                    "Projection": float(target_player.get("Projection", 0)),
                    "Ownership":  float(target_player.get("Ownership", 0)),
                    "Team":       target_player.get("Team", ""),
                })

    return lineups


def validate_and_fix_lineups(lineups, salary_cap=50000, roster_size=6):
    """
    Validate all lineups and remove any that violate constraints.
    This is a safety check after post-generation modifications.
    """
    valid_lineups = []
    invalid_details = []
    
    for i, lu in enumerate(lineups):
        # Check roster size
        if len(lu) != roster_size:
            invalid_details.append(f"Lineup {i+1}: Wrong roster size ({len(lu)} players)")
            continue
        
        # Check salary cap - ensure all salaries are integers
        try:
            total_salary = sum(int(p.get("Salary", 0)) for p in lu)
        except (ValueError, TypeError):
            invalid_details.append(f"Lineup {i+1}: Invalid salary data type")
            continue
            
        if total_salary > salary_cap:
            invalid_details.append(f"Lineup {i+1}: Over cap (${total_salary:,})")
            continue
        
        valid_lineups.append(lu)
    
    # Store invalid details for debugging
    if invalid_details and hasattr(st, 'session_state'):
        st.session_state.validation_errors = invalid_details
    
    return valid_lineups  # Return empty list if no valid lineups (don't return invalid ones)


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
                # Clear stale validation errors from previous runs
                if hasattr(st.session_state, 'validation_errors'):
                    delattr(st.session_state, 'validation_errors')
                
                adjusted = st.session_state.exposure_manager.apply_projection_adjustments(df)
                # Apply conditional rules (use empty lineup as starting context)
                adjusted = apply_conditional_rules_to_pool(adjusted, [])
                game_mode = GameModes.get_mode(st.session_state.game_mode)
                
                # Store game mode object for use in display
                st.session_state.current_game_mode = game_mode
                
                locked = st.session_state.exposure_manager.get_locked_players()
                
                # Generate extra lineups to account for post-filtering
                # If we have aggressive filters, we'll lose some lineups
                requested_count = settings["num_lineups"]
                generation_multiplier = 1.5  # Generate 50% more to account for filtering
                initial_generation_count = min(150, int(requested_count * generation_multiplier))
                
                optimizer = OptimizerEngine(
                    player_pool=adjusted,
                    game_mode=game_mode,
                    rule_engine=st.session_state.rule_engine,
                    min_unique_players=settings["unique_players"],
                    player_min_own=st.session_state.player_min_own,
                    player_max_own=st.session_state.player_max_own,
                )
                
                # Always use tournament mode
                lineups = optimizer.optimize_tournament(
                    num_lineups=initial_generation_count,
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
                
                initial_count = len(lineups)

                # Apply post-generation exposure conditional rules
                lineups = enforce_exposure_conditional_rules(lineups, adjusted)
                after_exposure = len(lineups)
                
                # Enforce global max ownership cap
                global_max = settings["global_max_ownership"]
                if global_max < 1.0:
                    lineups = enforce_global_max_ownership(lineups, global_max)
                after_global_max = len(lineups)
                
                # Enforce min/max combinatorial ownership
                lineups = enforce_combinatorial_ownership_bounds(
                    lineups, 
                    settings["min_combinatorial_own"],
                    settings["max_combinatorial_own"]
                )
                after_comb_own = len(lineups)
                
                # Enforce golfer pairing limits
                lineups = enforce_golfer_pairing_limits(lineups)
                after_pairing_limits = len(lineups)
                
                # Validate all lineups for salary cap and roster size violations
                lineups = validate_and_fix_lineups(
                    lineups, 
                    salary_cap=game_mode.salary_cap,
                    roster_size=game_mode.roster_size
                )
                after_validation = len(lineups)
                
                # Check if we have any valid lineups
                if not lineups:
                    st.error("❌ No valid lineups after validation! All lineups violated salary cap or roster constraints.")
                    if hasattr(st.session_state, 'validation_errors'):
                        with st.expander("🔍 Validation Errors", expanded=True):
                            for err in st.session_state.validation_errors[:10]:  # Show first 10
                                st.write(f"• {err}")
                    st.info("Try: Increase variance %, reduce conditional rules, or check your player pool data.")
                    return
                
                # Trim to requested count
                if len(lineups) > requested_count:
                    lineups = lineups[:requested_count]

                st.session_state.generated_lineups = lineups
                
                # Show diagnostic info if significant filtering occurred
                total_filtered = initial_count - after_validation
                if total_filtered > 0:
                    with st.expander("📊 Lineup Generation Details", expanded=False):
                        st.write(f"**Initial generation:** {initial_count} lineups")
                        if after_exposure < initial_count:
                            st.write(f"**After exposure rules:** {after_exposure} (-{initial_count - after_exposure})")
                        if after_global_max < after_exposure:
                            st.write(f"**After global max ownership:** {after_global_max} (-{after_exposure - after_global_max})")
                        if after_comb_own < after_global_max:
                            st.write(f"**After combinatorial ownership filter:** {after_comb_own} (-{after_global_max - after_comb_own})")
                        if after_pairing_limits < after_comb_own:
                            st.write(f"**After golfer pairing limits:** {after_pairing_limits} (-{after_comb_own - after_pairing_limits})")
                        if after_validation < after_pairing_limits:
                            st.write(f"**After validation (salary/roster):** {after_validation} (-{after_pairing_limits - after_validation})")
                        st.write(f"**Final count:** {len(lineups)} lineups")
                        
                        if len(lineups) < requested_count:
                            st.warning(f"⚠️ Requested {requested_count} but only {len(lineups)} valid lineups could be generated. Consider:")
                            st.markdown("""
                            - Increasing variance %
                            - Relaxing global max ownership %
                            - Widening combinatorial ownership bounds
                            - Reducing minimum unique players
                            - Removing some structural rules
                            """)
                
                st.success(f"✅ Generated {len(lineups)} valid unique lineups!")
                
                # Show pairing limit removals if any occurred
                if hasattr(st.session_state, 'pairing_limit_removals') and st.session_state.pairing_limit_removals:
                    with st.expander("⛓️ Golfer Pairing Limits Applied", expanded=True):
                        for removal in st.session_state.pairing_limit_removals:
                            st.write(
                                f"• **{removal['golfer_x']}** + **{removal['golfer_y']}**: "
                                f"Reduced from {removal['was_pct']}% to {removal['now_pct']}% "
                                f"({removal['removed']} lineups removed)"
                            )
                    # Clear the log for next generation
                    st.session_state.pairing_limit_removals = []
            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback; st.code(traceback.format_exc())


def render_lineup_results():
    if not st.session_state.generated_lineups:
        return

    st.header("8️⃣ Generated Lineups & Analysis")
    lineups = st.session_state.generated_lineups

    # Build stats for all lineups first
    stats_list = []
    for i, lu in enumerate(lineups):
        s = OptimizerEngine.calculate_lineup_stats(lu)
        s["lineup_num"] = i + 1
        s["lineup_data"] = lu  # Store lineup data for filtering
        stats_list.append(s)

    # ── CONTROLS SECTION ──
    st.subheader("🎛️ Controls")
    
    col1, col2 = st.columns([2, 2])
    
    # Sort dropdown
    with col1:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Ownership (High to Low)",
                "Ownership (Low to High)",
                "Projection (High to Low)",
                "Projection (Low to High)",
                "Default Order"
            ],
            help="Choose how to sort your lineups"
        )
    
    # Get all unique players from all lineups
    all_players = set()
    for lu in lineups:
        for p in lu:
            all_players.add(p["Player"])
    all_players = sorted(all_players)
    
    # Player filter
    with col2:
        selected_players = st.multiselect(
            "Filter by golfer(s)",
            options=all_players,
            default=[],
            help="Select one or more golfers to see lineups containing them"
        )
    
    # Filter lineups based on selected players
    filtered_stats = stats_list.copy()
    if selected_players:
        filtered_stats = []
        for s in stats_list:
            lineup_players = [p["Player"] for p in s["lineup_data"]]
            # Check if ALL selected players are in this lineup
            if all(player in lineup_players for player in selected_players):
                filtered_stats.append(s)
        
        if not filtered_stats:
            st.warning(f"⚠️ No lineups found containing all: {', '.join(selected_players)}")
            return
    
    # Apply sorting
    if sort_by == "Ownership (High to Low)":
        filtered_stats.sort(key=lambda x: x["combinatorial_ownership"], reverse=True)
    elif sort_by == "Ownership (Low to High)":
        filtered_stats.sort(key=lambda x: x["combinatorial_ownership"], reverse=False)
    elif sort_by == "Projection (High to Low)":
        filtered_stats.sort(key=lambda x: x["total_projection"], reverse=True)
    elif sort_by == "Projection (Low to High)":
        filtered_stats.sort(key=lambda x: x["total_projection"], reverse=False)
    # else: Default Order (no sorting)
    
    # ── OWNERSHIP ANALYTICS (when golfers selected) ──
    if selected_players:
        st.divider()
        st.subheader("📊 Ownership Analytics")
        
        # Calculate stats for filtered lineups
        total_filtered = len(filtered_stats)
        avg_ownership = sum(s["combinatorial_ownership"] for s in filtered_stats) / total_filtered
        avg_projection = sum(s["total_projection"] for s in filtered_stats) / total_filtered
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Lineups Found", total_filtered)
        col2.metric("Avg Total Ownership", f"{avg_ownership:.1%}")
        col3.metric("Avg Total Projection", f"{avg_projection:.1f} pts")
        
        st.markdown("**Other golfers in these lineups:**")
        st.caption("Shows which golfers appear alongside your selected player(s) and how often")
        
        # Build golfer appearance data
        golfer_counts = {}
        for s in filtered_stats:
            for player in s["lineup_data"]:
                player_name = player["Player"]
                if player_name not in selected_players:  # Exclude selected players
                    if player_name not in golfer_counts:
                        golfer_counts[player_name] = {
                            "count": 0,
                            "total_ownership": 0,
                            "total_projection": 0,
                            "salary": player["Salary"]
                        }
                    golfer_counts[player_name]["count"] += 1
                    golfer_counts[player_name]["total_ownership"] += player["Ownership"]
                    golfer_counts[player_name]["total_projection"] += player["Projection"]
        
        # Calculate percentages and averages
        analytics_data = []
        for player_name, data in golfer_counts.items():
            appearance_pct = (data["count"] / total_filtered) * 100
            avg_own = data["total_ownership"] / data["count"]
            avg_proj = data["total_projection"] / data["count"]
            
            analytics_data.append({
                "Golfer": player_name,
                "Appears In": f"{data['count']} lineups",
                "Appearance %": appearance_pct,
                "Avg Own": avg_own * 100,  # Convert to percentage
                "Avg Proj": avg_proj,
                "Salary": data["salary"]
            })
        
        # Sort by appearance percentage
        analytics_data.sort(key=lambda x: x["Appearance %"], reverse=True)
        
        if analytics_data:
            # Create tabs for table and chart
            tab1, tab2 = st.tabs(["📋 Table View", "📊 Chart View"])
            
            with tab1:
                # Show top golfers in table format
                analytics_df = pd.DataFrame(analytics_data)
                
                # Format for display
                display_df = analytics_df.copy()
                display_df["Appearance %"] = display_df["Appearance %"].apply(lambda x: f"{x:.1f}%")
                display_df["Avg Own"] = display_df["Avg Own"].apply(lambda x: f"{x:.1f}%")
                display_df["Avg Proj"] = display_df["Avg Proj"].apply(lambda x: f"{x:.1f}")
                display_df["Salary"] = display_df["Salary"].apply(lambda x: f"${x:,}")
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Appearance %": st.column_config.TextColumn("Appearance %", help="% of filtered lineups this golfer appears in")
                    }
                )
                
                # Highlight clustering issues
                high_correlation = [d for d in analytics_data if d["Appearance %"] > 80]
                if high_correlation:
                    st.warning(f"⚠️ **Clustering Alert:** {len(high_correlation)} golfer(s) appear in >80% of these lineups")
                    for golfer_data in high_correlation[:5]:  # Show top 5
                        st.write(f"• **{golfer_data['Golfer']}** in {golfer_data['Appearance %']:.0f}% of lineups")
            
            with tab2:
                # Bar chart of appearance rates
                chart_df = pd.DataFrame(analytics_data[:15])  # Top 15 golfers
                
                if not chart_df.empty:
                    st.bar_chart(
                        chart_df,
                        x="Golfer",
                        y="Appearance %",
                        use_container_width=True
                    )
                    st.caption("Showing top 15 golfers by appearance rate")
        else:
            st.info("No other golfers found in these lineups")
    
    # ── LINEUP CARDS ──
    st.divider()
    st.subheader(f"🎴 Lineups ({len(filtered_stats)} shown)")
    
    # Display lineups as compact cards
    for rank, stats in enumerate(filtered_stats, start=1):
        lineup = stats["lineup_data"]
        
        # Create a card-like container
        with st.container():
            # Header row with lineup number and key stats
            col1, col2, col3, col4 = st.columns([1, 2, 2, 2])
            
            col1.markdown(f"**#{stats['lineup_num']}**")
            col2.markdown(f"**{stats['total_projection']:.1f}** pts")
            col3.markdown(f"**${stats['total_salary']:,}**")
            col4.markdown(f"**{stats['combinatorial_ownership']:.1%}** total own")
            
            # Player rows - compact display
            player_cols = st.columns([3, 2, 2, 2])
            player_cols[0].caption("Golfer")
            player_cols[1].caption("Salary")
            player_cols[2].caption("Proj")
            player_cols[3].caption("Own")
            
            for player in lineup:
                player_cols = st.columns([3, 2, 2, 2])
                player_cols[0].write(player["Player"])
                player_cols[1].write(f"${player['Salary']:,}")
                player_cols[2].write(f"{player['Projection']:.1f}")
                player_cols[3].write(f"{player['Ownership']:.1%}")
            
            st.divider()
    
    # ── EXPORT SECTION ──
    st.subheader("📥 Export")
    
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
    
    # Format for display
    display_df = exp_df.copy()
    if "Exposure" in display_df.columns:
        display_df["Exposure"] = display_df["Exposure"].apply(lambda x: f"{x:.1%}")
    if "Ownership" in display_df.columns:
        display_df["Ownership"] = display_df["Ownership"].apply(lambda x: f"{x:.1%}")
    
    disp = [c for c in ["Player", "Exposure", "Count", "Salary", "Projection", "Ownership"] if c in display_df.columns]
    st.dataframe(display_df[disp], use_container_width=True)
    
    # Use original exp_df (with numeric values) for the chart
    fig = px.bar(exp_df.head(20), x="Player", y="Exposure",
                 color="Exposure", color_continuous_scale="Viridis",
                 title="Top 20 Players by Exposure")
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_yaxes(tickformat=".0%")  # Format y-axis as percentage
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
    
    # Reset Options
    with st.expander("🔄 Reset Options", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Reset Settings Only**")
            st.caption("Clears optimization settings but keeps uploaded player data")
            if st.button("🔄 Reset Settings", type="secondary", key="reset_settings"):
                # Clear settings but keep player data
                st.session_state.current_settings = {}
                st.session_state.excluded_players = set()
                st.session_state.player_min_own = {}
                st.session_state.player_max_own = {}
                st.session_state.custom_rules = []
                st.session_state.tee_time_labels = {}
                st.session_state.exposure_manager = ExposureManager()
                st.session_state.rule_engine = RuleEngine()
                st.session_state.generated_lineups = None
                st.session_state.simulation_results = None
                st.success("✅ Settings reset! Player data retained.")
                st.rerun()
        
        with col2:
            st.markdown("**Reset Everything**")
            st.caption("Clears uploaded data AND all settings")
            if st.button("🗑️ Reset All", type="secondary", key="reset_all"):
                # Clear ALL data and settings
                st.session_state.current_settings = {}
                st.session_state.player_pool = None
                st.session_state.validated_data = None
                st.session_state.uploaded_filename = None
                st.session_state.excluded_players = set()
                st.session_state.player_min_own = {}
                st.session_state.player_max_own = {}
                st.session_state.custom_rules = []
                st.session_state.tee_time_labels = {}
                st.session_state.exposure_manager = ExposureManager()
                st.session_state.rule_engine = RuleEngine()
                st.session_state.generated_lineups = None
                st.session_state.simulation_results = None
                st.success("✅ All data and settings cleared!")
                st.rerun()
    
    st.markdown("---")
    with st.sidebar:
        st.header("Steps")
        st.markdown("""
1. Upload DK CSV  
2. Manage Player Pool  
2.5. Tee Time Labels
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
        render_tee_time_manager()
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
