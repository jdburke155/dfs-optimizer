"""
DFS Optimizer - Main Streamlit Application
Production-ready Daily Fantasy Sports optimizer for DraftKings
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.data_validator import DataValidator, PlayerPool
from backend.game_modes import GameModes, GameMode
from backend.optimizer import OptimizerEngine
from backend.rule_engine import RuleEngine, Rule
from backend.monte_carlo import MonteCarloSimulator
from backend.exposure_manager import ExposureManager


# Page configuration
st.set_page_config(
    page_title="DFS Optimizer Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Initialize all session state variables"""
    if 'player_pool' not in st.session_state:
        st.session_state.player_pool = None
    if 'validated_data' not in st.session_state:
        st.session_state.validated_data = None
    if 'generated_lineups' not in st.session_state:
        st.session_state.generated_lineups = None
    if 'simulation_results' not in st.session_state:
        st.session_state.simulation_results = None
    if 'exposure_manager' not in st.session_state:
        st.session_state.exposure_manager = ExposureManager()
    if 'rule_engine' not in st.session_state:
        st.session_state.rule_engine = RuleEngine()
    if 'game_mode' not in st.session_state:
        st.session_state.game_mode = 'Golf Classic'


def render_header():
    """Render application header"""
    st.title("⚡ DFS Optimizer Pro")
    st.markdown("**Professional DraftKings Golf Optimizer** | MILP-Based | Tournament & Cash Games")
    st.markdown("---")


def render_file_upload():
    """Render file upload section"""
    st.header("1️⃣ Data Import")
    
    uploaded_file = st.file_uploader(
        "Upload Player CSV",
        type=['csv'],
        help="Required columns: Player, Position, Salary, Projection, Ownership"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate data
            validator = DataValidator()
            validated_df, stats = validator.validate_and_process(df)
            
            # Store in session state
            st.session_state.validated_data = validated_df
            st.session_state.player_pool = PlayerPool(validated_df)
            
            # Display validation results
            st.success(f"✅ Successfully loaded {stats['total_players']} players")
            
            # Show warnings if any
            if validator.warnings:
                for warning in validator.warnings:
                    st.warning(warning)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Players", stats['total_players'])
            with col2:
                st.metric("Salary Range", f"${stats['salary_min']:,} - ${stats['salary_max']:,}")
            with col3:
                st.metric("Proj Range", f"{stats['projection_min']:.1f} - {stats['projection_max']:.1f}")
            with col4:
                st.metric("Ownership Range", f"{stats['ownership_min']:.1%} - {stats['ownership_max']:.1%}")
            
            # Position breakdown
            with st.expander("📊 Position Breakdown"):
                pos_df = pd.DataFrame.from_dict(stats['position_breakdown'], orient='index', columns=['Count'])
                st.dataframe(pos_df)
            
            # Show duplicates if any
            if stats['duplicates'] > 0:
                st.warning(f"⚠️ Found {stats['duplicates']} duplicate players")
            
            # Show editable player table
            with st.expander("✏️ Edit Player Data"):
                render_player_editor()
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")


def render_player_editor():
    """Render player data editor"""
    if st.session_state.player_pool is None:
        return
    
    df = st.session_state.player_pool.get_player_data()
    
    st.write("**Edit projections and ownership:**")
    
    # Select player to edit
    player_names = df['Player'].tolist()
    selected_player = st.selectbox("Select Player", player_names, key="edit_player")
    
    if selected_player:
        player_data = df[df['Player'] == selected_player].iloc[0]
        player_id = player_data['ID']
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_proj = st.number_input(
                "Projection",
                value=float(player_data['Projection']),
                step=0.5,
                key=f"proj_{player_id}"
            )
            
            if st.button("Update Projection", key=f"update_proj_{player_id}"):
                st.session_state.player_pool.update_projection(player_id, new_proj)
                st.success("Projection updated!")
                st.rerun()
        
        with col2:
            new_own = st.number_input(
                "Ownership %",
                value=float(player_data['Ownership'] * 100),
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key=f"own_{player_id}"
            )
            
            if st.button("Update Ownership", key=f"update_own_{player_id}"):
                st.session_state.player_pool.update_ownership(player_id, new_own / 100)
                st.success("Ownership updated!")
                st.rerun()


def render_game_mode_selector():
    """Render game mode selection"""
    st.header("2️⃣ Game Mode")
    
    modes = GameModes.get_all_modes()
    mode_names = list(modes.keys())
    
    selected_mode = st.selectbox(
        "Select Contest Type",
        mode_names,
        index=mode_names.index(st.session_state.game_mode),
        help="Choose the DraftKings contest format"
    )
    
    st.session_state.game_mode = selected_mode
    
    # Show mode details
    mode = modes[selected_mode]
    st.info(f"**{mode.description}**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Roster Size", mode.roster_size)
    with col2:
        st.metric("Salary Cap", f"${mode.salary_cap:,}")
    with col3:
        if mode.has_captain:
            st.metric("Captain Multiplier", f"{mode.captain_multiplier}x")


def render_optimization_settings():
    """Render optimization settings"""
    st.header("3️⃣ Optimization Settings")
    
    # Optimization mode
    opt_mode = st.radio(
        "Optimization Mode",
        ["Cash Game", "Tournament"],
        help="Cash: Pure projection max | Tournament: Projection + ownership leverage"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_lineups = st.slider(
            "Number of Lineups",
            min_value=1,
            max_value=150,
            value=20 if opt_mode == "Tournament" else 3,
            help="Generate 1-150 unique lineups"
        )
        
        min_salary = st.number_input(
            "Minimum Salary Floor",
            min_value=0,
            max_value=50000,
            value=0,
            step=1000,
            help="Optional minimum total salary"
        )
    
    with col2:
        variance_pct = st.slider(
            "Projection Variance %",
            min_value=0.0,
            max_value=50.0,
            value=15.0 if opt_mode == "Tournament" else 0.0,
            step=1.0,
            help="Random variance for projection diversification (0% = deterministic)"
        ) / 100
        
        if opt_mode == "Tournament":
            proj_weight = st.slider(
                "Projection Weight %",
                min_value=0,
                max_value=100,
                value=70,
                help="Weight given to raw projection"
            ) / 100
            
            own_weight = st.slider(
                "Ownership Leverage Weight %",
                min_value=0,
                max_value=100,
                value=30,
                help="Weight given to ownership leverage"
            ) / 100
            
            own_penalty = st.slider(
                "High Ownership Penalty Threshold %",
                min_value=0,
                max_value=50,
                value=15,
                help="Ownership % above which to penalize"
            ) / 100
        else:
            proj_weight = 1.0
            own_weight = 0.0
            own_penalty = 1.0
    
    return {
        'mode': opt_mode,
        'num_lineups': num_lineups,
        'min_salary': min_salary if min_salary > 0 else None,
        'variance_pct': variance_pct,
        'projection_weight': proj_weight,
        'ownership_weight': own_weight,
        'ownership_penalty_threshold': own_penalty
    }


def render_exposure_controls():
    """Render exposure management controls"""
    st.header("4️⃣ Exposure & Player Controls")
    
    if st.session_state.player_pool is None:
        st.warning("Please upload player data first")
        return
    
    df = st.session_state.player_pool.get_player_data()
    
    tab1, tab2, tab3 = st.tabs(["Lock/Unlock", "Boost/Dock", "Exposure Limits"])
    
    with tab1:
        st.subheader("Lock Players (100% Exposure)")
        
        player_names = df['Player'].tolist()
        selected_lock = st.multiselect(
            "Select players to lock",
            player_names,
            help="These players will appear in every lineup"
        )
        
        if st.button("Apply Locks"):
            st.session_state.exposure_manager.clear_all()
            for player_name in selected_lock:
                player_id = df[df['Player'] == player_name].iloc[0]['ID']
                st.session_state.exposure_manager.set_lock(player_id, True)
            st.success(f"Locked {len(selected_lock)} players")
    
    with tab2:
        st.subheader("Boost/Dock Projections")
        
        selected_player = st.selectbox("Select Player", player_names, key="boost_player")
        
        if selected_player:
            boost_pct = st.slider(
                "Boost/Dock %",
                min_value=-50,
                max_value=100,
                value=0,
                help="Positive = boost, Negative = dock"
            )
            
            if st.button("Apply Boost/Dock"):
                player_id = df[df['Player'] == selected_player].iloc[0]['ID']
                st.session_state.exposure_manager.set_projection_boost(player_id, boost_pct)
                st.success(f"Applied {boost_pct:+d}% adjustment to {selected_player}")
    
    with tab3:
        st.subheader("Exposure Limits")
        
        selected_player_exp = st.selectbox("Select Player", player_names, key="exp_player")
        
        if selected_player_exp:
            max_exp = st.slider(
                "Maximum Exposure %",
                min_value=0,
                max_value=100,
                value=100,
                help="Maximum % of lineups this player can appear in"
            ) / 100
            
            if st.button("Set Max Exposure"):
                player_id = df[df['Player'] == selected_player_exp].iloc[0]['ID']
                st.session_state.exposure_manager.set_max_exposure(player_id, max_exp)
                st.success(f"Set max exposure {max_exp:.0%} for {selected_player_exp}")


def render_rule_engine():
    """Render custom rule engine"""
    st.header("5️⃣ Custom Rules")
    
    if st.session_state.player_pool is None:
        st.warning("Please upload player data first")
        return
    
    df = st.session_state.player_pool.get_player_data()
    
    # Display current rules
    rules = st.session_state.rule_engine.get_rules_summary()
    if rules:
        st.write("**Active Rules:**")
        for rule in rules:
            st.write(f"- {rule}")
        
        if st.button("Clear All Rules"):
            st.session_state.rule_engine.clear_rules()
            st.success("All rules cleared")
            st.rerun()
    
    # Add new rule
    st.subheader("Add New Rule")
    
    rule_type = st.selectbox(
        "Rule Type",
        [
            "At least one of (players)",
            "Max from team",
            "Min salary threshold",
            "Max salary on expensive players"
        ]
    )
    
    if rule_type == "At least one of (players)":
        selected_players = st.multiselect(
            "Select Players",
            df['Player'].tolist()
        )
        
        if st.button("Add Rule", key="add_rule_1") and selected_players:
            rule = Rule(
                rule_type="at_least_one_of",
                description=f"At least one of: {', '.join(selected_players)}",
                params={'players': selected_players}
            )
            st.session_state.rule_engine.add_rule(rule)
            st.success("Rule added!")
            st.rerun()
    
    elif rule_type == "Max from team":
        teams = df['Team'].unique().tolist()
        team = st.selectbox("Team", teams)
        max_count = st.number_input("Max players", min_value=0, max_value=6, value=2)
        
        if st.button("Add Rule", key="add_rule_2") and team:
            rule = Rule(
                rule_type="max_from_team",
                description=f"Max {max_count} from {team}",
                params={'team': team, 'max_count': max_count}
            )
            st.session_state.rule_engine.add_rule(rule)
            st.success("Rule added!")
            st.rerun()
    
    elif rule_type == "Min salary threshold":
        min_sal = st.number_input("Minimum Total Salary", min_value=0, max_value=50000, value=48000, step=1000)
        
        if st.button("Add Rule", key="add_rule_3"):
            rule = Rule(
                rule_type="min_salary_threshold",
                description=f"Minimum salary: ${min_sal:,}",
                params={'min_salary': min_sal}
            )
            st.session_state.rule_engine.add_rule(rule)
            st.success("Rule added!")
            st.rerun()
    
    elif rule_type == "Max salary on expensive players":
        threshold = st.number_input("Expensive threshold", min_value=5000, max_value=15000, value=10000, step=1000)
        max_sal = st.number_input("Max salary on expensive", min_value=0, max_value=50000, value=20000, step=1000)
        
        if st.button("Add Rule", key="add_rule_4"):
            rule = Rule(
                rule_type="max_salary_on_expensive",
                description=f"Max ${max_sal:,} on players >${threshold:,}",
                params={'threshold': threshold, 'max_salary': max_sal}
            )
            st.session_state.rule_engine.add_rule(rule)
            st.success("Rule added!")
            st.rerun()


def render_optimization_button(settings):
    """Render optimize button and execute optimization"""
    st.header("6️⃣ Generate Lineups")
    
    if st.session_state.player_pool is None:
        st.warning("Please upload player data first")
        return
    
    # Validate rules
    df = st.session_state.player_pool.get_player_data()
    rules_valid, errors = st.session_state.rule_engine.validate_all(df)
    
    if not rules_valid:
        st.error("❌ Rule validation failed:")
        for error in errors:
            st.error(error)
        return
    
    if st.button("🚀 Generate Lineups", type="primary", use_container_width=True):
        with st.spinner("Optimizing lineups..."):
            try:
                # Apply projection boosts
                adjusted_pool = st.session_state.exposure_manager.apply_projection_adjustments(df)
                
                # Get game mode
                game_mode = GameModes.get_mode(st.session_state.game_mode)
                
                # Create optimizer
                optimizer = OptimizerEngine(
                    player_pool=adjusted_pool,
                    game_mode=game_mode,
                    rule_engine=st.session_state.rule_engine
                )
                
                # Get locked players
                locked_players = st.session_state.exposure_manager.get_locked_players()
                
                # Optimize based on mode
                if settings['mode'] == "Cash Game":
                    lineups = optimizer.optimize_cash(
                        num_lineups=settings['num_lineups'],
                        min_salary=settings['min_salary'],
                        locked_players=locked_players,
                        variance_pct=settings['variance_pct']
                    )
                else:
                    lineups = optimizer.optimize_tournament(
                        num_lineups=settings['num_lineups'],
                        projection_weight=settings['projection_weight'],
                        ownership_weight=settings['ownership_weight'],
                        ownership_penalty_threshold=settings['ownership_penalty_threshold'],
                        min_salary=settings['min_salary'],
                        locked_players=locked_players,
                        variance_pct=settings['variance_pct']
                    )
                
                if not lineups:
                    st.error("❌ No feasible lineups found. Try relaxing constraints.")
                    return
                
                # Store lineups
                st.session_state.generated_lineups = lineups
                
                st.success(f"✅ Generated {len(lineups)} unique lineups!")
                
            except Exception as e:
                st.error(f"❌ Optimization error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())


def render_lineup_results():
    """Render generated lineup results"""
    if st.session_state.generated_lineups is None:
        return
    
    st.header("7️⃣ Generated Lineups")
    
    lineups = st.session_state.generated_lineups
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Projection", "Salary", "Ownership", "Combinatorial Ownership"]
    )
    
    # Calculate stats for all lineups
    lineup_stats = []
    for i, lineup in enumerate(lineups):
        stats = OptimizerEngine.calculate_lineup_stats(lineup)
        stats['lineup_num'] = i + 1
        lineup_stats.append(stats)
    
    # Sort
    if sort_by == "Projection":
        lineup_stats.sort(key=lambda x: x['total_projection'], reverse=True)
    elif sort_by == "Salary":
        lineup_stats.sort(key=lambda x: x['total_salary'], reverse=True)
    elif sort_by == "Ownership":
        lineup_stats.sort(key=lambda x: x['avg_ownership'])
    else:
        lineup_stats.sort(key=lambda x: x['combinatorial_ownership'])
    
    # Display summary table
    summary_data = []
    for stats in lineup_stats:
        summary_data.append({
            'Lineup': stats['lineup_num'],
            'Projection': f"{stats['total_projection']:.2f}",
            'Salary': f"${stats['total_salary']:,}",
            'Avg Own': f"{stats['avg_ownership']:.1%}",
            'Comb Own': f"{stats['combinatorial_ownership']:.2%}"
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
    
    # Detailed lineup view
    st.subheader("Detailed Lineups")
    
    selected_lineup_num = st.selectbox(
        "Select Lineup to View",
        [s['lineup_num'] for s in lineup_stats]
    )
    
    selected_lineup = lineups[selected_lineup_num - 1]
    
    lineup_df = pd.DataFrame(selected_lineup)
    st.dataframe(lineup_df, use_container_width=True)
    
    # Download lineups
    st.subheader("Export Lineups")
    
    # Create export dataframe
    export_data = []
    for i, lineup in enumerate(lineups):
        for player in lineup:
            export_data.append({
                'Lineup': i + 1,
                'Player': player['Player'],
                'Position': player['Position'],
                'Salary': player['Salary'],
                'Projection': player['Projection'],
                'Ownership': f"{player['Ownership']:.1%}"
            })
    
    export_df = pd.DataFrame(export_data)
    csv = export_df.to_csv(index=False)
    
    st.download_button(
        label="Download Lineups (CSV)",
        data=csv,
        file_name="dfs_lineups.csv",
        mime="text/csv"
    )


def render_exposure_analysis():
    """Render exposure analysis"""
    if st.session_state.generated_lineups is None:
        return
    
    st.header("8️⃣ Exposure Analysis")
    
    lineups = st.session_state.generated_lineups
    exposure_df = st.session_state.exposure_manager.calculate_exposure(lineups)
    
    if exposure_df.empty:
        st.warning("No exposure data available")
        return
    
    # Display exposure table
    st.dataframe(
        exposure_df[['Player', 'Exposure', 'Count', 'Salary', 'Projection', 'Ownership']],
        use_container_width=True
    )
    
    # Exposure distribution chart
    fig = px.bar(
        exposure_df.head(20),
        x='Player',
        y='Exposure',
        title='Top 20 Players by Exposure',
        labels={'Exposure': 'Exposure %'},
        color='Exposure',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Exposure statistics
    stats = ExposureManager.analyze_exposure_distribution(exposure_df)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Players Used", stats['total_players_used'])
    with col2:
        st.metric("Avg Exposure", f"{stats['avg_exposure']:.1%}")
    with col3:
        st.metric("Concentrated (>50%)", stats['concentrated_players'])
    with col4:
        st.metric("Rare (<10%)", stats['rare_players'])


def render_monte_carlo():
    """Render Monte Carlo simulation"""
    if st.session_state.generated_lineups is None:
        return
    
    st.header("9️⃣ Monte Carlo Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_simulations = st.slider(
            "Number of Simulations",
            min_value=1000,
            max_value=20000,
            value=10000,
            step=1000,
            help="More simulations = more accurate but slower"
        )
    
    with col2:
        variance_pct = st.slider(
            "Variance %",
            min_value=5,
            max_value=50,
            value=20,
            help="Standard deviation as % of projection"
        ) / 100
    
    field_size = st.number_input(
        "Tournament Field Size",
        min_value=10,
        max_value=10000,
        value=100,
        help="Number of entries in tournament"
    )
    
    if st.button("Run Simulation", type="primary"):
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                simulator = MonteCarloSimulator(variance_pct=variance_pct)
                
                lineups = st.session_state.generated_lineups
                results = simulator.simulate_lineups(
                    lineups,
                    num_simulations=num_simulations,
                    field_size=field_size
                )
                
                st.session_state.simulation_results = results
                
                st.success("✅ Simulation complete!")
                
            except Exception as e:
                st.error(f"❌ Simulation error: {str(e)}")
    
    # Display results
    if st.session_state.simulation_results is not None:
        render_simulation_results()


def render_simulation_results():
    """Display Monte Carlo simulation results"""
    results = st.session_state.simulation_results
    
    st.subheader("Simulation Results")
    
    # Summary table
    summary_data = []
    for result in results:
        summary_data.append({
            'Lineup': result['lineup_idx'] + 1,
            'Mean Score': f"{result['mean_score']:.2f}",
            'Std Dev': f"{result['std_score']:.2f}",
            '10th %ile': f"{result['percentile_10']:.2f}",
            '90th %ile': f"{result['percentile_90']:.2f}",
            '99th %ile': f"{result['percentile_99']:.2f}",
            'Win Prob': f"{result['win_probability']:.2%}",
            'Top 1%': f"{result['top1_probability']:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Select lineup for detailed view
    selected_sim_lineup = st.selectbox(
        "Select Lineup for Detailed View",
        [i + 1 for i in range(len(results))]
    )
    
    result = results[selected_sim_lineup - 1]
    
    # Score distribution
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=result['simulated_scores'],
        nbinsx=50,
        name='Score Distribution'
    ))
    fig.add_vline(x=result['mean_score'], line_dash="dash", line_color="red", annotation_text="Mean")
    fig.add_vline(x=result['percentile_10'], line_dash="dot", line_color="orange", annotation_text="10th %ile")
    fig.add_vline(x=result['percentile_90'], line_dash="dot", line_color="orange", annotation_text="90th %ile")
    fig.update_layout(
        title=f"Lineup {selected_sim_lineup} Score Distribution",
        xaxis_title="Score",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Player statistics
    st.subheader("Player Simulation Stats")
    player_stats_df = pd.DataFrame(result['player_stats'])
    st.dataframe(player_stats_df, use_container_width=True)


def main():
    """Main application entry point"""
    initialize_session_state()
    render_header()
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("Follow these steps:")
        st.markdown("1. Upload player data")
        st.markdown("2. Select game mode")
        st.markdown("3. Configure optimization")
        st.markdown("4. Set exposure controls")
        st.markdown("5. Add custom rules")
        st.markdown("6. Generate lineups")
        st.markdown("7. Analyze results")
        st.markdown("8. Run simulations")
        
        st.markdown("---")
        st.markdown("**About**")
        st.markdown("DFS Optimizer Pro uses Mixed Integer Linear Programming (MILP) for deterministic, mathematically optimal lineup generation.")
    
    # Main content
    render_file_upload()
    
    if st.session_state.validated_data is not None:
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
