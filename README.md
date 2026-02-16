# DFS Optimizer Pro

Production-ready Daily Fantasy Sports optimizer for DraftKings Golf using Mixed Integer Linear Programming (MILP).

## Features

### ✅ Complete Implementation

- **Data Import & Validation**: CSV upload with automatic validation and normalization
- **Game Modes**: Golf Classic, Golf Showdown, Golf Captain Showdown
- **Optimization Modes**: Cash (pure projection) and Tournament (projection + ownership leverage)
- **Exposure Management**: Lock/boost/dock players, set exposure limits
- **Rule Engine**: Custom constraints (at least one of, max from team, salary thresholds)
- **Monte Carlo Simulation**: 1,000-20,000 simulations with win probability analysis
- **Lineup Export**: Download lineups as CSV for DraftKings upload

### 🧮 Mathematical Correctness

- MILP-based optimization using PuLP
- Deterministic optimal solutions
- No constraint violations
- Proper combinatorial ownership calculation: `1 - ∏(1 - ownership_i)`
- Gaussian variance with configurable std dev
- Vectorized Monte Carlo simulations

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download this repository**

```bash
cd dfs_optimizer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## Usage Workflow

### 1. Data Import

Prepare a CSV file with the following columns:

**Required:**
- `Player` - Player name
- `Position` - Position (e.g., 'G' for golfer)
- `Salary` - DraftKings salary
- `Projection` - Projected fantasy points
- `Ownership` - Projected ownership (0.15 or 15% format accepted)

**Optional:**
- `Team` - Team name (for team-based rules)
- `Game` - Game information
- `ID` - Unique identifier (auto-generated if missing)

**Example CSV:**
```csv
Player,Position,Salary,Projection,Ownership,Team
Scottie Scheffler,G,11500,85.5,0.35,
Rory McIlroy,G,10800,82.3,0.28,
Jon Rahm,G,10200,79.8,0.22,
Viktor Hovland,G,9500,76.2,0.15,
Xander Schauffele,G,9000,74.5,0.18,
Cameron Smith,G,8500,71.8,0.12,
```

### 2. Select Game Mode

Choose from:
- **Golf Classic**: 6 golfers, $50K cap
- **Golf Showdown**: 6 golfers, $50K cap (higher variance mode)
- **Golf Captain Showdown**: 1 Captain (1.5x salary/points) + 5 Flex, $50K cap

### 3. Configure Optimization

**Cash Game Mode:**
- Pure projection maximization
- Deterministic (0% variance) or slight randomization
- Generate 1-3 optimal lineups

**Tournament Mode:**
- Projection weight: 70% (default)
- Ownership leverage weight: 30% (default)
- Ownership penalty threshold: 15% (default)
- Variance: 15% (default)
- Generate 20-150 lineups

### 4. Set Exposure Controls

**Lock Players:**
- Forces players into 100% of lineups
- Use for core plays

**Boost/Dock:**
- Adjust projections by -50% to +100%
- Influences optimization without hard constraints

**Max Exposure:**
- Limit player appearances (e.g., max 50% exposure)

### 5. Add Custom Rules

**At least one of:**
- Requires at least one player from a list
- Example: "At least one of: Player A, Player B, Player C"

**Max from team:**
- Limits players from same team
- Example: "Max 2 from Team X"

**Min salary threshold:**
- Enforces minimum total salary
- Example: "Minimum $48,000"

**Max salary on expensive players:**
- Caps spending on high-salary players
- Example: "Max $20,000 on players over $10,000"

### 6. Generate Lineups

Click **"Generate Lineups"** to run MILP optimization.

The system will:
- Validate all constraints
- Generate unique lineups
- Calculate combinatorial ownership
- Display results sorted by projection, salary, or ownership

### 7. Analyze Results

**Lineup Summary:**
- Total projection
- Total salary
- Average ownership
- Combinatorial ownership

**Exposure Analysis:**
- See which players appear most frequently
- Identify concentrated vs. diversified builds
- Visual exposure distribution chart

### 8. Run Monte Carlo Simulation

**Configure:**
- Number of simulations: 1,000-20,000
- Variance %: Standard deviation as % of projection
- Field size: Tournament entry count

**Results:**
- Mean score
- Standard deviation
- 10th/90th/99th percentile scores
- Win probability
- Top 1% probability
- Per-player distribution stats

## Architecture

### Backend (`/backend/`)

**data_validator.py**
- CSV validation and preprocessing
- Ownership normalization
- Data statistics calculation
- Player pool management with editing

**game_modes.py**
- Game mode configurations
- Roster validation
- Captain mode support

**optimizer.py**
- MILP optimization engine using PuLP
- Cash and tournament optimization modes
- Leverage score calculation
- Duplicate lineup prevention
- Combinatorial ownership calculation

**rule_engine.py**
- Custom constraint management
- Rule validation
- PuLP constraint generation
- Post-optimization rule checking

**monte_carlo.py**
- Vectorized Monte Carlo simulations
- Gaussian performance distributions
- Team correlation modeling
- Win probability estimation

**exposure_manager.py**
- Lock/boost/dock functionality
- Exposure tracking and analysis
- Projection adjustments

### Frontend (`app.py`)

Streamlit-based UI with:
- File upload widget
- Interactive sliders and selects
- Plotly visualizations
- CSV export functionality
- Session state management

## Technology Stack

- **Optimization**: PuLP (MILP solver with CBC backend)
- **Simulation**: NumPy (vectorized operations)
- **Data Processing**: Pandas
- **UI**: Streamlit
- **Visualization**: Plotly

## Key Algorithms

### MILP Formulation

**Decision Variables:**
- `x_i ∈ {0,1}` for each player i

**Objective:**
- Maximize: `Σ(projection_i × x_i)` (Cash mode)
- Maximize: `Σ(leverage_i × x_i)` (Tournament mode)

**Constraints:**
- `Σ(salary_i × x_i) ≤ salary_cap`
- `Σ(x_i) = roster_size`
- Position requirements
- Captain exclusivity (captain mode)
- Custom rules

### Leverage Score Calculation

```
leverage = (w_proj × normalized_projection) - (w_own × ownership_penalty)

where ownership_penalty = max(0, ownership - threshold)
```

### Combinatorial Ownership

```
P(lineup duplicated) = 1 - ∏(1 - ownership_i)
```

### Monte Carlo Simulation

For each player:
```
simulated_score ~ N(projection, projection × variance_pct)
```

Lineup score = Σ(simulated_player_scores)

Win probability estimated via comparison to field distribution.

## File Structure

```
dfs_optimizer/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── backend/
│   ├── __init__.py
│   ├── data_validator.py      # Data validation & preprocessing
│   ├── game_modes.py           # Game mode configurations
│   ├── optimizer.py            # MILP optimization engine
│   ├── rule_engine.py          # Custom constraint management
│   ├── monte_carlo.py          # Monte Carlo simulation
│   └── exposure_manager.py     # Exposure controls
└── data/
    └── sample_golf.csv         # Example data file
```

## Troubleshooting

**No feasible lineups generated:**
- Check if rules conflict with each other
- Verify salary cap isn't too restrictive
- Ensure locked players fit under cap
- Review position requirements vs. available players

**Optimization is slow:**
- Reduce number of lineups
- Simplify custom rules
- Use cash mode for faster optimization

**Import errors:**
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Check Python version is 3.8+
- Verify file structure is correct

## Performance Notes

- Optimization: ~0.1-1 second per lineup (depending on complexity)
- Monte Carlo: ~1-5 seconds for 10,000 simulations
- Scales well to 150 lineups and 20,000 simulations

## Extension Guide

### Adding New Sports

1. Define new GameMode in `game_modes.py`
2. Add position requirements
3. Update roster validation logic
4. Modify optimizer for sport-specific constraints

### Adding New Rules

1. Add rule type to `rule_engine.py`
2. Implement validation logic
3. Create PuLP constraints
4. Add UI elements in `app.py`

## Credits

Developed as a production-ready DFS optimization system using industry-standard MILP techniques and Monte Carlo simulation methods.

## License

This is a demonstration project. Use at your own risk for DFS contests.

## Support

For issues or questions, review the code documentation in each module.
