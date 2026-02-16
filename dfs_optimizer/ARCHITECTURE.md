# Architecture Documentation

## System Overview

The DFS Optimizer is a production-grade Daily Fantasy Sports lineup optimizer built on mathematical optimization principles. It uses Mixed Integer Linear Programming (MILP) to generate deterministically optimal lineups while respecting complex constraints.

## Core Design Principles

1. **Mathematical Correctness**: All optimizations are mathematically sound with no constraint violations
2. **Deterministic Results**: Same inputs produce same outputs (unless variance is explicitly enabled)
3. **Modular Architecture**: Easily extensible to new sports and contest types
4. **Separation of Concerns**: Clear boundaries between data, optimization, UI, and simulation layers

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│                       (Streamlit UI)                         │
├─────────────────────────────────────────────────────────────┤
│                     Application Layer                        │
│              (Session Management, Workflows)                 │
├─────────────────────────────────────────────────────────────┤
│                       Business Logic                         │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │   Optimizer  │  │ Rule Engine  │  │ Exposure Mgmt   │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ Monte Carlo  │  │  Game Modes  │  │ Data Validator  │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                       Data Layer                             │
│                  (Pandas DataFrames)                         │
├─────────────────────────────────────────────────────────────┤
│                   Optimization Engine                        │
│                   (PuLP + CBC Solver)                        │
└─────────────────────────────────────────────────────────────┘
```

## Module Details

### 1. Data Validator (`data_validator.py`)

**Responsibilities:**
- CSV parsing and validation
- Data type conversion
- Ownership normalization
- Statistical analysis
- Player pool management

**Key Classes:**
- `DataValidator`: Validates and preprocesses uploaded data
- `PlayerPool`: Manages player data with edit capabilities

**Data Flow:**
```
CSV Upload → Validation → Normalization → Statistics → PlayerPool
```

**Validation Rules:**
- Required columns present
- Numeric types for Salary, Projection, Ownership
- Ownership normalized to 0-1 range
- No missing critical values
- Duplicate detection

### 2. Game Modes (`game_modes.py`)

**Responsibilities:**
- Contest type definitions
- Roster requirement specifications
- Captain mode configuration

**Key Classes:**
- `GameMode`: Dataclass defining contest parameters
- `GameModes`: Registry of all supported modes

**Supported Modes:**
```python
Golf Classic: 6G, $50K, no captain
Golf Showdown: 6G, $50K, high variance
Golf Captain Showdown: 1 CPT + 5 FLEX, $50K, 1.5x multiplier
```

**Extensibility:**
Add new modes by defining new `GameMode` instances with:
- Roster size
- Salary cap
- Position requirements
- Captain multiplier (if applicable)

### 3. Optimizer Engine (`optimizer.py`)

**Responsibilities:**
- MILP formulation and solving
- Lineup generation
- Duplicate prevention
- Leverage calculation
- Combinatorial ownership computation

**Key Classes:**
- `OptimizerEngine`: Main optimization engine

**Optimization Modes:**

**Cash Mode:**
```
Maximize: Σ(projection_i × x_i)
Subject to:
  - Σ(salary_i × x_i) ≤ salary_cap
  - Σ(x_i) = roster_size
  - Position constraints
  - Custom rules
```

**Tournament Mode:**
```
Leverage = w_proj × proj_norm - w_own × own_penalty
Maximize: Σ(leverage_i × x_i)
Subject to: [same constraints as cash]
```

**Key Algorithms:**

**Duplicate Prevention:**
```python
For each used lineup L:
  Σ(x_i for i in L) ≤ |L| - 1
```

**Captain Exclusivity:**
```python
For each player p:
  x_captain_p + x_flex_p ≤ 1
```

**Combinatorial Ownership:**
```python
P(duplicate) = 1 - ∏(1 - ownership_i)
```

### 4. Rule Engine (`rule_engine.py`)

**Responsibilities:**
- Custom constraint management
- Rule validation
- PuLP constraint generation
- Post-optimization verification

**Key Classes:**
- `Rule`: Represents a single constraint
- `RuleEngine`: Manages all rules

**Supported Rules:**

**At Least One Of:**
```python
Σ(x_i for i in player_set) ≥ 1
```

**Max From Team:**
```python
Σ(x_i for i in team) ≤ max_count
```

**Salary Constraints:**
```python
Σ(salary_i × x_i for i where salary_i > threshold) ≤ max_spend
```

**Rule Validation Flow:**
```
Define Rule → Validate Feasibility → Generate PuLP Constraint → Add to Model
```

### 5. Monte Carlo Simulator (`monte_carlo.py`)

**Responsibilities:**
- Performance distribution simulation
- Win probability estimation
- Percentile calculation
- Team correlation modeling

**Key Classes:**
- `MonteCarloSimulator`: Vectorized simulation engine

**Simulation Algorithm:**
```python
For each simulation iteration:
  For each player:
    score ~ N(projection, projection × variance)
  
  lineup_score = Σ(player_scores)
  
  P(win) = P(beat all opponents)
```

**Team Correlation:**
```python
correlated_score = independent_score × √(1-ρ²) + team_factor × ρ
```

**Performance Metrics:**
- Mean score
- Standard deviation
- 10th/90th/99th percentiles
- Win probability
- Top 1% probability

### 6. Exposure Manager (`exposure_manager.py`)

**Responsibilities:**
- Player locking (100% exposure)
- Exposure caps
- Projection adjustments (boost/dock)
- Exposure tracking and analysis

**Key Classes:**
- `ExposureManager`: Manages exposure constraints

**Features:**
- Lock players to appear in all lineups
- Set maximum exposure per player
- Boost/dock projections by percentage
- Track actual vs. target exposure
- Exposure distribution analysis

### 7. Streamlit Application (`app.py`)

**Responsibilities:**
- User interface
- Session state management
- Workflow orchestration
- Visualization

**Key Functions:**
- `render_file_upload()`: CSV upload and validation
- `render_game_mode_selector()`: Contest type selection
- `render_optimization_settings()`: Parameter configuration
- `render_exposure_controls()`: Exposure management UI
- `render_rule_engine()`: Custom rule creation
- `render_optimization_button()`: Optimization execution
- `render_lineup_results()`: Results display
- `render_exposure_analysis()`: Exposure visualization
- `render_monte_carlo()`: Simulation interface

**State Management:**
```python
st.session_state.player_pool        # PlayerPool instance
st.session_state.generated_lineups  # List of lineups
st.session_state.exposure_manager   # ExposureManager instance
st.session_state.rule_engine        # RuleEngine instance
st.session_state.simulation_results # Simulation results
```

## Data Structures

### Player Record
```python
{
    'ID': int,
    'Player': str,
    'Position': str,
    'Salary': int,
    'Projection': float,
    'Ownership': float (0-1),
    'Team': str (optional),
    'Game': str (optional)
}
```

### Lineup
```python
List[Player Record]  # Length = roster_size
```

### Simulation Result
```python
{
    'lineup_idx': int,
    'mean_score': float,
    'std_score': float,
    'percentile_10': float,
    'percentile_90': float,
    'percentile_99': float,
    'win_probability': float,
    'top1_probability': float,
    'player_stats': List[PlayerStat]
}
```

## Optimization Flow

```
1. Load Data
   ↓
2. Validate & Normalize
   ↓
3. Select Game Mode
   ↓
4. Configure Parameters
   ↓
5. Set Exposure Controls
   ↓
6. Define Custom Rules
   ↓
7. Apply Projection Adjustments
   ↓
8. Formulate MILP Problem
   ↓
9. Solve (CBC Solver)
   ↓
10. Validate Solution
    ↓
11. Generate Additional Lineups (if needed)
    ↓
12. Calculate Statistics
    ↓
13. Display Results
    ↓
14. (Optional) Run Monte Carlo Simulation
```

## Performance Characteristics

### Time Complexity

**Single Lineup Optimization:**
- Best case: O(n log n) where n = number of players
- Average case: O(n²) 
- Worst case: O(2ⁿ) (rare, with many constraints)

**Multiple Lineups:**
- O(k × n²) where k = number of lineups

**Monte Carlo Simulation:**
- O(m × p) where m = simulations, p = players per lineup
- Fully vectorized using NumPy

### Space Complexity

**Player Pool:** O(n)
**MILP Model:** O(n) variables + O(c) constraints
**Simulation:** O(m × p) for score storage

### Scalability

**Tested Limits:**
- Players: 1-500 (typical: 20-50)
- Lineups: 1-150
- Simulations: 1,000-20,000
- Rules: 0-20

**Bottlenecks:**
- MILP solving time increases with constraint complexity
- Duplicate prevention becomes expensive at 100+ lineups
- Simulation memory usage grows linearly with iterations

## Error Handling

### Validation Errors
- Missing required columns
- Invalid data types
- Out-of-range values

### Optimization Errors
- Infeasible problem (no valid solution)
- Conflicting constraints
- Solver timeout (rare)

### Runtime Errors
- Division by zero (handled with guards)
- Empty player pool
- Invalid rule parameters

**Error Strategy:**
- Validate early (at input time)
- Provide clear error messages
- Suggest corrective actions
- Never silently fail

## Security Considerations

1. **Input Validation**: All CSV inputs validated before processing
2. **Type Safety**: Strong typing throughout backend
3. **Resource Limits**: Bounded simulation counts and lineup numbers
4. **No Code Execution**: Pure data processing, no eval/exec
5. **State Isolation**: Session state prevents cross-user pollution

## Testing Strategy

### Unit Tests
- Data validation logic
- Optimization correctness
- Rule validation
- Mathematical calculations

### Integration Tests
- End-to-end optimization flow
- Multi-lineup generation
- Simulation accuracy
- Exposure tracking

### Validation Tests
- Constraint satisfaction
- No duplicate lineups
- Salary cap compliance
- Position requirements

## Extension Points

### Adding New Sports

1. Define new `GameMode` with sport-specific rules
2. Update position validation logic
3. Add sport-specific constraints if needed
4. Update UI labels and descriptions

### Adding New Optimization Modes

1. Create new objective function
2. Define mode-specific parameters
3. Implement in `OptimizerEngine`
4. Add UI controls

### Adding New Rules

1. Define rule type in `RuleEngine`
2. Implement validation logic
3. Generate PuLP constraints
4. Add UI interface

### Adding New Metrics

1. Calculate in appropriate module
2. Store in lineup stats
3. Display in results table
4. Add to export format

## Dependencies

**Core:**
- Python 3.8+
- PuLP 2.8+ (MILP solver)
- NumPy 1.24+ (numerical operations)
- Pandas 2.0+ (data manipulation)

**UI:**
- Streamlit 1.31+ (web interface)
- Plotly 5.18+ (visualizations)

**Solver:**
- CBC (PuLP default, fast and reliable)
- Alternative: GLPK, Gurobi (if available)

## Future Enhancements

**Potential Features:**
- Multi-sport support (NFL, NBA, MLB)
- Stacking strategies (QB + WR correlation)
- Late swap optimization
- Historical backtesting
- API integration (DraftKings, FanDuel)
- Machine learning for projections
- Parallel optimization
- Cloud deployment
- User authentication
- Database persistence

## Conclusion

This architecture provides a solid foundation for DFS optimization with clear separation of concerns, extensibility, and mathematical rigor. The modular design allows for easy testing, maintenance, and feature additions while maintaining production-quality code standards.
