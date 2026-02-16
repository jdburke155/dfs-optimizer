# Quick Start Guide

## Installation & First Run

### Step 1: Install Dependencies
```bash
cd dfs_optimizer
pip install -r requirements.txt
```

### Step 2: Launch Application
```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501`

### Step 3: Load Sample Data

1. Click "Browse files" under "Upload Player CSV"
2. Select `data/sample_golf.csv`
3. You should see: "✅ Successfully loaded 20 players"

## Example Workflow: Cash Game

### Configuration
1. **Game Mode**: Golf Classic
2. **Optimization Mode**: Cash Game
3. **Number of Lineups**: 3
4. **Variance**: 0%
5. **Minimum Salary**: 48000

### Generate
- Click "🚀 Generate Lineups"
- View the top 3 optimal lineups
- Export to CSV if desired

## Example Workflow: Tournament

### Configuration
1. **Game Mode**: Golf Classic
2. **Optimization Mode**: Tournament
3. **Number of Lineups**: 20
4. **Variance**: 15%
5. **Projection Weight**: 70%
6. **Ownership Weight**: 30%
7. **Ownership Penalty Threshold**: 15%

### Exposure Controls
1. Go to "Lock/Unlock" tab
2. Lock "Scottie Scheffler" (your core play)
3. Go to "Boost/Dock" tab
4. Boost "Tom Kim" by +20% (contrarian play)

### Generate & Analyze
- Click "🚀 Generate Lineups"
- Review exposure distribution
- Run Monte Carlo simulation with 10,000 iterations

## Example Workflow: Captain Mode

### Configuration
1. **Game Mode**: Golf Captain Showdown
2. **Optimization Mode**: Tournament
3. **Number of Lineups**: 30

### Custom Rules
1. Add rule: "At least one of: Rory McIlroy, Jon Rahm, Viktor Hovland"
2. Add rule: "Max 2 from USA"

### Generate
- System creates 30 unique lineups
- Each has 1 Captain (1.5x multiplier)
- No player appears as both Captain and Flex

## Monte Carlo Simulation Example

After generating lineups:

1. Go to "9️⃣ Monte Carlo Simulation"
2. Set simulations to 10,000
3. Set variance to 20%
4. Set field size to 100
5. Click "Run Simulation"

**Results show:**
- Mean score (expected points)
- 10th/90th/99th percentile (upside/downside)
- Win probability (chance to take 1st)
- Top 1% probability (chance to cash big)

## Tips for Success

### Cash Games
- Use 0% variance for pure optimal
- Lock your core plays
- Set high salary floor ($48K+)
- Generate 1-3 lineups

### GPP Tournaments
- Use 15-20% variance
- Generate 20-150 lineups
- Balance projection and ownership
- Use custom rules for correlation plays

### Captain Mode
- Captain your highest-upside play
- Balance salary between Captain and Flex
- Use rules to ensure correlation

## Common Adjustments

### Player Projection Override
1. Expand "✏️ Edit Player Data"
2. Select player
3. Enter new projection
4. Click "Update Projection"

### Boost High-Upside Player
1. Go to "Boost/Dock" tab
2. Select player
3. Set boost to +25%
4. Click "Apply Boost/Dock"

### Force Roster Correlation
1. Go to "Custom Rules"
2. Add "At least one of" rule
3. Select correlated players
4. Click "Add Rule"

## Interpreting Results

### Lineup Metrics

**Projection**: Total expected fantasy points
- Higher = more points expected

**Salary**: Total salary spent
- Closer to $50K = more efficient

**Avg Ownership**: Mean of individual ownership %
- Lower in tournaments = contrarian

**Combinatorial Ownership**: Probability lineup is duplicated
- Formula: `1 - ∏(1 - ownership_i)`
- Lower = more unique

### Simulation Metrics

**Mean Score**: Expected lineup score
- Use for comparing lineups

**Std Dev**: Score volatility
- Higher = more variance/upside

**10th/90th Percentile**: Range of likely outcomes
- Wide range = boom/bust potential

**Win Probability**: Chance to finish 1st
- Based on field size and distribution

**Top 1%**: Chance to finish top 1% of field
- Key metric for GPP min-cash threshold

## Export & Upload to DraftKings

1. After generating lineups, scroll to "Export Lineups"
2. Click "Download Lineups (CSV)"
3. Open CSV and verify lineups
4. Format for DraftKings upload if needed
5. Upload to DraftKings contest

## Troubleshooting

**"No feasible lineups found"**
- Reduce number of locked players
- Lower salary floor
- Remove conflicting rules
- Check if enough players at each position

**Lineups are too similar**
- Increase variance %
- Reduce locked players
- Add diversity rules

**Simulation is slow**
- Reduce simulation count to 5,000
- Reduce number of lineups to analyze
- Close other applications

**Ownership seems wrong**
- Check CSV format (0.15 or 15% both work)
- Verify ownership column isn't missing data
- Edit individual ownership values if needed

## Next Steps

1. **Import your own data**: Use your projections and ownership
2. **Experiment with settings**: Find what works for your strategy
3. **Run simulations**: Understand lineup upside/downside
4. **Track results**: Compare generated lineups to actual outcomes
5. **Refine**: Adjust projection weights and variance based on results

## Support

For detailed technical documentation, see `README.md`

For code-level details, see inline documentation in backend modules
