# ⚡ DFS Optimizer Pro

**Production-ready Daily Fantasy Sports optimizer for DraftKings Golf**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 🎯 Features

- **MILP Optimization** - Mathematically optimal lineups using Mixed Integer Linear Programming
- **Monte Carlo Simulation** - 1,000-20,000 iterations with win probability analysis
- **Multiple Game Modes** - Classic, Showdown, and Captain Showdown
- **Smart Ownership** - Leverage calculations with combinatorial ownership
- **Custom Rules** - User-defined constraints (stacking, team limits, salary rules)
- **Exposure Management** - Lock, boost, dock players with exposure tracking

## 🚀 Quick Start

### Try It Online (No Installation!)

👉 **[Launch the App](https://share.streamlit.io)** _(add your deployed URL here after deployment)_

### Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/dfs-optimizer.git
cd dfs-optimizer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 How It Works

1. **Upload CSV** - Player data with salary, projection, and ownership
2. **Select Game Mode** - Classic, Showdown, or Captain mode
3. **Configure Settings** - Cash or tournament optimization
4. **Set Constraints** - Lock players, add rules, manage exposure
5. **Generate Lineups** - Get 1-150 unique optimal lineups
6. **Analyze Results** - Exposure charts, Monte Carlo simulations
7. **Export** - Download lineups as CSV

## 📁 Sample Data

Test the optimizer with the included `data/sample_golf.csv` file containing 20 PGA players.

## 🧮 Mathematical Foundation

- **Optimization**: PuLP MILP solver with CBC backend
- **Combinatorial Ownership**: `1 - ∏(1 - ownership_i)`
- **Leverage Score**: `w_proj × proj_norm - w_own × own_penalty`
- **Monte Carlo**: Gaussian distributions with vectorized NumPy operations

## 📖 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Step-by-step usage examples
- **[Architecture Guide](ARCHITECTURE.md)** - Technical deep dive
- **[Full Documentation](README.md)** - Complete reference

## 🛠️ Tech Stack

- **Backend**: Python, PuLP, NumPy, Pandas
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Optimization**: MILP with CBC solver

## 📝 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## 🎓 Game Modes

### Golf Classic
- 6 golfers
- $50,000 salary cap
- Pure roster construction

### Golf Showdown
- 6 golfers  
- $50,000 salary cap
- Higher variance optimization

### Golf Captain Showdown
- 1 Captain (1.5x salary and points)
- 5 Flex players
- $50,000 salary cap
- Captain cannot also be Flex

## 🔧 Features Detail

### Cash Game Optimization
- Pure projection maximization
- Deterministic results
- Optimal lineup generation

### Tournament Optimization
- Projection + ownership leverage
- Configurable weights
- Variance for diversification
- Low ownership targeting

### Custom Rules Engine
- "At least one of" constraints
- Team-based limits
- Salary thresholds
- Position requirements

### Monte Carlo Simulation
- Win probability estimation
- Percentile calculations (10th, 90th, 99th)
- Top 1% probability
- Score distribution visualization

## 📊 Example Output

```
Lineup #1
Player              Position  Salary  Projection  Ownership
Scottie Scheffler   G        $11,500    88.5       35%
Rory McIlroy        G        $10,800    84.2       28%
Viktor Hovland      G         $9,500    78.4       18%
...

Total Salary: $49,800
Total Projection: 455.3
Avg Ownership: 18.5%
Combinatorial Ownership: 0.0023 (0.23%)
```

## 🤝 Contributing

This is a demonstration project. Feel free to fork and extend!

## 📄 License

Open source for educational and personal use.

## 🙏 Acknowledgments

Built with industry-standard MILP optimization techniques and Monte Carlo simulation methods.

---

**Ready to optimize your DFS lineups?** [Launch the app](#) and start winning! 🏆
