"""
Monte Carlo simulation engine for lineup analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any


class MonteCarloSimulator:
    """
    Monte Carlo simulation for DFS lineup analysis
    
    Simulates player performance using Gaussian distributions
    to estimate lineup upside, downside, and win probability
    """
    
    def __init__(self, variance_pct: float = 0.20, correlation: float = 0.0):
        """
        Args:
            variance_pct: Standard deviation as % of projection (default 20%)
            correlation: Team correlation coefficient (0 to 1)
        """
        self.variance_pct = variance_pct
        self.correlation = correlation
    
    def simulate_lineups(
        self, 
        lineups: List[List[Dict]], 
        num_simulations: int = 10000,
        field_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Run Monte Carlo simulations for multiple lineups
        
        Args:
            lineups: List of lineups (each lineup is list of player dicts)
            num_simulations: Number of simulation iterations
            field_size: Size of tournament field for win probability
        
        Returns:
            List of simulation results for each lineup
        """
        results = []
        
        for lineup_idx, lineup in enumerate(lineups):
            result = self._simulate_single_lineup(lineup, num_simulations, field_size)
            result['lineup_idx'] = lineup_idx
            results.append(result)
        
        return results
    
    def _simulate_single_lineup(
        self, 
        lineup: List[Dict], 
        num_simulations: int,
        field_size: int
    ) -> Dict[str, Any]:
        """Simulate a single lineup"""
        
        # Extract player projections and create distributions
        projections = np.array([p['Projection'] for p in lineup])
        std_devs = projections * self.variance_pct
        
        # Generate simulated scores
        # Shape: (num_simulations, num_players)
        simulated_scores = np.random.normal(
            loc=projections,
            scale=std_devs,
            size=(num_simulations, len(lineup))
        )
        
        # Apply correlation if specified
        if self.correlation > 0:
            simulated_scores = self._apply_team_correlation(simulated_scores, lineup)
        
        # Calculate lineup totals
        lineup_scores = simulated_scores.sum(axis=1)
        
        # Calculate statistics
        mean_score = lineup_scores.mean()
        std_score = lineup_scores.std()
        percentile_10 = np.percentile(lineup_scores, 10)
        percentile_90 = np.percentile(lineup_scores, 90)
        percentile_99 = np.percentile(lineup_scores, 99)
        
        # Estimate win probability (simplified)
        # Assume field follows similar distribution
        field_mean = mean_score * 0.95  # Assume we're slightly above average
        field_std = std_score * 1.2  # Field has more variance
        
        # For each simulation, estimate probability of beating random field opponent
        win_prob = self._estimate_win_probability(lineup_scores, field_mean, field_std, field_size)
        
        # Calculate top 1% probability
        top1_threshold = np.percentile(lineup_scores, 99)
        top1_prob = (lineup_scores >= top1_threshold).mean()
        
        # Per-player statistics
        player_stats = []
        for i, player in enumerate(lineup):
            player_scores = simulated_scores[:, i]
            player_stats.append({
                'Player': player['Player'],
                'Mean': player_scores.mean(),
                'Std': player_scores.std(),
                'P10': np.percentile(player_scores, 10),
                'P90': np.percentile(player_scores, 90)
            })
        
        return {
            'mean_score': mean_score,
            'std_score': std_score,
            'percentile_10': percentile_10,
            'percentile_90': percentile_90,
            'percentile_99': percentile_99,
            'win_probability': win_prob,
            'top1_probability': top1_prob,
            'player_stats': player_stats,
            'simulated_scores': lineup_scores  # Keep for analysis
        }
    
    def _apply_team_correlation(self, simulated_scores: np.ndarray, lineup: List[Dict]) -> np.ndarray:
        """
        Apply team correlation to simulated scores
        
        Players on the same team are positively correlated
        """
        # Group players by team
        teams = [p.get('Team', '') for p in lineup]
        team_groups = {}
        for i, team in enumerate(teams):
            if team and team != '':
                if team not in team_groups:
                    team_groups[team] = []
                team_groups[team].append(i)
        
        # Apply correlation within teams
        for team, player_indices in team_groups.items():
            if len(player_indices) > 1:
                # Generate correlated noise
                num_sims = simulated_scores.shape[0]
                team_factor = np.random.normal(0, 1, num_sims)
                
                for idx in player_indices:
                    # Mix independent and correlated noise
                    independent = simulated_scores[:, idx]
                    mean = lineup[idx]['Projection']
                    std = mean * self.variance_pct
                    
                    correlated_component = team_factor * std * self.correlation
                    independent_component = (independent - mean) * np.sqrt(1 - self.correlation**2)
                    
                    simulated_scores[:, idx] = mean + independent_component + correlated_component
        
        return simulated_scores
    
    def _estimate_win_probability(
        self, 
        lineup_scores: np.ndarray, 
        field_mean: float, 
        field_std: float,
        field_size: int
    ) -> float:
        """
        Estimate probability of winning tournament
        
        Simplified model: probability of beating (field_size - 1) opponents
        """
        # For each simulated score, calculate probability of beating field
        win_probs = []
        
        for score in lineup_scores:
            # Probability this score beats a random opponent from field
            prob_beat_one = self._normal_cdf(score, field_mean, field_std)
            
            # Probability of beating all (field_size - 1) opponents
            # Simplified: assumes independence (not fully accurate but reasonable)
            prob_beat_all = prob_beat_one ** (field_size - 1)
            win_probs.append(prob_beat_all)
        
        return np.mean(win_probs)
    
    @staticmethod
    def _normal_cdf(x: float, mean: float, std: float) -> float:
        """Calculate cumulative distribution function for normal distribution"""
        from math import erf, sqrt
        return 0.5 * (1 + erf((x - mean) / (std * sqrt(2))))
    
    def analyze_player_distribution(self, player_data: Dict, num_samples: int = 10000) -> Dict[str, Any]:
        """
        Analyze distribution for a single player
        
        Args:
            player_data: Dict with 'Projection' key
            num_samples: Number of samples to generate
        
        Returns:
            Statistical summary
        """
        projection = player_data['Projection']
        std_dev = projection * self.variance_pct
        
        samples = np.random.normal(projection, std_dev, num_samples)
        
        return {
            'mean': samples.mean(),
            'std': samples.std(),
            'min': samples.min(),
            'max': samples.max(),
            'p10': np.percentile(samples, 10),
            'p25': np.percentile(samples, 25),
            'p50': np.percentile(samples, 50),
            'p75': np.percentile(samples, 75),
            'p90': np.percentile(samples, 90),
            'samples': samples
        }
