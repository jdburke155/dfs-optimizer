"""
Exposure management and analysis utilities
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class ExposureManager:
    """
    Manages player exposure across multiple lineups
    
    Handles:
    - Exposure calculation
    - Exposure enforcement
    - Exposure boosting/docking
    """
    
    def __init__(self):
        self.exposure_targets = {}  # player_id -> target exposure (0-1)
        self.locked_players = set()  # player_ids with 100% exposure
        self.projection_boosts = {}  # player_id -> boost multiplier
    
    def set_lock(self, player_id: int, locked: bool = True):
        """Lock a player to 100% exposure"""
        if locked:
            self.locked_players.add(player_id)
            self.exposure_targets[player_id] = 1.0
        else:
            self.locked_players.discard(player_id)
            if player_id in self.exposure_targets:
                del self.exposure_targets[player_id]
    
    def set_max_exposure(self, player_id: int, max_exposure: float):
        """Set maximum exposure for a player (0-1)"""
        self.exposure_targets[player_id] = min(1.0, max(0.0, max_exposure))
    
    def set_projection_boost(self, player_id: int, boost_pct: float):
        """
        Boost or dock a player's projection
        
        Args:
            boost_pct: Percentage boost (e.g., 10 for +10%, -10 for -10%)
        """
        multiplier = 1 + (boost_pct / 100)
        self.projection_boosts[player_id] = multiplier
    
    def apply_projection_adjustments(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """Apply projection boosts/docks to player pool"""
        adjusted = player_pool.copy()
        
        for player_id, multiplier in self.projection_boosts.items():
            mask = adjusted['ID'] == player_id
            if mask.any():
                adjusted.loc[mask, 'Projection'] *= multiplier
        
        return adjusted
    
    def calculate_exposure(self, lineups: List[List[Dict]]) -> pd.DataFrame:
        """
        Calculate actual exposure for each player across lineups
        
        Returns:
            DataFrame with player exposure statistics
        """
        if not lineups:
            return pd.DataFrame()
        
        # Count appearances
        player_counts = {}
        total_lineups = len(lineups)
        
        for lineup in lineups:
            for player in lineup:
                player_id = player['ID']
                player_name = player['Player']
                
                if player_id not in player_counts:
                    player_counts[player_id] = {
                        'Player': player_name,
                        'Count': 0,
                        'Salary': player['Salary'],
                        'Projection': player['Projection'],
                        'Ownership': player['Ownership']
                    }
                player_counts[player_id]['Count'] += 1
        
        # Calculate exposure percentages
        exposure_data = []
        for player_id, data in player_counts.items():
            exposure_pct = data['Count'] / total_lineups
            target = self.exposure_targets.get(player_id, None)
            
            exposure_data.append({
                'Player': data['Player'],
                'Exposure': exposure_pct,
                'Count': data['Count'],
                'Target': target,
                'Variance': (exposure_pct - target) if target is not None else None,
                'Salary': data['Salary'],
                'Projection': data['Projection'],
                'Ownership': data['Ownership']
            })
        
        df = pd.DataFrame(exposure_data)
        df = df.sort_values('Exposure', ascending=False)
        
        return df
    
    def enforce_exposure(
        self, 
        lineups: List[List[Dict]], 
        player_pool: pd.DataFrame,
        tolerance: float = 0.05
    ) -> List[List[Dict]]:
        """
        Enforce exposure targets through lineup swapping
        
        This is a post-processing step that adjusts lineups to match
        exposure targets as closely as possible
        
        Args:
            lineups: Generated lineups
            player_pool: Full player pool
            tolerance: Acceptable exposure variance (0.05 = 5%)
        
        Returns:
            Adjusted lineups
        """
        if not self.exposure_targets:
            return lineups
        
        # Calculate current exposure
        exposure_df = self.calculate_exposure(lineups)
        
        # Identify players that need adjustment
        needs_more = []
        needs_less = []
        
        for player_id, target in self.exposure_targets.items():
            current = exposure_df[exposure_df['Player'] == self._get_player_name(player_id, player_pool)]
            
            if current.empty:
                if target > 0:
                    needs_more.append((player_id, target, 0))
            else:
                current_exp = current.iloc[0]['Exposure']
                if current_exp < target - tolerance:
                    needs_more.append((player_id, target, current_exp))
                elif current_exp > target + tolerance:
                    needs_less.append((player_id, target, current_exp))
        
        # Perform swaps (simplified implementation)
        # In production, this would be more sophisticated
        adjusted_lineups = lineups.copy()
        
        # TODO: Implement sophisticated swapping algorithm
        # For now, return original lineups
        
        return adjusted_lineups
    
    def _get_player_name(self, player_id: int, player_pool: pd.DataFrame) -> str:
        """Get player name from ID"""
        player = player_pool[player_pool['ID'] == player_id]
        if not player.empty:
            return player.iloc[0]['Player']
        return ""
    
    def get_locked_players(self) -> List[int]:
        """Get list of locked player IDs"""
        return list(self.locked_players)
    
    def clear_all(self):
        """Clear all exposure settings"""
        self.exposure_targets = {}
        self.locked_players = set()
        self.projection_boosts = {}
    
    @staticmethod
    def analyze_exposure_distribution(exposure_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exposure distribution statistics"""
        if exposure_df.empty:
            return {}
        
        return {
            'total_players_used': len(exposure_df),
            'avg_exposure': exposure_df['Exposure'].mean(),
            'max_exposure': exposure_df['Exposure'].max(),
            'min_exposure': exposure_df['Exposure'].min(),
            'std_exposure': exposure_df['Exposure'].std(),
            'concentrated_players': len(exposure_df[exposure_df['Exposure'] > 0.5]),
            'rare_players': len(exposure_df[exposure_df['Exposure'] < 0.1])
        }
