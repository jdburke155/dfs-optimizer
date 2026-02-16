"""
Core optimization engine using Mixed Integer Linear Programming (MILP)
"""
import pulp
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from .game_modes import GameMode
from .rule_engine import RuleEngine


class OptimizerEngine:
    """
    MILP-based DFS lineup optimizer
    
    Supports multiple optimization modes:
    - Cash: Pure projection maximization
    - Tournament: Projection + ownership leverage
    """
    
    def __init__(
        self, 
        player_pool: pd.DataFrame, 
        game_mode: GameMode,
        rule_engine: Optional[RuleEngine] = None
    ):
        """
        Args:
            player_pool: DataFrame with player data
            game_mode: Game mode configuration
            rule_engine: Optional rule engine for custom constraints
        """
        self.player_pool = player_pool.copy()
        self.game_mode = game_mode
        self.rule_engine = rule_engine
        
        # Prepare player pool for captain mode
        if game_mode.has_captain:
            self._prepare_captain_pool()
    
    def _prepare_captain_pool(self):
        """Create separate captain and flex pools for captain mode"""
        # Create captain versions of all players
        captain_pool = self.player_pool.copy()
        captain_pool['Position'] = 'CPT'
        captain_pool['Salary'] = (captain_pool['Salary'] * self.game_mode.captain_multiplier).astype(int)
        captain_pool['Projection'] = captain_pool['Projection'] * self.game_mode.captain_multiplier
        captain_pool['ID'] = captain_pool['ID'].astype(str) + '_CPT'
        captain_pool['OriginalID'] = self.player_pool['ID']
        
        # Keep original players as flex
        flex_pool = self.player_pool.copy()
        flex_pool['Position'] = 'FLEX'
        flex_pool['OriginalID'] = self.player_pool['ID']
        
        # Combine pools
        self.player_pool = pd.concat([captain_pool, flex_pool], ignore_index=True)
    
    def optimize_cash(
        self, 
        num_lineups: int = 1,
        min_salary: Optional[int] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
        variance_pct: float = 0.0
    ) -> List[List[Dict]]:
        """
        Generate lineups optimized for cash games (pure projection)
        
        Args:
            num_lineups: Number of unique lineups to generate
            min_salary: Minimum salary floor
            locked_players: Player IDs that must be included
            excluded_players: Player IDs to exclude
            variance_pct: Random variance to apply (0 = deterministic)
        
        Returns:
            List of lineups (each lineup is list of player dicts)
        """
        lineups = []
        used_lineups = set()
        
        for i in range(num_lineups):
            # Apply randomization if variance > 0
            if variance_pct > 0:
                projections = self._apply_variance(self.player_pool['Projection'], variance_pct)
            else:
                projections = self.player_pool['Projection'].values
            
            # Solve optimization
            lineup_ids = self._solve_milp(
                projections=projections,
                min_salary=min_salary,
                locked_players=locked_players,
                excluded_players=excluded_players,
                exclude_lineups=used_lineups
            )
            
            if lineup_ids is None:
                break  # No more feasible lineups
            
            # Convert to lineup dict and add to results
            lineup = self._ids_to_lineup(lineup_ids)
            lineups.append(lineup)
            
            # Track used lineup
            lineup_signature = self._get_lineup_signature(lineup_ids)
            used_lineups.add(lineup_signature)
        
        return lineups
    
    def optimize_tournament(
        self,
        num_lineups: int = 20,
        projection_weight: float = 0.7,
        ownership_weight: float = 0.3,
        ownership_penalty_threshold: float = 0.15,
        min_salary: Optional[int] = None,
        max_combinatorial_ownership: Optional[float] = None,
        min_combinatorial_ownership: Optional[float] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
        variance_pct: float = 0.15
    ) -> List[List[Dict]]:
        """
        Generate lineups optimized for tournaments (projection + ownership leverage)
        
        Args:
            num_lineups: Number of unique lineups
            projection_weight: Weight for projection (0-1)
            ownership_weight: Weight for ownership leverage (0-1)
            ownership_penalty_threshold: Ownership % above which to penalize
            min_salary: Minimum salary floor
            max_combinatorial_ownership: Maximum combined ownership
            min_combinatorial_ownership: Minimum combined ownership
            locked_players: Player IDs that must be included
            excluded_players: Player IDs to exclude
            variance_pct: Random variance to apply
        
        Returns:
            List of lineups
        """
        lineups = []
        used_lineups = set()
        
        for i in range(num_lineups):
            # Calculate leverage scores
            leverage_scores = self._calculate_leverage_scores(
                projections=self.player_pool['Projection'].values,
                ownership=self.player_pool['Ownership'].values,
                projection_weight=projection_weight,
                ownership_weight=ownership_weight,
                ownership_penalty_threshold=ownership_penalty_threshold
            )
            
            # Apply randomization
            if variance_pct > 0:
                leverage_scores = self._apply_variance(leverage_scores, variance_pct)
            
            # Solve optimization
            lineup_ids = self._solve_milp(
                projections=leverage_scores,
                min_salary=min_salary,
                locked_players=locked_players,
                excluded_players=excluded_players,
                exclude_lineups=used_lineups,
                max_ownership=max_combinatorial_ownership,
                min_ownership=min_combinatorial_ownership
            )
            
            if lineup_ids is None:
                break
            
            lineup = self._ids_to_lineup(lineup_ids)
            lineups.append(lineup)
            
            lineup_signature = self._get_lineup_signature(lineup_ids)
            used_lineups.add(lineup_signature)
        
        return lineups
    
    def _solve_milp(
        self,
        projections: np.ndarray,
        min_salary: Optional[int] = None,
        locked_players: Optional[List[int]] = None,
        excluded_players: Optional[List[int]] = None,
        exclude_lineups: Optional[set] = None,
        max_ownership: Optional[float] = None,
        min_ownership: Optional[float] = None
    ) -> Optional[List]:
        """
        Solve MILP optimization problem
        
        Returns:
            List of selected player IDs, or None if infeasible
        """
        # Create problem
        prob = pulp.LpProblem("DFS_Optimizer", pulp.LpMaximize)
        
        # Create binary variables for each player
        player_vars = {}
        for idx, row in self.player_pool.iterrows():
            player_vars[row['ID']] = pulp.LpVariable(f"player_{row['ID']}", cat='Binary')
        
        # Objective: maximize sum of projections (or leverage scores)
        prob += pulp.lpSum([
            projections[idx] * player_vars[row['ID']] 
            for idx, row in self.player_pool.iterrows()
        ])
        
        # Constraint: Salary cap
        prob += pulp.lpSum([
            row['Salary'] * player_vars[row['ID']] 
            for idx, row in self.player_pool.iterrows()
        ]) <= self.game_mode.salary_cap
        
        # Constraint: Salary floor
        if min_salary is not None:
            prob += pulp.lpSum([
                row['Salary'] * player_vars[row['ID']] 
                for idx, row in self.player_pool.iterrows()
            ]) >= min_salary
        
        # Constraint: Roster size
        prob += pulp.lpSum([
            player_vars[row['ID']] 
            for idx, row in self.player_pool.iterrows()
        ]) == self.game_mode.roster_size
        
        # Constraint: Position requirements
        for position, count in self.game_mode.positions.items():
            position_players = self.player_pool[self.player_pool['Position'] == position]
            prob += pulp.lpSum([
                player_vars[row['ID']] 
                for idx, row in position_players.iterrows()
            ]) == count
        
        # Constraint: Captain mode - same player can't be captain and flex
        if self.game_mode.has_captain:
            original_ids = self.player_pool[self.player_pool['Position'] == 'FLEX']['OriginalID'].unique()
            for orig_id in original_ids:
                cpt_id = str(orig_id) + '_CPT'
                flex_id = orig_id
                if cpt_id in player_vars and flex_id in player_vars:
                    prob += player_vars[cpt_id] + player_vars[flex_id] <= 1
        
        # Constraint: Locked players
        if locked_players:
            for player_id in locked_players:
                if player_id in player_vars:
                    prob += player_vars[player_id] == 1
        
        # Constraint: Excluded players
        if excluded_players:
            for player_id in excluded_players:
                if player_id in player_vars:
                    prob += player_vars[player_id] == 0
        
        # Constraint: Exclude previously used lineups
        if exclude_lineups:
            for used_signature in exclude_lineups:
                used_ids = used_signature.split('|')
                # At least one player must be different
                prob += pulp.lpSum([
                    player_vars[pid] for pid in used_ids if pid in player_vars
                ]) <= len(used_ids) - 1
        
        # Constraint: Ownership limits
        if max_ownership is not None:
            # This is approximate - true combinatorial ownership is nonlinear
            # We use sum of individual ownership as proxy
            prob += pulp.lpSum([
                row['Ownership'] * player_vars[row['ID']] 
                for idx, row in self.player_pool.iterrows()
            ]) <= max_ownership * self.game_mode.roster_size
        
        if min_ownership is not None:
            prob += pulp.lpSum([
                row['Ownership'] * player_vars[row['ID']] 
                for idx, row in self.player_pool.iterrows()
            ]) >= min_ownership * self.game_mode.roster_size
        
        # Add custom rule constraints
        if self.rule_engine:
            custom_constraints = self.rule_engine.get_pulp_constraints(player_vars, self.player_pool)
            for constraint in custom_constraints:
                prob += constraint
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Check if solution is feasible
        if prob.status != pulp.LpStatusOptimal:
            return None
        
        # Extract selected players
        selected_ids = [
            player_id for player_id, var in player_vars.items() 
            if var.varValue == 1
        ]
        
        return selected_ids
    
    def _calculate_leverage_scores(
        self,
        projections: np.ndarray,
        ownership: np.ndarray,
        projection_weight: float,
        ownership_weight: float,
        ownership_penalty_threshold: float
    ) -> np.ndarray:
        """
        Calculate leverage scores combining projection and ownership
        
        Leverage = (projection_weight * projection) - (ownership_weight * ownership_penalty)
        """
        # Normalize projections to 0-1 scale
        proj_min, proj_max = projections.min(), projections.max()
        proj_norm = (projections - proj_min) / (proj_max - proj_min) if proj_max > proj_min else projections
        
        # Calculate ownership penalty (only above threshold)
        ownership_penalty = np.maximum(0, ownership - ownership_penalty_threshold)
        
        # Normalize ownership penalty
        if ownership_penalty.max() > 0:
            ownership_penalty = ownership_penalty / ownership_penalty.max()
        
        # Calculate leverage scores
        leverage = (projection_weight * proj_norm) - (ownership_weight * ownership_penalty)
        
        # Scale back to projection range for consistency
        leverage = leverage * (proj_max - proj_min) + proj_min
        
        return leverage
    
    def _apply_variance(self, values: np.ndarray, variance_pct: float) -> np.ndarray:
        """Apply Gaussian randomization to values"""
        noise = np.random.normal(0, variance_pct, size=len(values))
        return values * (1 + noise)
    
    def _ids_to_lineup(self, player_ids: List) -> List[Dict]:
        """Convert player IDs to lineup dictionary"""
        lineup = []
        for player_id in player_ids:
            player_data = self.player_pool[self.player_pool['ID'] == player_id].iloc[0]
            lineup.append({
                'ID': player_data['ID'],
                'Player': player_data['Player'],
                'Position': player_data['Position'],
                'Salary': player_data['Salary'],
                'Projection': player_data['Projection'],
                'Ownership': player_data['Ownership'],
                'Team': player_data.get('Team', '')
            })
        return lineup
    
    def _get_lineup_signature(self, player_ids: List) -> str:
        """Create unique signature for lineup to detect duplicates"""
        # For captain mode, use original IDs to detect duplicates
        if self.game_mode.has_captain:
            # Extract original player IDs
            original_ids = []
            for pid in player_ids:
                if '_CPT' in str(pid):
                    original_ids.append(str(pid).replace('_CPT', ''))
                else:
                    original_ids.append(str(pid))
            return '|'.join(sorted(set(original_ids)))
        else:
            return '|'.join(sorted([str(pid) for pid in player_ids]))
    
    @staticmethod
    def calculate_combinatorial_ownership(lineup: List[Dict]) -> float:
        """
        Calculate combinatorial ownership: 1 - product(1 - individual_ownership)
        
        This represents the probability that at least one person in the field
        has this exact lineup
        """
        product = 1.0
        for player in lineup:
            product *= (1 - player['Ownership'])
        
        return 1 - product
    
    @staticmethod
    def calculate_lineup_stats(lineup: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for a lineup"""
        total_salary = sum(p['Salary'] for p in lineup)
        total_projection = sum(p['Projection'] for p in lineup)
        avg_ownership = np.mean([p['Ownership'] for p in lineup])
        comb_ownership = OptimizerEngine.calculate_combinatorial_ownership(lineup)
        
        return {
            'total_salary': total_salary,
            'total_projection': total_projection,
            'avg_ownership': avg_ownership,
            'combinatorial_ownership': comb_ownership
        }
