"""
Rule engine for user-defined lineup constraints
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class Rule:
    """Represents a single constraint rule"""
    rule_type: str
    description: str
    params: Dict[str, Any]
    
    def validate(self, player_pool: pd.DataFrame) -> tuple[bool, Optional[str]]:
        """
        Validate that the rule is feasible given the player pool
        
        Returns:
            (is_valid, error_message)
        """
        if self.rule_type == "at_least_one_of":
            player_names = self.params.get("players", [])
            available = player_pool[player_pool['Player'].isin(player_names)]
            if len(available) == 0:
                return False, f"No players available from: {', '.join(player_names)}"
        
        elif self.rule_type == "pick_x_of_group":
            player_names = self.params.get("players", [])
            min_count = self.params.get("min_count", 1)
            max_count = self.params.get("max_count", 6)
            available = player_pool[player_pool['Player'].isin(player_names)]
            if len(available) < min_count:
                return False, f"Not enough players from group (need at least {min_count}, have {len(available)})"
        
        elif self.rule_type == "max_from_team":
            team = self.params.get("team", "")
            max_count = self.params.get("max_count", 0)
            team_players = player_pool[player_pool['Team'] == team]
            if len(team_players) < max_count:
                return False, f"Not enough players from team {team} (need {max_count}, have {len(team_players)})"
        
        elif self.rule_type == "min_salary_threshold":
            min_salary = self.params.get("min_salary", 0)
            # This is always feasible
            pass
        
        elif self.rule_type == "max_salary_on_expensive":
            max_salary = self.params.get("max_salary", 0)
            threshold = self.params.get("threshold", 10000)
            expensive_players = player_pool[player_pool['Salary'] >= threshold]
            if len(expensive_players) > 0 and expensive_players['Salary'].min() > max_salary:
                return False, f"No expensive players available under ${max_salary}"
        
        return True, None


class RuleEngine:
    """Manages and validates user-defined rules"""
    
    def __init__(self):
        self.rules: List[Rule] = []
    
    def add_rule(self, rule: Rule) -> tuple[bool, Optional[str]]:
        """
        Add a rule to the engine
        
        Returns:
            (success, error_message)
        """
        self.rules.append(rule)
        return True, None
    
    def remove_rule(self, index: int):
        """Remove a rule by index"""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
    
    def clear_rules(self):
        """Remove all rules"""
        self.rules = []
    
    def validate_all(self, player_pool: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Validate all rules against the player pool
        
        Returns:
            (all_valid, list_of_errors)
        """
        errors = []
        for i, rule in enumerate(self.rules):
            is_valid, error_msg = rule.validate(player_pool)
            if not is_valid:
                errors.append(f"Rule {i+1}: {error_msg}")
        
        return len(errors) == 0, errors
    
    def get_pulp_constraints(self, player_vars: Dict, player_pool: pd.DataFrame):
        """
        Generate PuLP constraints from rules
        
        Args:
            player_vars: Dict mapping player ID to PuLP binary variable
            player_pool: DataFrame with player data
        
        Returns:
            List of PuLP constraint objects
        """
        constraints = []
        
        for rule in self.rules:
            if rule.rule_type == "at_least_one_of":
                player_names = rule.params.get("players", [])
                matching_ids = player_pool[player_pool['Player'].isin(player_names)]['ID'].tolist()
                if matching_ids:
                    # At least one of these players must be selected
                    constraint = sum(player_vars[pid] for pid in matching_ids if pid in player_vars) >= 1
                    constraints.append(constraint)
            
            elif rule.rule_type == "pick_x_of_group":
                player_names = rule.params.get("players", [])
                min_count = rule.params.get("min_count", 1)
                max_count = rule.params.get("max_count", 6)
                matching_ids = player_pool[player_pool['Player'].isin(player_names)]['ID'].tolist()
                if matching_ids:
                    player_sum = sum(player_vars[pid] for pid in matching_ids if pid in player_vars)
                    constraints.append(player_sum >= min_count)
                    constraints.append(player_sum <= max_count)
            
            elif rule.rule_type == "max_from_team":
                team = rule.params.get("team", "")
                max_count = rule.params.get("max_count", 0)
                team_ids = player_pool[player_pool['Team'] == team]['ID'].tolist()
                if team_ids:
                    constraint = sum(player_vars[pid] for pid in team_ids if pid in player_vars) <= max_count
                    constraints.append(constraint)
            
            elif rule.rule_type == "max_salary_on_expensive":
                threshold = rule.params.get("threshold", 10000)
                max_salary = rule.params.get("max_salary", 0)
                expensive = player_pool[player_pool['Salary'] >= threshold]
                if not expensive.empty:
                    # Sum of salaries of expensive players <= max_salary
                    expensive_ids = expensive['ID'].tolist()
                    expensive_salaries = expensive.set_index('ID')['Salary'].to_dict()
                    constraint = sum(
                        player_vars[pid] * expensive_salaries[pid] 
                        for pid in expensive_ids if pid in player_vars
                    ) <= max_salary
                    constraints.append(constraint)
        
        return constraints
    
    def check_lineup_against_rules(self, lineup: List[Dict], player_pool: pd.DataFrame) -> tuple[bool, List[str]]:
        """
        Check if a generated lineup satisfies all rules
        
        Returns:
            (satisfies_all, list_of_violations)
        """
        violations = []
        
        for i, rule in enumerate(self.rules):
            if rule.rule_type == "at_least_one_of":
                player_names = rule.params.get("players", [])
                lineup_players = [p['Player'] for p in lineup]
                if not any(name in lineup_players for name in player_names):
                    violations.append(f"Rule {i+1}: No player from {player_names}")
            
            elif rule.rule_type == "max_from_team":
                team = rule.params.get("team", "")
                max_count = rule.params.get("max_count", 0)
                team_count = sum(1 for p in lineup if p.get('Team') == team)
                if team_count > max_count:
                    violations.append(f"Rule {i+1}: Too many players from {team}")
            
            elif rule.rule_type == "min_salary_threshold":
                min_salary = rule.params.get("min_salary", 0)
                total_salary = sum(p['Salary'] for p in lineup)
                if total_salary < min_salary:
                    violations.append(f"Rule {i+1}: Salary ${total_salary} below ${min_salary}")
        
        return len(violations) == 0, violations
    
    def get_rules_summary(self) -> List[str]:
        """Get human-readable summary of all rules"""
        summaries = []
        for i, rule in enumerate(self.rules):
            summaries.append(f"{i+1}. {rule.description}")
        return summaries
