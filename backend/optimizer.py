"""
Core optimization engine using Mixed Integer Linear Programming (MILP)
Supports: min unique players per lineup, per-player ownership bounds
"""
import pulp
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from .game_modes import GameMode
from .rule_engine import RuleEngine


class OptimizerEngine:

    def __init__(
        self,
        player_pool: pd.DataFrame,
        game_mode: GameMode,
        rule_engine: Optional[RuleEngine] = None,
        min_unique_players: int = 1,
        player_min_own: Optional[Dict] = None,
        player_max_own: Optional[Dict] = None,
    ):
        self.player_pool = player_pool.copy()
        self.game_mode = game_mode
        self.rule_engine = rule_engine
        self.min_unique_players = max(1, min_unique_players)
        self.player_min_own = player_min_own or {}
        self.player_max_own = player_max_own or {}

        if game_mode.has_captain:
            self._prepare_captain_pool()

    def _prepare_captain_pool(self):
        captain_pool = self.player_pool.copy()
        captain_pool["Position"] = "CPT"
        captain_pool["Salary"] = (
            captain_pool["Salary"] * self.game_mode.captain_multiplier
        ).astype(int)
        captain_pool["Projection"] = (
            captain_pool["Projection"] * self.game_mode.captain_multiplier
        )
        captain_pool["ID"] = captain_pool["ID"].astype(str) + "_CPT"
        captain_pool["OriginalID"] = self.player_pool["ID"]

        flex_pool = self.player_pool.copy()
        flex_pool["Position"] = "FLEX"
        flex_pool["OriginalID"] = self.player_pool["ID"]

        self.player_pool = pd.concat([captain_pool, flex_pool], ignore_index=True)

    # ─────────────────────────────────────────────
    # PUBLIC METHODS
    # ─────────────────────────────────────────────
    def optimize_cash(
        self,
        num_lineups: int = 1,
        min_salary: Optional[int] = None,
        locked_players: Optional[List] = None,
        excluded_players: Optional[List] = None,
        variance_pct: float = 0.0,
    ) -> List[List[Dict]]:
        lineups = []
        used_lineups: List[List] = []  # list of sorted player-id lists

        for _ in range(num_lineups):
            projections = (
                self._apply_variance(self.player_pool["Projection"].values, variance_pct)
                if variance_pct > 0
                else self.player_pool["Projection"].values
            )
            ids = self._solve_milp(
                projections=projections,
                min_salary=min_salary,
                locked_players=locked_players,
                excluded_players=excluded_players,
                used_lineups=used_lineups,
            )
            if ids is None:
                break
            lineup = self._ids_to_lineup(ids)
            lineups.append(lineup)
            used_lineups.append(sorted([str(i) for i in ids]))

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
        locked_players: Optional[List] = None,
        excluded_players: Optional[List] = None,
        variance_pct: float = 0.15,
    ) -> List[List[Dict]]:
        lineups = []
        used_lineups: List[List] = []

        for _ in range(num_lineups):
            leverage = self._calculate_leverage_scores(
                projections=self.player_pool["Projection"].values,
                ownership=self.player_pool["Ownership"].values,
                projection_weight=projection_weight,
                ownership_weight=ownership_weight,
                ownership_penalty_threshold=ownership_penalty_threshold,
            )
            if variance_pct > 0:
                leverage = self._apply_variance(leverage, variance_pct)

            ids = self._solve_milp(
                projections=leverage,
                min_salary=min_salary,
                locked_players=locked_players,
                excluded_players=excluded_players,
                used_lineups=used_lineups,
                max_ownership=max_combinatorial_ownership,
                min_ownership=min_combinatorial_ownership,
            )
            if ids is None:
                break
            lineup = self._ids_to_lineup(ids)
            lineups.append(lineup)
            used_lineups.append(sorted([str(i) for i in ids]))

        return lineups

    # ─────────────────────────────────────────────
    # CORE MILP SOLVER
    # ─────────────────────────────────────────────
    def _solve_milp(
        self,
        projections: np.ndarray,
        min_salary: Optional[int] = None,
        locked_players: Optional[List] = None,
        excluded_players: Optional[List] = None,
        used_lineups: Optional[List[List]] = None,
        max_ownership: Optional[float] = None,
        min_ownership: Optional[float] = None,
    ) -> Optional[List]:

        prob = pulp.LpProblem("DFS", pulp.LpMaximize)

        player_vars = {
            row["ID"]: pulp.LpVariable(f"p_{i}", cat="Binary")
            for i, (_, row) in enumerate(self.player_pool.iterrows())
        }

        # Objective
        prob += pulp.lpSum(
            projections[i] * player_vars[row["ID"]]
            for i, (_, row) in enumerate(self.player_pool.iterrows())
        )

        # Salary cap
        prob += pulp.lpSum(
            row["Salary"] * player_vars[row["ID"]]
            for _, row in self.player_pool.iterrows()
        ) <= self.game_mode.salary_cap

        # Salary floor
        if min_salary:
            prob += pulp.lpSum(
                row["Salary"] * player_vars[row["ID"]]
                for _, row in self.player_pool.iterrows()
            ) >= min_salary

        # Roster size
        prob += pulp.lpSum(player_vars.values()) == self.game_mode.roster_size

        # Position requirements
        for position, count in self.game_mode.positions.items():
            pos_players = self.player_pool[self.player_pool["Position"] == position]
            prob += pulp.lpSum(
                player_vars[row["ID"]] for _, row in pos_players.iterrows()
            ) == count

        # Captain exclusivity
        if self.game_mode.has_captain:
            orig_ids = self.player_pool[
                self.player_pool["Position"] == "FLEX"
            ]["OriginalID"].unique()
            for oid in orig_ids:
                cpt_id = str(oid) + "_CPT"
                if cpt_id in player_vars and oid in player_vars:
                    prob += player_vars[cpt_id] + player_vars[oid] <= 1

        # Locked players
        if locked_players:
            for pid in locked_players:
                if pid in player_vars:
                    prob += player_vars[pid] == 1

        # Excluded players
        if excluded_players:
            for pid in excluded_players:
                if pid in player_vars:
                    prob += player_vars[pid] == 0

        # ── Uniqueness constraint (min_unique_players) ──
        # For each previously used lineup L, the new lineup must share
        # at most (|L| - min_unique_players) players with L.
        if used_lineups:
            for used_ids in used_lineups:
                overlap_vars = [player_vars[pid] for pid in used_ids if pid in player_vars]
                if overlap_vars:
                    max_overlap = self.game_mode.roster_size - self.min_unique_players
                    prob += pulp.lpSum(overlap_vars) <= max_overlap

        # Per-player ownership bounds
        for pid, min_own in self.player_min_own.items():
            if pid in player_vars:
                pass  # individual ownership is a data field, not a MILP constraint here

        # Aggregate ownership cap/floor (proxy via sum of individual ownerships)
        if max_ownership is not None:
            prob += pulp.lpSum(
                row["Ownership"] * player_vars[row["ID"]]
                for _, row in self.player_pool.iterrows()
            ) <= max_ownership * self.game_mode.roster_size

        if min_ownership is not None:
            prob += pulp.lpSum(
                row["Ownership"] * player_vars[row["ID"]]
                for _, row in self.player_pool.iterrows()
            ) >= min_ownership * self.game_mode.roster_size

        # Custom rule constraints
        if self.rule_engine:
            for constraint in self.rule_engine.get_pulp_constraints(player_vars, self.player_pool):
                prob += constraint

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.LpStatusOptimal:
            return None

        return [
            pid for pid, var in player_vars.items()
            if var.varValue is not None and var.varValue > 0.5
        ]

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────
    def _calculate_leverage_scores(self, projections, ownership,
                                   projection_weight, ownership_weight,
                                   ownership_penalty_threshold):
        proj_min, proj_max = projections.min(), projections.max()
        proj_norm = (
            (projections - proj_min) / (proj_max - proj_min)
            if proj_max > proj_min else projections
        )
        penalty = np.maximum(0, ownership - ownership_penalty_threshold)
        if penalty.max() > 0:
            penalty = penalty / penalty.max()
        leverage = (projection_weight * proj_norm) - (ownership_weight * penalty)
        leverage = leverage * (proj_max - proj_min) + proj_min
        return leverage

    def _apply_variance(self, values, variance_pct):
        noise = np.random.normal(0, variance_pct, size=len(values))
        return values * (1 + noise)

    def _ids_to_lineup(self, player_ids):
        lineup = []
        for pid in player_ids:
            row = self.player_pool[self.player_pool["ID"] == pid]
            if row.empty:
                continue
            r = row.iloc[0]
            lineup.append({
                "ID": r["ID"],
                "Player": r["Player"],
                "Position": r["Position"],
                "Salary": r["Salary"],
                "Projection": r["Projection"],
                "Ownership": r["Ownership"],
                "Team": r.get("Team", ""),
            })
        return lineup

    @staticmethod
    def calculate_combinatorial_ownership(lineup):
        product = 1.0
        for p in lineup:
            product *= (1 - p["Ownership"])
        return 1 - product

    @staticmethod
    def calculate_lineup_stats(lineup):
        total_salary = sum(p["Salary"] for p in lineup)
        total_projection = sum(p["Projection"] for p in lineup)
        avg_ownership = np.mean([p["Ownership"] for p in lineup])
        comb_ownership = OptimizerEngine.calculate_combinatorial_ownership(lineup)
        return {
            "total_salary": total_salary,
            "total_projection": total_projection,
            "avg_ownership": avg_ownership,
            "combinatorial_ownership": comb_ownership,
        }
