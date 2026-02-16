"""
Game mode configurations for different DFS contests
"""
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class GameMode:
    """Configuration for a specific game mode"""
    name: str
    roster_size: int
    salary_cap: int
    positions: Dict[str, int]  # Position: count
    has_captain: bool = False
    captain_multiplier: float = 1.0
    description: str = ""
    
    def validate_roster(self, lineup: List[Dict]) -> bool:
        """Validate that a lineup meets roster requirements"""
        if len(lineup) != self.roster_size:
            return False
        
        position_counts = {}
        for player in lineup:
            pos = player.get('Position', '')
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        for pos, required_count in self.positions.items():
            if position_counts.get(pos, 0) != required_count:
                return False
        
        return True


class GameModes:
    """Registry of all supported game modes"""
    
    GOLF_CLASSIC = GameMode(
        name="Golf Classic",
        roster_size=6,
        salary_cap=50000,
        positions={"G": 6},
        has_captain=False,
        description="Classic golf lineup: 6 golfers, $50k cap"
    )
    
    GOLF_SHOWDOWN = GameMode(
        name="Golf Showdown",
        roster_size=6,
        salary_cap=50000,
        positions={"G": 6},
        has_captain=False,
        description="Showdown golf: 6 golfers, $50k cap, higher variance"
    )
    
    GOLF_CAPTAIN_SHOWDOWN = GameMode(
        name="Golf Captain Showdown",
        roster_size=6,
        salary_cap=50000,
        positions={"CPT": 1, "FLEX": 5},
        has_captain=True,
        captain_multiplier=1.5,
        description="Captain mode: 1 Captain (1.5x), 5 Flex, $50k cap"
    )
    
    @classmethod
    def get_all_modes(cls) -> Dict[str, GameMode]:
        """Get all available game modes"""
        return {
            "Golf Classic": cls.GOLF_CLASSIC,
            "Golf Showdown": cls.GOLF_SHOWDOWN,
            "Golf Captain Showdown": cls.GOLF_CAPTAIN_SHOWDOWN
        }
    
    @classmethod
    def get_mode(cls, mode_name: str) -> Optional[GameMode]:
        """Get a specific game mode by name"""
        return cls.get_all_modes().get(mode_name)
