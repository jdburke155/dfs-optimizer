"""
Data validation and preprocessing module for DFS optimizer
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any


class DataValidator:
    """Validates and preprocesses DFS player data"""
    
    REQUIRED_COLUMNS = ['Player', 'Position', 'Salary', 'Projection', 'Ownership']
    OPTIONAL_COLUMNS = ['Team', 'Game', 'ID']
    
    def __init__(self):
        self.validation_errors = []
        self.warnings = []
    
    def validate_and_process(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and process uploaded data
        
        Returns:
            Processed DataFrame and statistics dictionary
        """
        self.validation_errors = []
        self.warnings = []
        
        # Check required columns
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
        
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Add optional columns if missing
        for col in self.OPTIONAL_COLUMNS:
            if col not in df.columns:
                if col == 'ID':
                    df['ID'] = range(1, len(df) + 1)
                else:
                    df[col] = ''
        
        # Normalize ownership to decimal format (handle percentage strings)
        df['Ownership'] = df['Ownership'].apply(self._normalize_ownership)
        
        # Validate salary
        df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
        if df['Salary'].isnull().any():
            null_indices = df[df['Salary'].isnull()].index.tolist()
            raise ValueError(f"Non-numeric or missing salary values at rows: {null_indices}")
        
        # Validate projection
        df['Projection'] = pd.to_numeric(df['Projection'], errors='coerce')
        if df['Projection'].isnull().any():
            null_indices = df[df['Projection'].isnull()].index.tolist()
            raise ValueError(f"Non-numeric or missing projection values at rows: {null_indices}")
        
        # Validate ownership
        if df['Ownership'].isnull().any():
            null_indices = df[df['Ownership'].isnull()].index.tolist()
            self.warnings.append(f"Missing ownership values at rows {null_indices}, defaulting to 0")
            df.loc[df['Ownership'].isnull(), 'Ownership'] = 0.0
        
        # Ensure ownership is between 0 and 1
        df['Ownership'] = df['Ownership'].clip(0, 1)
        
        # Check for duplicates
        duplicates = df[df.duplicated(subset=['Player'], keep=False)]
        if not duplicates.empty:
            self.warnings.append(f"Found {len(duplicates)} duplicate player entries")
        
        # Calculate statistics
        stats = self._calculate_stats(df)
        
        return df, stats
    
    def _normalize_ownership(self, value):
        """Convert ownership to decimal format"""
        if pd.isnull(value):
            return None
        
        # If string, try to parse
        if isinstance(value, str):
            value = value.strip().replace('%', '')
            try:
                value = float(value)
            except ValueError:
                return None
        
        # Convert to float
        value = float(value)
        
        # If greater than 1, assume it's a percentage
        if value > 1:
            value = value / 100
        
        return value
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data statistics"""
        stats = {
            'total_players': len(df),
            'salary_min': df['Salary'].min(),
            'salary_max': df['Salary'].max(),
            'salary_mean': df['Salary'].mean(),
            'projection_min': df['Projection'].min(),
            'projection_max': df['Projection'].max(),
            'projection_mean': df['Projection'].mean(),
            'ownership_min': df['Ownership'].min(),
            'ownership_max': df['Ownership'].max(),
            'ownership_mean': df['Ownership'].mean(),
            'position_breakdown': df['Position'].value_counts().to_dict(),
            'duplicates': df.duplicated(subset=['Player']).sum()
        }
        return stats


class PlayerPool:
    """Manages player pool with editing capabilities"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_projections = df['Projection'].copy()
        self.original_ownership = df['Ownership'].copy()
    
    def update_projection(self, player_id: int, new_projection: float):
        """Update a player's projection"""
        mask = self.df['ID'] == player_id
        if mask.any():
            self.df.loc[mask, 'Projection'] = new_projection
    
    def update_ownership(self, player_id: int, new_ownership: float):
        """Update a player's ownership"""
        mask = self.df['ID'] == player_id
        if mask.any():
            self.df.loc[mask, 'Ownership'] = max(0, min(1, new_ownership))
    
    def get_projection_delta(self, player_id: int) -> float:
        """Get the delta between current and original projection"""
        mask = self.df['ID'] == player_id
        if mask.any():
            idx = self.df[mask].index[0]
            return self.df.loc[idx, 'Projection'] - self.original_projections.iloc[idx]
        return 0
    
    def reset_projections(self):
        """Reset all projections to original values"""
        self.df['Projection'] = self.original_projections.copy()
    
    def reset_ownership(self):
        """Reset all ownership to original values"""
        self.df['Ownership'] = self.original_ownership.copy()
    
    def get_player_data(self) -> pd.DataFrame:
        """Get current player data"""
        return self.df.copy()
