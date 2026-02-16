#!/usr/bin/env python3
"""
DFS Optimizer - Setup Validation Script

Run this script to validate your installation and test core functionality.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    required = {
        'streamlit': '1.31.0',
        'pandas': '2.0.0',
        'numpy': '1.24.0',
        'pulp': '2.8.0',
        'plotly': '5.18.0'
    }
    
    missing = []
    installed = []
    
    for package, min_version in required.items():
        try:
            module = __import__(package)
            installed.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - NOT INSTALLED")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"\nTo install:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def check_file_structure():
    """Verify all required files exist"""
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'backend/__init__.py',
        'backend/data_validator.py',
        'backend/game_modes.py',
        'backend/optimizer.py',
        'backend/rule_engine.py',
        'backend/monte_carlo.py',
        'backend/exposure_manager.py',
        'data/sample_golf.csv'
    ]
    
    missing = []
    for filepath in required_files:
        if not Path(filepath).exists():
            missing.append(filepath)
            print(f"❌ {filepath}")
        else:
            print(f"✅ {filepath}")
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    return True


def test_backend_imports():
    """Test that backend modules can be imported"""
    modules = [
        'backend.data_validator',
        'backend.game_modes',
        'backend.optimizer',
        'backend.rule_engine',
        'backend.monte_carlo',
        'backend.exposure_manager'
    ]
    
    print("\nTesting backend imports...")
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module}: {str(e)}")
            return False
    
    return True


def test_data_validator():
    """Test data validation functionality"""
    print("\nTesting data validator...")
    
    try:
        import pandas as pd
        from backend.data_validator import DataValidator
        
        # Create test data
        test_data = {
            'Player': ['Player A', 'Player B', 'Player C'],
            'Position': ['G', 'G', 'G'],
            'Salary': [10000, 8000, 6000],
            'Projection': [80, 70, 60],
            'Ownership': [0.25, 0.15, 0.10]
        }
        df = pd.DataFrame(test_data)
        
        # Validate
        validator = DataValidator()
        validated_df, stats = validator.validate_and_process(df)
        
        assert len(validated_df) == 3
        assert stats['total_players'] == 3
        assert stats['salary_min'] == 6000
        assert stats['salary_max'] == 10000
        
        print("✅ Data validation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Data validation test failed: {str(e)}")
        return False


def test_game_modes():
    """Test game mode configurations"""
    print("\nTesting game modes...")
    
    try:
        from backend.game_modes import GameModes
        
        # Test each mode
        modes = GameModes.get_all_modes()
        assert 'Golf Classic' in modes
        assert 'Golf Showdown' in modes
        assert 'Golf Captain Showdown' in modes
        
        classic = modes['Golf Classic']
        assert classic.roster_size == 6
        assert classic.salary_cap == 50000
        assert not classic.has_captain
        
        captain = modes['Golf Captain Showdown']
        assert captain.has_captain
        assert captain.captain_multiplier == 1.5
        
        print("✅ Game modes configured correctly")
        return True
        
    except Exception as e:
        print(f"❌ Game modes test failed: {str(e)}")
        return False


def test_optimizer():
    """Test basic optimization"""
    print("\nTesting optimizer...")
    
    try:
        import pandas as pd
        from backend.optimizer import OptimizerEngine
        from backend.game_modes import GameModes
        
        # Create test player pool
        test_data = {
            'ID': [1, 2, 3, 4, 5, 6, 7],
            'Player': ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7'],
            'Position': ['G', 'G', 'G', 'G', 'G', 'G', 'G'],
            'Salary': [10000, 9000, 8000, 7000, 6000, 5000, 4000],
            'Projection': [85, 80, 75, 70, 65, 60, 55],
            'Ownership': [0.30, 0.25, 0.20, 0.15, 0.10, 0.08, 0.05],
            'Team': ['', '', '', '', '', '', '']
        }
        df = pd.DataFrame(test_data)
        
        # Create optimizer
        game_mode = GameModes.GOLF_CLASSIC
        optimizer = OptimizerEngine(df, game_mode)
        
        # Generate lineup
        lineups = optimizer.optimize_cash(num_lineups=1, variance_pct=0.0)
        
        assert len(lineups) == 1
        assert len(lineups[0]) == 6
        
        # Check salary constraint
        total_salary = sum(p['Salary'] for p in lineups[0])
        assert total_salary <= 50000
        
        print("✅ Optimizer working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Optimizer test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_monte_carlo():
    """Test Monte Carlo simulation"""
    print("\nTesting Monte Carlo simulation...")
    
    try:
        from backend.monte_carlo import MonteCarloSimulator
        
        # Create test lineup
        test_lineup = [
            {'Player': 'P1', 'Projection': 80, 'Team': ''},
            {'Player': 'P2', 'Projection': 75, 'Team': ''},
            {'Player': 'P3', 'Projection': 70, 'Team': ''},
            {'Player': 'P4', 'Projection': 65, 'Team': ''},
            {'Player': 'P5', 'Projection': 60, 'Team': ''},
            {'Player': 'P6', 'Projection': 55, 'Team': ''}
        ]
        
        # Run simulation
        simulator = MonteCarloSimulator(variance_pct=0.20)
        results = simulator.simulate_lineups([test_lineup], num_simulations=1000)
        
        assert len(results) == 1
        result = results[0]
        
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'win_probability' in result
        
        # Mean should be close to sum of projections
        expected_mean = sum(p['Projection'] for p in test_lineup)
        assert abs(result['mean_score'] - expected_mean) < 10
        
        print("✅ Monte Carlo simulation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Monte Carlo test failed: {str(e)}")
        return False


def main():
    """Run all validation tests"""
    print("=" * 60)
    print("DFS Optimizer - Setup Validation")
    print("=" * 60)
    
    tests = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Backend Imports", test_backend_imports),
        ("Data Validator", test_data_validator),
        ("Game Modes", test_game_modes),
        ("Optimizer", test_optimizer),
        ("Monte Carlo", test_monte_carlo)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print(f"{'='*60}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("\nYou can now run the application:")
        print("  streamlit run app.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the application.")
    print(f"{'='*60}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
