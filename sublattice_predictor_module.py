"""
Sublattice Predictor Module

Predicts metal sublattice configurations given N (number of unique positions).
Based on analysis of 1802 structures from AFLOW Prototype Encyclopedia.

Usage:
    from sublattice_predictor_module import SublatticePredictor
    
    predictor = SublatticePredictor('sublattice_lookup.json')
    predictions = predictor.predict(N=4)
"""

import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LatticeTypePrediction:
    system: str
    centering: str
    count: int
    probability: float
    cumulative: float


@dataclass 
class ZPatternPrediction:
    system: str
    centering: str
    z_values: List[str]
    count: int
    probability: float
    cumulative: float


@dataclass
class OffsetPrediction:
    system: str
    centering: str
    offsets: List[str]
    count: int
    probability: float
    cumulative: float


class SublatticePredictor:
    """
    Predicts metal sublattice configurations for ionic crystals.
    
    Provides three levels of prediction:
    1. lattice_type: Crystal system + centering (highest coverage)
    2. z_pattern: + z-coordinate pattern (moderate coverage)
    3. full_offsets: Complete offset specification (lowest coverage for large N)
    """
    
    def __init__(self, lookup_path: str = 'sublattice_lookup.json'):
        """Load the lookup table."""
        with open(lookup_path, 'r') as f:
            self._data = json.load(f)
    
    def available_n_values(self) -> List[int]:
        """Return list of N values in the database."""
        return sorted([int(k) for k in self._data.keys()])
    
    def has_data(self, N: int) -> bool:
        """Check if we have data for this N value."""
        return str(N) in self._data
    
    def get_stats(self, N: int) -> Dict:
        """Get statistics for a given N value."""
        if not self.has_data(N):
            return None
        
        entry = self._data[str(N)]
        return {
            'total_structures': entry['total_structures'],
            'unique_patterns': entry['unique_patterns'],
            'predictability': entry['predictability'],
            'recommendation': entry['recommendation']
        }
    
    def is_predictable(self, N: int) -> bool:
        """Check if exact offsets are predictable for this N."""
        if not self.has_data(N):
            return False
        return self._data[str(N)]['predictability'] == 'high'
    
    def predict(self, N: int) -> Dict:
        """
        Get all predictions for a given N value.
        
        Returns dict with keys:
        - stats: total_structures, unique_patterns, predictability, recommendation
        - lattice_type: list of LatticeTypePrediction
        - z_pattern: list of ZPatternPrediction
        - full_offsets: list of OffsetPrediction
        """
        if not self.has_data(N):
            return None
        
        entry = self._data[str(N)]
        
        return {
            'stats': {
                'total_structures': entry['total_structures'],
                'unique_patterns': entry['unique_patterns'],
                'predictability': entry['predictability'],
                'recommendation': entry['recommendation']
            },
            'lattice_type': [
                LatticeTypePrediction(**item) for item in entry['lattice_type']
            ],
            'z_pattern': [
                ZPatternPrediction(**item) for item in entry['z_pattern']
            ],
            'full_offsets': [
                OffsetPrediction(**item) for item in entry['full_offsets']
            ]
        }
    
    def get_top_lattice_types(self, N: int, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Get top-k most likely lattice types for given N.
        
        Returns list of (system, centering, probability) tuples.
        """
        if not self.has_data(N):
            return []
        
        entry = self._data[str(N)]
        results = []
        for item in entry['lattice_type'][:top_k]:
            results.append((item['system'], item['centering'], item['probability']))
        return results
    
    def get_top_z_patterns(self, N: int, top_k: int = 5) -> List[Tuple[str, str, List[str], float]]:
        """
        Get top-k most likely z-patterns for given N.
        
        Returns list of (system, centering, z_values, probability) tuples.
        """
        if not self.has_data(N):
            return []
        
        entry = self._data[str(N)]
        results = []
        for item in entry['z_pattern'][:top_k]:
            results.append((
                item['system'], 
                item['centering'], 
                item['z_values'],
                item['probability']
            ))
        return results
    
    def get_top_offsets(self, N: int, top_k: int = 10) -> List[Tuple[str, str, List[str], float]]:
        """
        Get top-k most likely full offset patterns for given N.
        
        Returns list of (system, centering, offsets, probability) tuples.
        
        WARNING: For N >= 4, coverage is very low. Most structures have unique patterns.
        """
        if not self.has_data(N):
            return []
        
        entry = self._data[str(N)]
        results = []
        for item in entry['full_offsets'][:top_k]:
            results.append((
                item['system'],
                item['centering'],
                item['offsets'],
                item['probability']
            ))
        return results
    
    def get_z_patterns_for_lattice(self, N: int, system: str, centering: str) -> List[Tuple[List[str], float]]:
        """
        Get z-patterns filtered by a specific lattice type.
        
        Returns list of (z_values, probability) tuples.
        """
        if not self.has_data(N):
            return []
        
        entry = self._data[str(N)]
        results = []
        for item in entry['z_pattern']:
            if item['system'] == system and item['centering'] == centering:
                results.append((item['z_values'], item['probability']))
        return results
    
    def get_offsets_for_lattice(self, N: int, system: str, centering: str) -> List[Tuple[List[str], float]]:
        """
        Get full offsets filtered by a specific lattice type.
        
        Returns list of (offsets, probability) tuples.
        """
        if not self.has_data(N):
            return []
        
        entry = self._data[str(N)]
        results = []
        for item in entry['full_offsets']:
            if item['system'] == system and item['centering'] == centering:
                results.append((item['offsets'], item['probability']))
        return results
    
    def suggest_strategy(self, N: int) -> str:
        """
        Get recommended prediction strategy for given N.
        
        Returns a string describing the best approach.
        """
        if not self.has_data(N):
            return f"No data available for N={N}"
        
        entry = self._data[str(N)]
        pred = entry['predictability']
        
        if pred == 'high':
            return (
                f"N={N} is HIGHLY PREDICTABLE. "
                f"Use get_top_offsets() directly. "
                f"Top 10 patterns cover most structures."
            )
        elif pred == 'moderate':
            return (
                f"N={N} is MODERATELY PREDICTABLE. "
                f"Use get_top_lattice_types() + get_z_patterns_for_lattice(). "
                f"Full offsets useful for refinement but not exhaustive."
            )
        else:
            return (
                f"N={N} has LOW PREDICTABILITY ({entry['unique_patterns']} unique patterns). "
                f"Use get_top_lattice_types() only. "
                f"Z-patterns provide some guidance. "
                f"Exact xy-coordinates must be optimized per compound."
            )


# Convenience function for quick lookups
def predict_sublattice(N: int, lookup_path: str = 'sublattice_lookup.json') -> Dict:
    """
    Quick function to get predictions for a given N.
    
    Returns dict with lattice_type, z_pattern, full_offsets predictions.
    """
    predictor = SublatticePredictor(lookup_path)
    return predictor.predict(N)


# Common canonical patterns (for reference)
CANONICAL_PATTERNS = {
    # N=2 common patterns
    'layered_c': ['(0,0,0)', '(0,0,1/2)'],
    'wurtzite': ['(0,0,0)', '(1/3,2/3,1/2)'],
    'bcc_like': ['(0,0,0)', '(1/2,1/2,1/2)'],
    'ab_centered': ['(0,0,0)', '(1/2,1/2,0)'],
    
    # N=4 common patterns  
    'fcc_like': ['(0,0,0)', '(0,1/2,1/2)', '(1/2,0,1/2)', '(1/2,1/2,0)'],
    'tet_inplane': ['(0,0,0)', '(0,1/2,0)', '(1/2,0,0)', '(1/2,1/2,0)'],
}


if __name__ == '__main__':
    # Demo usage
    import sys
    
    lookup_path = 'sublattice_lookup.json'
    if len(sys.argv) > 1:
        lookup_path = sys.argv[1]
    
    try:
        predictor = SublatticePredictor(lookup_path)
    except FileNotFoundError:
        print(f"Error: Could not find {lookup_path}")
        print("Please provide path to sublattice_lookup.json")
        sys.exit(1)
    
    print("Sublattice Predictor Module")
    print("=" * 50)
    print(f"Available N values: {predictor.available_n_values()}")
    print()
    
    # Demo for a few N values
    for N in [2, 4, 8]:
        print(f"\n{'='*50}")
        print(f"N = {N}")
        print('='*50)
        
        stats = predictor.get_stats(N)
        print(f"Structures: {stats['total_structures']}")
        print(f"Unique patterns: {stats['unique_patterns']}")
        print(f"Predictability: {stats['predictability']}")
        print()
        print(f"Strategy: {predictor.suggest_strategy(N)}")
        print()
        
        print("Top 3 lattice types:")
        for system, centering, prob in predictor.get_top_lattice_types(N, top_k=3):
            print(f"  {system}-{centering}: {prob:.1%}")
        
        print()
        print("Top 3 z-patterns:")
        for system, centering, z_vals, prob in predictor.get_top_z_patterns(N, top_k=3):
            print(f"  {system}-{centering} z={z_vals}: {prob:.1%}")
