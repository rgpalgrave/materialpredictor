"""
Predictor Integration Module

Bridges the AFLOW-based sublattice predictor with the Crystal Coordination Calculator.
Converts predictor outputs to the config format used by get_default_search_configs().
"""

import os
import re
from typing import List, Dict, Tuple, Optional
from fractions import Fraction

# Try to import the predictor - handle case where it's not available
try:
    from sublattice_predictor_module import SublatticePredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    SublatticePredictor = None


# Mapping from (system, centering) to app's (lattice_type, bravais_type)
# The app uses a specific naming convention for Bravais types
LATTICE_MAPPING = {
    # Cubic
    ('Cubic', 'P'): ('Cubic', 'cubic_P'),
    ('Cubic', 'I'): ('Cubic', 'cubic_I'),
    ('Cubic', 'F'): ('Cubic', 'cubic_F'),
    
    # Tetragonal
    ('Tetragonal', 'P'): ('Tetragonal', 'tetragonal_P'),
    ('Tetragonal', 'I'): ('Tetragonal', 'tetragonal_I'),
    
    # Orthorhombic
    ('Orthorhombic', 'P'): ('Orthorhombic', 'orthorhombic_P'),
    ('Orthorhombic', 'I'): ('Orthorhombic', 'orthorhombic_I'),
    ('Orthorhombic', 'F'): ('Orthorhombic', 'orthorhombic_F'),
    ('Orthorhombic', 'C'): ('Orthorhombic', 'orthorhombic_C'),
    ('Orthorhombic', 'A'): ('Orthorhombic', 'orthorhombic_A'),
    
    # Hexagonal  
    ('Hexagonal', 'P'): ('Hexagonal', 'hexagonal_P'),
    ('Hexagonal', 'H'): ('Hexagonal', 'hexagonal_H'),  # HCP
    
    # Rhombohedral - map to Hexagonal in the app's convention
    ('Rhombohedral', 'P'): ('Rhombohedral', 'rhombohedral_P'),
    ('Rhombohedral', 'R'): ('Rhombohedral', 'rhombohedral_P'),  # R centering = primitive rhombohedral
    
    # Monoclinic
    ('Monoclinic', 'P'): ('Monoclinic', 'monoclinic_P'),
    ('Monoclinic', 'C'): ('Monoclinic', 'monoclinic_C'),
    
    # Triclinic
    ('Triclinic', 'P'): ('Monoclinic', 'monoclinic_P'),  # Map triclinic to monoclinic_P as fallback
}


def parse_fraction(s: str) -> float:
    """
    Parse a fraction string like '1/2', '2/3', or a number to float.
    
    Args:
        s: String like '0', '1/2', '2/3', '0.5'
    
    Returns:
        Float value
    """
    s = s.strip()
    if '/' in s:
        try:
            frac = Fraction(s)
            return float(frac)
        except:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
    else:
        return float(s)


def parse_offset_string(offset_str: str) -> Tuple[float, float, float]:
    """
    Parse offset string like "(0,0,0)" or "(1/2,1/2,1/2)" to tuple.
    
    Args:
        offset_str: String like "(0,0,0)", "(1/2,1/2,0)", "(0.333,0.667,0)"
    
    Returns:
        Tuple of (x, y, z) as floats
    """
    # Remove parentheses and split
    clean = offset_str.strip().strip('()')
    parts = clean.split(',')
    
    if len(parts) != 3:
        raise ValueError(f"Invalid offset format: {offset_str}")
    
    return tuple(parse_fraction(p) for p in parts)


def get_default_c_ratio(lattice_type: str, num_metals: int) -> float:
    """
    Get default c/a ratio for a lattice type.
    
    Args:
        lattice_type: 'Cubic', 'Tetragonal', 'Hexagonal', etc.
        num_metals: Number of metal sublattices (L)
    
    Returns:
        Default c/a ratio
    """
    if lattice_type == 'Cubic':
        return 1.0
    elif lattice_type == 'Hexagonal':
        return 1.633  # Ideal for HCP-like
    elif lattice_type == 'Tetragonal':
        # Use L as default c/a for tetragonal (layered assumption)
        return float(num_metals)
    elif lattice_type == 'Orthorhombic':
        return float(num_metals)  # Similar to tetragonal
    elif lattice_type == 'Rhombohedral':
        return 1.0
    elif lattice_type == 'Monoclinic':
        return float(num_metals)
    else:
        return 1.0


class PredictorIntegration:
    """
    Integration layer between the AFLOW predictor and the Crystal Coordination Calculator.
    """
    
    def __init__(self, lookup_path: Optional[str] = None):
        """
        Initialize the predictor integration.
        
        Args:
            lookup_path: Path to sublattice_lookup.json. If None, tries common locations.
        """
        self.predictor = None
        self.available = False
        
        if not PREDICTOR_AVAILABLE:
            return
        
        # Try to find the lookup file
        if lookup_path is None:
            possible_paths = [
                'sublattice_lookup.json',
                os.path.join(os.path.dirname(__file__), 'sublattice_lookup.json'),
                '/home/claude/sublattice_lookup.json',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lookup_path = path
                    break
        
        if lookup_path and os.path.exists(lookup_path):
            try:
                self.predictor = SublatticePredictor(lookup_path)
                self.available = True
            except Exception as e:
                print(f"Warning: Could not load predictor: {e}")
    
    def is_available(self) -> bool:
        """Check if the predictor is available."""
        return self.available
    
    def get_predictability(self, num_metals: int) -> str:
        """
        Get predictability level for given N.
        
        Returns: 'high', 'moderate', 'low', or 'unknown'
        """
        if not self.available or not self.predictor.has_data(num_metals):
            return 'unknown'
        
        stats = self.predictor.get_stats(num_metals)
        return stats['predictability']
    
    def get_search_configs_from_predictor(
        self,
        num_metals: int,
        top_k_lattices: int = 5,
        top_k_offsets: int = 10,
        include_fallback: bool = True,
        always_include_cubic: bool = True
    ) -> List[dict]:
        """
        Get search configurations using the AFLOW-based predictor.
        
        Uses different strategies based on predictability:
        - HIGH (N=1,5,7): Use exact offsets directly
        - MODERATE (N=2,3): Use lattice types + offsets with good coverage
        - LOW (N=4,6,8+): Use lattice types primarily, top offsets as hints
        
        Args:
            num_metals: Number of metal sublattices (L)
            top_k_lattices: Number of top lattice types to consider
            top_k_offsets: Number of top offset patterns to include
            include_fallback: Include fallback configs if predictor fails
            always_include_cubic: Always include cubic P/I/F configs (fast to compute)
        
        Returns:
            List of config dicts compatible with app.py
        """
        if not self.available:
            return self._get_fallback_configs(num_metals) if include_fallback else []
        
        if not self.predictor.has_data(num_metals):
            return self._get_fallback_configs(num_metals) if include_fallback else []
        
        configs = []
        predictability = self.get_predictability(num_metals)
        
        # Get predictions
        predictions = self.predictor.predict(num_metals)
        
        if predictability == 'high':
            # Use exact offsets directly - they're highly predictive
            configs = self._configs_from_offsets(
                predictions['full_offsets'][:top_k_offsets],
                num_metals
            )
        
        elif predictability == 'moderate':
            # Use combination of lattice types and best offsets
            # First add configs from top offsets
            configs = self._configs_from_offsets(
                predictions['full_offsets'][:top_k_offsets],
                num_metals
            )
            
            # Add configs from top lattice types that aren't already covered
            existing_bravais = set(c['bravais_type'] for c in configs)
            lattice_configs = self._configs_from_lattice_types(
                predictions['lattice_type'][:top_k_lattices],
                num_metals,
                exclude_bravais=existing_bravais
            )
            configs.extend(lattice_configs)
        
        else:  # low predictability
            # Primarily use lattice types, offsets are just hints
            # Get top lattice types
            configs = self._configs_from_lattice_types(
                predictions['lattice_type'][:top_k_lattices],
                num_metals
            )
            
            # Add a few top offset patterns as bonus search candidates
            if predictions['full_offsets']:
                offset_configs = self._configs_from_offsets(
                    predictions['full_offsets'][:min(5, top_k_offsets)],
                    num_metals
                )
                # Only add if not duplicating
                existing_ids = set(c['id'] for c in configs)
                for oc in offset_configs:
                    if oc['id'] not in existing_ids:
                        configs.append(oc)
        
        # Ensure we have at least some configs
        if not configs and include_fallback:
            configs = self._get_fallback_configs(num_metals)
        
        # Always include cubic P/I/F - they're fast to compute
        if always_include_cubic:
            configs = self._ensure_cubic_configs(configs, num_metals)
        
        return configs
    
    def _ensure_cubic_configs(self, configs: List[dict], num_metals: int) -> List[dict]:
        """
        Ensure cubic P/I/F configs are always included.
        
        Cubic structures are fast to compute and commonly important,
        so we always include them even if the predictor doesn't suggest them.
        """
        from lattice_configs import get_configs_for_n
        
        # Check which cubic types are already present
        existing_bravais = set(c['bravais_type'] for c in configs)
        cubic_types = ['cubic_P', 'cubic_I', 'cubic_F']
        
        # Get all configs for this N
        all_configs = get_configs_for_n(num_metals)
        arity0_configs = all_configs.get('arity0', [])
        
        # Add missing cubic configs
        for cubic_type in cubic_types:
            if cubic_type not in existing_bravais:
                # Find configs with this bravais type
                matching = [c for c in arity0_configs 
                           if c.bravais_type == cubic_type and c.offsets is not None]
                
                for lc in matching:
                    configs.append({
                        'id': f"CUBIC-{lc.id}",
                        'lattice': 'Cubic',
                        'bravais_type': cubic_type,
                        'offsets': list(lc.offsets),
                        'pattern': f"{lc.pattern} (always)",
                        'c_ratio': 1.0,
                        'probability': 0.0,  # Not from predictor
                        'source': 'always_cubic'
                    })
        
        return configs
    
    def _configs_from_offsets(
        self,
        offset_predictions,
        num_metals: int
    ) -> List[dict]:
        """
        Convert offset predictions to config dicts.
        """
        configs = []
        
        for i, pred in enumerate(offset_predictions):
            system = pred.system
            centering = pred.centering
            offset_strs = pred.offsets
            probability = pred.probability
            
            # Get lattice type and bravais type
            mapping_key = (system, centering)
            if mapping_key not in LATTICE_MAPPING:
                continue  # Skip unknown lattice types
            
            lattice_type, bravais_type = LATTICE_MAPPING[mapping_key]
            
            # Parse offsets
            try:
                offsets = [parse_offset_string(o) for o in offset_strs]
            except ValueError:
                continue  # Skip if offsets can't be parsed
            
            # Get c/a ratio
            c_ratio = get_default_c_ratio(lattice_type, num_metals)
            
            # Create config ID
            config_id = f"PRED-N{num_metals}-{bravais_type}-{i+1}"
            
            # Pattern description
            pattern = f"AFLOW {probability:.1%}"
            
            configs.append({
                'id': config_id,
                'lattice': lattice_type,
                'bravais_type': bravais_type,
                'offsets': offsets,
                'pattern': pattern,
                'c_ratio': c_ratio,
                'probability': probability,
                'source': 'aflow_predictor'
            })
        
        return configs
    
    def _configs_from_lattice_types(
        self,
        lattice_predictions,
        num_metals: int,
        exclude_bravais: Optional[set] = None
    ) -> List[dict]:
        """
        Create configs from lattice type predictions using standard offsets.
        
        For each lattice type, we use the canonical offset patterns
        from lattice_configs.py.
        """
        from lattice_configs import get_configs_for_n
        
        if exclude_bravais is None:
            exclude_bravais = set()
        
        configs = []
        added_ids = set()
        all_lattice_configs = get_configs_for_n(num_metals)
        arity0_configs = all_lattice_configs.get('arity0', [])
        
        for pred in lattice_predictions:
            system = pred.system
            centering = pred.centering
            probability = pred.probability
            
            # Get lattice type and bravais type
            mapping_key = (system, centering)
            if mapping_key not in LATTICE_MAPPING:
                continue
            
            lattice_type, bravais_type = LATTICE_MAPPING[mapping_key]
            
            if bravais_type in exclude_bravais:
                continue
            
            # Find matching configs from lattice_configs.py
            matching = [c for c in arity0_configs 
                       if c.bravais_type == bravais_type and c.offsets is not None]
            
            if not matching:
                # Try to find configs with same lattice type
                matching = [c for c in arity0_configs 
                           if c.lattice == lattice_type and c.offsets is not None][:3]
            
            for lc in matching[:2]:  # Limit to 2 per lattice type
                if lc.id in added_ids:
                    continue
                added_ids.add(lc.id)
                
                c_ratio = get_default_c_ratio(lattice_type, num_metals)
                
                config_id = f"PRED-{lc.id}"
                
                configs.append({
                    'id': config_id,
                    'lattice': lc.lattice,  # Use actual lattice from config
                    'bravais_type': lc.bravais_type,
                    'offsets': list(lc.offsets),
                    'pattern': f"{lc.pattern} ({probability:.1%})",
                    'c_ratio': c_ratio,
                    'probability': probability,
                    'source': 'aflow_lattice_type'
                })
        
        return configs
    
    def _get_fallback_configs(self, num_metals: int) -> List[dict]:
        """
        Get fallback configurations when predictor is not available.
        
        Uses the same logic as the original get_default_search_configs.
        """
        from lattice_configs import get_configs_for_n
        
        default_bravais = {
            'cubic_P': {'lattice': 'Cubic', 'c_ratio': 1.0},
            'cubic_F': {'lattice': 'Cubic', 'c_ratio': 1.0},
            'cubic_I': {'lattice': 'Cubic', 'c_ratio': 1.0},
            'hexagonal_H': {'lattice': 'Hexagonal', 'c_ratio': 1.633},
            'hexagonal_P': {'lattice': 'Hexagonal', 'c_ratio': 1.633},
            'tetragonal_P': {'lattice': 'Tetragonal', 'c_ratio': float(num_metals)},
            'tetragonal_I': {'lattice': 'Tetragonal', 'c_ratio': float(num_metals)},
        }
        
        all_configs = get_configs_for_n(num_metals)
        arity0_configs = all_configs.get('arity0', [])
        
        search_configs = []
        
        for config in arity0_configs:
            if config.offsets is None:
                continue
            
            bravais = config.bravais_type
            if bravais not in default_bravais:
                continue
            
            bravais_info = default_bravais[bravais]
            
            search_configs.append({
                'id': config.id,
                'lattice': bravais_info['lattice'],
                'bravais_type': bravais,
                'offsets': list(config.offsets),
                'pattern': config.pattern,
                'c_ratio': bravais_info['c_ratio'],
                'source': 'fallback'
            })
        
        return search_configs


# Global instance for easy access
_predictor_instance = None


def get_predictor() -> PredictorIntegration:
    """Get or create the global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PredictorIntegration()
    return _predictor_instance


def get_predicted_search_configs(num_metals: int, **kwargs) -> List[dict]:
    """
    Convenience function to get search configs using the predictor.
    
    Args:
        num_metals: Number of metal sublattices
        **kwargs: Additional arguments passed to get_search_configs_from_predictor
    
    Returns:
        List of config dicts
    """
    predictor = get_predictor()
    return predictor.get_search_configs_from_predictor(num_metals, **kwargs)


if __name__ == '__main__':
    # Demo/test
    integration = PredictorIntegration()
    
    if integration.is_available():
        print("Predictor integration available!")
        print()
        
        for n in [1, 2, 3, 4, 5, 6, 7, 8]:
            print(f"\nN = {n}")
            print(f"  Predictability: {integration.get_predictability(n)}")
            
            configs = integration.get_search_configs_from_predictor(n)
            print(f"  Generated {len(configs)} search configs")
            
            for c in configs[:3]:
                print(f"    - {c['id']}: {c['lattice']}/{c['bravais_type']} ({c.get('probability', 0):.1%})")
    else:
        print("Predictor not available - check sublattice_lookup.json path")
