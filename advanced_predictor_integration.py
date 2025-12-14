"""
Advanced Lattice Predictor Integration

Integrates the advanced multi-source predictor with the Crystal Coordination Calculator.
Combines empirical patterns, template filling rules, and N-decomposition strategies.
"""

import os
import json
from pathlib import Path
from fractions import Fraction
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ============================================================================
# STRATEGY CONSTANTS (embedded for standalone operation)
# ============================================================================

LATTICE_FILLING_ORDER = {
    'Cubic-P': {
        'positions': ['(0,0,0)', '(1/2,1/2,1/2)', '(1/2,1/2,0)', '(1/2,0,1/2)', '(0,1/2,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(1/2,1/2,1/2)']],
            4: [['(0,0,0)', '(0,1/2,1/2)', '(1/2,0,1/2)', '(1/2,1/2,0)']],
        }
    },
    'Cubic-F': {
        'positions': ['(0,0,0)', '(1/2,1/2,1/2)', '(1/4,1/4,1/4)', '(3/4,3/4,3/4)'],
        'templates': {
            2: [['(0,0,0)', '(1/2,1/2,1/2)'], ['(0,0,0)', '(1/4,1/4,1/4)']],
        }
    },
    'Cubic-I': {
        'positions': ['(0,0,0)', '(1/2,1/2,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(1/2,1/2,1/2)']],
        }
    },
    'Hexagonal-P': {
        'positions': ['(0,0,0)', '(0,0,1/2)', '(1/3,2/3,1/2)', '(2/3,1/3,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(0,0,1/2)'], ['(0,0,0)', '(1/3,2/3,1/2)'], ['(0,0,0)', '(2/3,1/3,1/2)']],
            4: [['(0,0,0)', '(0,0,1/2)', '(2/3,1/3,0)', '(2/3,1/3,1/2)'],
                ['(0,0,0)', '(0,0,1/2)', '(1/3,2/3,0)', '(1/3,2/3,1/2)']],
        }
    },
    'Tetragonal-P': {
        'positions': ['(0,0,0)', '(1/2,1/2,0)', '(0,1/2,0)', '(1/2,0,0)', '(0,0,1/2)', '(1/2,1/2,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(1/2,1/2,0)'], ['(0,0,0)', '(1/2,1/2,1/2)']],
            4: [['(0,0,0)', '(0,1/2,0)', '(1/2,0,0)', '(1/2,1/2,0)']],
            8: [['(0,0,0)', '(0,0,1/2)', '(0,1/2,0)', '(0,1/2,1/2)',
                 '(1/2,0,0)', '(1/2,0,1/2)', '(1/2,1/2,0)', '(1/2,1/2,1/2)']],
        }
    },
    'Tetragonal-I': {
        'positions': ['(0,0,0)', '(0,1/2,1/4)', '(1/4,1/4,1/2)', '(3/4,3/4,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(0,1/2,1/4)']],
        }
    },
    'Orthorhombic-P': {
        'positions': ['(0,0,0)', '(0,1/2,1/2)', '(1/2,0,1/2)', '(1/2,1/2,0)', '(0,1/2,0)', '(1/2,0,0)', '(0,0,1/2)', '(1/2,1/2,1/2)'],
        'templates': {
            4: [['(0,0,0)', '(0,1/2,1/2)', '(1/2,0,1/2)', '(1/2,1/2,0)']],
            8: [['(0,0,0)', '(0,0,1/2)', '(0,1/2,0)', '(0,1/2,1/2)',
                 '(1/2,0,0)', '(1/2,0,1/2)', '(1/2,1/2,0)', '(1/2,1/2,1/2)']],
        }
    },
    'Rhombohedral-R': {
        'positions': ['(0,0,0)', '(0,0,1/2)', '(0,0,1/3)', '(0,0,2/3)', '(0,0,1/4)', '(0,0,3/4)'],
        'templates': {
            2: [['(0,0,0)', '(0,0,1/2)']],
            3: [['(0,0,0)', '(0,0,1/4)', '(0,0,3/4)'], ['(0,0,0)', '(0,0,1/3)', '(0,0,2/3)']],
        }
    },
    'Monoclinic-C': {
        'positions': ['(0,0,0)', '(0,0,1/2)', '(0,1/2,1/2)', '(2/3,0,0)'],
        'templates': {
            2: [['(0,0,0)', '(0,0,1/2)']],
        }
    },
    'Monoclinic-P': {
        'positions': ['(0,0,0)', '(0,0,1/2)', '(0,1/2,1/2)', '(0,1/2,0)'],
        'templates': {
            2: [['(0,0,0)', '(0,1/2,1/2)'], ['(0,0,0)', '(0,0,1/2)']],
        }
    },
    'Orthorhombic-C': {
        'positions': ['(0,0,0)', '(0,0,1/2)', '(1/2,0,0)', '(1/2,0,1/2)'],
        'templates': {
            2: [['(0,0,0)', '(0,0,1/2)']],
            4: [['(0,0,0)', '(0,0,1/2)', '(1/2,0,0)', '(1/2,0,1/2)']],
        }
    },
}

Z_LAYER_TEMPLATES = {
    1: ['0'],
    2: ['0', '1/2'],
    3: ['0', '1/3', '2/3'],
    4: ['0', '1/4', '1/2', '3/4'],
    6: ['0', '1/6', '1/3', '1/2', '2/3', '5/6'],
}

IN_PLANE_TEMPLATES = {
    1: [['(0,0,Z)']],
    2: [['(0,0,Z)', '(1/2,1/2,Z)']],
    4: [['(0,0,Z)', '(0,1/2,Z)', '(1/2,0,Z)', '(1/2,1/2,Z)']],
}

N_DECOMPOSITIONS = {
    2: [(1, 2, 86), (2, 1, 14)],
    3: [(1, 3, 55), (3, 1, 27)],
    4: [(1, 4, 48), (2, 2, 47), (4, 1, 5)],
    6: [(3, 2, 53), (2, 3, 26), (1, 6, 18), (6, 1, 3)],
    8: [(2, 4, 64), (4, 2, 32), (1, 8, 2), (8, 1, 2)],
    9: [(3, 3, 55)],
    10: [(2, 5, 18), (5, 2, 18)],
    12: [(3, 4, 40), (4, 3, 30), (2, 6, 21), (6, 2, 4)],
    16: [(4, 4, 65), (8, 2, 23), (2, 8, 3)],
}

# Mapping from predictor lattice names to app's bravais_type
LATTICE_TO_BRAVAIS = {
    'Cubic-P': ('Cubic', 'cubic_P'),
    'Cubic-I': ('Cubic', 'cubic_I'),
    'Cubic-F': ('Cubic', 'cubic_F'),
    'Tetragonal-P': ('Tetragonal', 'tetragonal_P'),
    'Tetragonal-I': ('Tetragonal', 'tetragonal_I'),
    'Orthorhombic-P': ('Orthorhombic', 'orthorhombic_P'),
    'Orthorhombic-I': ('Orthorhombic', 'orthorhombic_I'),
    'Orthorhombic-F': ('Orthorhombic', 'orthorhombic_F'),
    'Orthorhombic-C': ('Orthorhombic', 'orthorhombic_C'),
    'Hexagonal-P': ('Hexagonal', 'hexagonal_P'),
    'Hexagonal-H': ('Hexagonal', 'hexagonal_H'),
    'Rhombohedral-R': ('Rhombohedral', 'rhombohedral_P'),
    'Rhombohedral-P': ('Rhombohedral', 'rhombohedral_P'),
    'Monoclinic-P': ('Monoclinic', 'monoclinic_P'),
    'Monoclinic-C': ('Monoclinic', 'monoclinic_C'),
    'Triclinic-P': ('Monoclinic', 'monoclinic_P'),  # Map to monoclinic
}


def parse_fraction_str(s: str) -> float:
    """Parse a fraction string like '1/2', '2/3', or a number to float."""
    s = s.strip()
    if '/' in s:
        try:
            return float(Fraction(s))
        except:
            parts = s.split('/')
            return float(parts[0]) / float(parts[1])
    return float(s)


def parse_offset_str(offset_str: str) -> Tuple[float, float, float]:
    """Parse offset string like '(0,0,0)' or '(1/2,1/2,1/2)' to tuple."""
    clean = offset_str.strip().strip('()')
    parts = clean.split(',')
    if len(parts) != 3:
        raise ValueError(f"Invalid offset format: {offset_str}")
    return tuple(parse_fraction_str(p) for p in parts)


def get_default_c_ratio(lattice_type: str, num_metals: int) -> float:
    """Get default c/a ratio for a lattice type."""
    if lattice_type == 'Cubic':
        return 1.0
    elif lattice_type == 'Hexagonal':
        return 1.633
    elif lattice_type == 'Tetragonal':
        return float(num_metals)
    elif lattice_type == 'Orthorhombic':
        return float(num_metals)
    elif lattice_type == 'Rhombohedral':
        return 1.0
    elif lattice_type == 'Monoclinic':
        return float(num_metals)
    return 1.0


class AdvancedLatticePredictor:
    """
    Advanced predictor combining empirical data, templates, and N-decomposition.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize with optional data directory."""
        self.data_dir = Path(data_dir) if data_dir else None
        self.lookup = {}
        self.strategy = {}
        self._load_data()
    
    def _load_data(self):
        """Load empirical data from JSON files."""
        # Try multiple locations
        search_paths = [
            self.data_dir,
            Path('.'),
            Path(__file__).parent if '__file__' in dir() else None,
            Path('/home/claude'),
            Path('/mnt/user-data/outputs'),
        ]
        
        for base in search_paths:
            if base is None:
                continue
            
            lookup_path = base / 'sublattice_lookup.json'
            if lookup_path.exists() and not self.lookup:
                try:
                    with open(lookup_path) as f:
                        self.lookup = json.load(f)
                except:
                    pass
            
            strategy_path = base / 'lattice_prediction_strategy.json'
            if strategy_path.exists() and not self.strategy:
                try:
                    with open(strategy_path) as f:
                        self.strategy = json.load(f)
                except:
                    pass
    
    def is_available(self) -> bool:
        """Check if predictor has loaded data."""
        return bool(self.lookup) or bool(LATTICE_FILLING_ORDER)
    
    def get_lattice_probabilities(self, n: int) -> Dict[str, float]:
        """Get probability distribution of lattice types for given N."""
        key = str(n)
        if key in self.lookup and 'lattice_type' in self.lookup[key]:
            total = self.lookup[key]['total_structures']
            lattice_data = self.lookup[key]['lattice_type']
            
            if isinstance(lattice_data, list):
                return {
                    f"{item['system']}-{item['centering']}": item['count'] / total
                    for item in lattice_data
                }
            return {
                lt: info['count'] / total
                for lt, info in lattice_data.items()
            }
        return {}
    
    def get_observed_patterns(self, n: int) -> List[Dict]:
        """Get empirically observed patterns for given N."""
        key = str(n)
        
        # Try strategy file first
        if 'predictions_by_n' in self.strategy:
            patterns = self.strategy['predictions_by_n'].get(key, [])
            if patterns:
                return patterns
        
        # Fall back to lookup file full_offsets
        if key in self.lookup and 'full_offsets' in self.lookup[key]:
            total = self.lookup[key]['total_structures']
            results = []
            for item in self.lookup[key]['full_offsets'][:15]:
                results.append({
                    'lattice': f"{item['system']}-{item['centering']}",
                    'offsets': item['offsets'],
                    'probability': item['probability'] * 100,
                    'source': 'observed'
                })
            return results
        
        return []
    
    def generate_from_templates(self, n: int, lattice: str) -> List[List[str]]:
        """Generate patterns from lattice-specific templates."""
        patterns = []
        
        if lattice in LATTICE_FILLING_ORDER:
            templates = LATTICE_FILLING_ORDER[lattice].get('templates', {})
            if n in templates:
                patterns.extend(templates[n])
        
        return patterns
    
    def generate_from_decomposition(self, n: int) -> List[Tuple[str, List[str]]]:
        """Generate patterns from N = in_plane × z_layers decomposition."""
        patterns = []
        
        decomps = N_DECOMPOSITIONS.get(n, [])
        if not decomps:
            for z_layers in [1, 2, 4]:
                if n % z_layers == 0:
                    in_plane = n // z_layers
                    decomps.append((in_plane, z_layers, 10))
        
        for in_plane, z_layers, _ in decomps:
            if z_layers not in Z_LAYER_TEMPLATES:
                continue
            if in_plane not in IN_PLANE_TEMPLATES:
                continue
            
            z_vals = Z_LAYER_TEMPLATES[z_layers]
            
            for xy_templates in IN_PLANE_TEMPLATES[in_plane]:
                offsets = []
                for xy in xy_templates:
                    for z in z_vals:
                        offset = xy.replace('Z', z)
                        offsets.append(offset)
                
                if len(offsets) == n:
                    for lattice in ['Orthorhombic-P', 'Tetragonal-P', 'Cubic-P', 'Monoclinic-P']:
                        patterns.append((lattice, offsets))
        
        return patterns
    
    def predict(self, n: int, top_k: int = 15) -> List[Dict]:
        """
        Generate top-k most likely configurations for N positions.
        
        Returns list of dicts with: lattice, offsets, probability, source
        """
        candidates = []
        
        # 1. Observed patterns (highest priority)
        observed = self.get_observed_patterns(n)
        for p in observed:
            candidates.append({
                'lattice': p['lattice'],
                'offsets': p['offsets'],
                'probability': p['probability'],
                'source': p.get('source', 'observed'),
            })
        
        # 2. Template-based patterns
        lattice_probs = self.get_lattice_probabilities(n)
        for lattice, prob in sorted(lattice_probs.items(), key=lambda x: -x[1])[:8]:
            templates = self.generate_from_templates(n, lattice)
            for template in templates:
                key = (lattice, tuple(sorted(template)))
                if not any(c['lattice'] == lattice and tuple(sorted(c['offsets'])) == key[1] 
                          for c in candidates):
                    candidates.append({
                        'lattice': lattice,
                        'offsets': template,
                        'probability': prob * 50,
                        'source': 'template',
                    })
        
        # 3. Decomposition-based patterns
        decomp_patterns = self.generate_from_decomposition(n)
        for lattice, offsets in decomp_patterns:
            key = (lattice, tuple(sorted(offsets)))
            if not any(c['lattice'] == lattice and tuple(sorted(c['offsets'])) == key[1]
                      for c in candidates):
                base_prob = lattice_probs.get(lattice, 0.05)
                candidates.append({
                    'lattice': lattice,
                    'offsets': offsets,
                    'probability': base_prob * 20,
                    'source': 'decomposition',
                })
        
        # Sort and deduplicate
        seen = set()
        unique = []
        for c in sorted(candidates, key=lambda x: -x['probability']):
            key = (c['lattice'], tuple(sorted(c['offsets'])))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        
        return unique[:top_k]
    
    def get_search_configs(
        self,
        num_metals: int,
        top_k: int = 15,
        always_include_cubic: bool = True,
        always_include_common: bool = True
    ) -> List[dict]:
        """
        Get search configurations for the app.
        
        Converts predictions to app-compatible config format.
        
        Args:
            num_metals: Number of metal sublattices
            top_k: Maximum predictions to return
            always_include_cubic: Always include cubic P/I/F (default True)
            always_include_common: Always include tetragonal P/I and hexagonal H/P (default True)
        """
        predictions = self.predict(num_metals, top_k=top_k)
        configs = []
        seen_ids = set()
        
        for i, pred in enumerate(predictions):
            lattice_name = pred['lattice']
            offset_strs = pred['offsets']
            probability = pred['probability']
            source = pred['source']
            
            # Map to app's lattice/bravais types
            if lattice_name not in LATTICE_TO_BRAVAIS:
                continue
            
            lattice_type, bravais_type = LATTICE_TO_BRAVAIS[lattice_name]
            
            # Parse offsets
            try:
                offsets = [parse_offset_str(o) for o in offset_strs]
            except ValueError:
                continue
            
            # Generate config ID
            config_id = f"ADV-N{num_metals}-{bravais_type}-{i+1}"
            if config_id in seen_ids:
                config_id = f"{config_id}-{source}"
            seen_ids.add(config_id)
            
            c_ratio = get_default_c_ratio(lattice_type, num_metals)
            
            configs.append({
                'id': config_id,
                'lattice': lattice_type,
                'bravais_type': bravais_type,
                'offsets': offsets,
                'pattern': f"{source} ({probability:.1f}%)",
                'c_ratio': c_ratio,
                'probability': probability / 100.0,
                'source': f'advanced_{source}'
            })
        
        # Always include cubic P/I/F
        if always_include_cubic:
            configs = self._ensure_cubic_configs(configs, num_metals)
        
        # Always include tetragonal P/I and hexagonal H/P
        if always_include_common:
            configs = self._ensure_common_configs(configs, num_metals)
        
        return configs
    
    def _ensure_cubic_configs(self, configs: List[dict], num_metals: int) -> List[dict]:
        """Ensure cubic P/I/F are always included."""
        try:
            from lattice_configs import get_configs_for_n
            
            existing_bravais = set(c['bravais_type'] for c in configs)
            cubic_types = ['cubic_P', 'cubic_I', 'cubic_F']
            
            all_configs = get_configs_for_n(num_metals)
            arity0_configs = all_configs.get('arity0', [])
            
            for cubic_type in cubic_types:
                if cubic_type not in existing_bravais:
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
                            'probability': 0.0,
                            'source': 'always_cubic'
                        })
        except ImportError:
            # lattice_configs not available, add basic cubic templates
            existing = set(c['bravais_type'] for c in configs)
            
            if 'cubic_P' not in existing and num_metals <= 4:
                templates = LATTICE_FILLING_ORDER.get('Cubic-P', {}).get('templates', {})
                if num_metals in templates:
                    for template in templates[num_metals]:
                        offsets = [parse_offset_str(o) for o in template]
                        configs.append({
                            'id': f"CUBIC-cP-{num_metals}",
                            'lattice': 'Cubic',
                            'bravais_type': 'cubic_P',
                            'offsets': offsets,
                            'pattern': 'template (always)',
                            'c_ratio': 1.0,
                            'probability': 0.0,
                            'source': 'always_cubic'
                        })
        
        return configs
    
    def _ensure_common_configs(self, configs: List[dict], num_metals: int) -> List[dict]:
        """Ensure tetragonal P/I and hexagonal H/P are always included."""
        try:
            from lattice_configs import get_configs_for_n
            
            existing_bravais = set(c['bravais_type'] for c in configs)
            
            # Common types to always include with their c/a ratios
            common_types = {
                'tetragonal_P': ('Tetragonal', float(num_metals)),
                'tetragonal_I': ('Tetragonal', float(num_metals)),
                'hexagonal_H': ('Hexagonal', 1.633),
                'hexagonal_P': ('Hexagonal', 1.633),
            }
            
            all_configs = get_configs_for_n(num_metals)
            arity0_configs = all_configs.get('arity0', [])
            
            for bravais_type, (lattice_type, c_ratio) in common_types.items():
                if bravais_type not in existing_bravais:
                    matching = [c for c in arity0_configs 
                               if c.bravais_type == bravais_type and c.offsets is not None]
                    
                    for lc in matching:
                        prefix = bravais_type.split('_')[0].upper()[:3]  # TET or HEX
                        configs.append({
                            'id': f"{prefix}-{lc.id}",
                            'lattice': lattice_type,
                            'bravais_type': bravais_type,
                            'offsets': list(lc.offsets),
                            'pattern': f"{lc.pattern} (always)",
                            'c_ratio': c_ratio,
                            'probability': 0.0,
                            'source': 'always_common'
                        })
        except ImportError:
            # lattice_configs not available, add basic templates
            existing = set(c['bravais_type'] for c in configs)
            
            # Add basic tetragonal and hexagonal templates if available
            for lattice_key, bravais_type, lattice_type, c_ratio in [
                ('Tetragonal-P', 'tetragonal_P', 'Tetragonal', float(num_metals)),
                ('Tetragonal-I', 'tetragonal_I', 'Tetragonal', float(num_metals)),
                ('Hexagonal-H', 'hexagonal_H', 'Hexagonal', 1.633),
                ('Hexagonal-P', 'hexagonal_P', 'Hexagonal', 1.633),
            ]:
                if bravais_type not in existing and num_metals <= 4:
                    templates = LATTICE_FILLING_ORDER.get(lattice_key, {}).get('templates', {})
                    if num_metals in templates:
                        for template in templates[num_metals]:
                            offsets = [parse_offset_str(o) for o in template]
                            prefix = bravais_type.split('_')[0].upper()[:3]
                            configs.append({
                                'id': f"{prefix}-{bravais_type}-{num_metals}",
                                'lattice': lattice_type,
                                'bravais_type': bravais_type,
                                'offsets': offsets,
                                'pattern': 'template (always)',
                                'c_ratio': c_ratio,
                                'probability': 0.0,
                                'source': 'always_common'
                            })
        
        return configs


# Global instance
_predictor_instance = None


def get_predictor(data_dir: Optional[str] = None) -> AdvancedLatticePredictor:
    """Get or create the global predictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AdvancedLatticePredictor(data_dir)
    return _predictor_instance


def get_predicted_search_configs(num_metals: int, **kwargs) -> List[dict]:
    """Convenience function to get search configs."""
    predictor = get_predictor()
    return predictor.get_search_configs(num_metals, **kwargs)


if __name__ == '__main__':
    predictor = AdvancedLatticePredictor()
    
    print("Advanced Lattice Predictor")
    print("=" * 60)
    print(f"Data available: {predictor.is_available()}")
    print(f"Lookup entries: {len(predictor.lookup)}")
    print()
    
    for n in [2, 4, 6, 8]:
        print(f"\nN = {n}")
        print("-" * 40)
        
        configs = predictor.get_search_configs(n, top_k=10)
        print(f"Generated {len(configs)} configs:")
        
        for c in configs[:5]:
            print(f"  {c['id']}: {c['bravais_type']} - {c['pattern']}")
