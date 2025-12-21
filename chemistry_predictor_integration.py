"""
Chemistry-Based Lattice Predictor Integration

Integrates orbit_search_generator and lattice_search modules to provide
chemistry-driven lattice configuration predictions.

This replaces the empirical predictor with a first-principles approach:
1. Derive stoichiometry from charge balance
2. Predict coordination numbers using Pauling radius-ratio rules
3. Generate CN hypotheses (bond allocation models A/B/C/D)
4. Run lattice enumeration with constraint filtering
5. Return ranked configurations for the sphere intersection model
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

# Add module paths
_module_dir = Path(__file__).parent
if str(_module_dir) not in sys.path:
    sys.path.insert(0, str(_module_dir))


def _ensure_imports():
    """Lazy import of heavy modules."""
    global orbit_search_generator, lattice_search_integration, lattice_search
    
    try:
        import orbit_search_generator
        import lattice_search_integration
        import lattice_search
        return True
    except ImportError as e:
        print(f"Warning: Could not import chemistry predictor modules: {e}")
        return False


# Mapping from bravais_type to lattice display name
BRAVAIS_TO_LATTICE = {
    'cubic_P': 'Cubic',
    'cubic_I': 'Cubic',
    'cubic_F': 'Cubic',
    'tetragonal_P': 'Tetragonal',
    'tetragonal_I': 'Tetragonal',
    'hexagonal_P': 'Hexagonal',
    'hexagonal_H': 'Hexagonal',
    'orthorhombic_P': 'Orthorhombic',
    'orthorhombic_I': 'Orthorhombic',
    'orthorhombic_F': 'Orthorhombic',
    'orthorhombic_C': 'Orthorhombic',
    'rhombohedral_P': 'Rhombohedral',
    'monoclinic_P': 'Monoclinic',
    'monoclinic_C': 'Monoclinic',
}

# Default c/a ratios by lattice type
DEFAULT_C_RATIOS = {
    'Cubic': 1.0,
    'Tetragonal': 1.0,
    'Hexagonal': 1.633,
    'Orthorhombic': 1.0,
    'Rhombohedral': 1.0,
    'Monoclinic': 1.0,
}


def metals_to_chemistry(
    metals: List[Dict],
    anion_symbol: str,
    anion_charge: int
) -> Dict:
    """
    Convert app metals format to orbit_search_generator chemistry format.
    
    Args:
        metals: List of {'symbol': str, 'charge': int, 'ratio': int, 'cn': int, 'radius': float}
        anion_symbol: e.g., 'O', 'F', 'Cl'
        anion_charge: Positive integer (magnitude of negative charge)
    
    Returns:
        Chemistry dict for orbit_search_generator
    """
    cations = []
    cn_overrides = {}
    
    for m in metals:
        cations.append({
            'element': m['symbol'],
            'charge': m['charge'],
            'count': m.get('ratio', 1)
        })
        # Use user-specified CN as override
        if 'cn' in m:
            cn_overrides[m['symbol']] = m['cn']
    
    chemistry = {
        'cations': cations,
        'anion': {
            'element': anion_symbol,
            'charge': -anion_charge  # Negative
        }
    }
    
    # Add CN overrides if specified
    if cn_overrides:
        chemistry['overrides'] = {'cation_cn': cn_overrides}
    
    return chemistry


def get_chemistry_search_configs(
    metals: List[Dict],
    anion_symbol: str,
    anion_charge: int,
    top_k: int = 15,
    run_lattice_search: bool = False,
    diagonal_only: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Get search configurations using chemistry-based prediction.
    
    Uses Pauling radius-ratio rules and bond allocation models
    to predict likely lattice configurations.
    
    Note: The CN values in SearchSpecs refer to cation-ANION coordination
    (chemistry), while lattice_search computes cation-CATION neighbors
    (geometry). We filter by geometry constraints only during lattice search.
    """
    if not _ensure_imports():
        return []
    
    # Convert to chemistry format
    chemistry = metals_to_chemistry(metals, anion_symbol, anion_charge)
    
    if verbose:
        print(f"Chemistry: {chemistry}")
    
    # Generate search specs
    try:
        specs = orbit_search_generator.generate_search_specs(
            chemistry, 
            top_X=top_k,
            m_max=4,
            M_max=32
        )
    except Exception as e:
        print(f"Warning: Failed to generate search specs: {e}")
        return []
    
    if verbose:
        print(f"Generated {len(specs)} SearchSpecs")
    
    configs = []
    num_metals = len(metals)
    
    for i, spec in enumerate(specs):
        # Extract info from spec
        label = spec.get('label', f'Spec-{i}')
        orbit_sizes = spec.get('orbit_sizes', [1] * num_metals)
        orbit_species = spec.get('orbit_species', [m['symbol'] for m in metals])
        m_list = spec.get('M_list', [2, 4, 8])
        
        # Get CN targets from chemistry echo (for labeling, not constraint filtering)
        chem_echo = spec.get('chemistry', {})
        cn_targets = chem_echo.get('cn_targets', {})
        
        # For each parent lattice in the spec
        for parent in spec.get('parent_lattices', []):
            parent_name = parent.get('name', 'SC')
            
            # Map parent to bravais_type
            parent_to_bravais = {
                'SC': 'cubic_P',
                'BCC': 'cubic_I', 
                'FCC': 'cubic_F',
                'TRICLINIC': 'tetragonal_P',
            }
            
            bravais_type = parent_to_bravais.get(parent_name, 'cubic_P')
            lattice_type = BRAVAIS_TO_LATTICE.get(bravais_type, 'Cubic')
            
            # Generate offsets from orbit structure
            M = m_list[0] if m_list else sum(orbit_sizes)
            offsets = _generate_simple_offsets(orbit_sizes, M, bravais_type)
            
            # Build config ID
            cn_str = "_".join([f"{sp}CN{_format_cn(cn_targets.get(sp, '?'))}" 
                              for sp in orbit_species])
            config_id = f"CHEM-{parent_name}-{cn_str}-{i}"
            
            # Get c/a ratio
            c_ratio = DEFAULT_C_RATIOS.get(lattice_type, 1.0)
            if lattice_type == 'Tetragonal':
                c_ratio = float(num_metals)
            
            # Extract CN1 values (handle list or int)
            cn1_values = []
            for sp in orbit_species:
                cn_val = cn_targets.get(sp, 6)
                if isinstance(cn_val, list):
                    cn1_values.append(cn_val[0] if cn_val else 6)
                else:
                    cn1_values.append(cn_val)
            
            configs.append({
                'id': config_id,
                'lattice': lattice_type,
                'bravais_type': bravais_type,
                'offsets': offsets,
                'pattern': label,
                'c_ratio': c_ratio,
                'CN1': sum(cn1_values) // len(cn1_values) if cn1_values else 6,
                'cn_targets': cn_targets,
                'source': 'chemistry_predictor',
                'spec_index': i,
                'orbit_sizes': orbit_sizes,
                'orbit_species': orbit_species,
            })
    
    # If requested, also run full lattice search with GEOMETRY-ONLY constraints
    if run_lattice_search and specs:
        try:
            lattice_configs = _run_lattice_search_geometry(
                specs[:3],
                metals,
                top_k=top_k,
                diagonal_only=diagonal_only,
                verbose=verbose
            )
            # Prepend lattice search results
            configs = lattice_configs + configs
        except Exception as e:
            if verbose:
                print(f"Lattice search failed: {e}")
    
    # Deduplicate by offsets
    seen_offsets = set()
    unique_configs = []
    for c in configs:
        offset_key = tuple(tuple(o) for o in c['offsets'])
        if offset_key not in seen_offsets:
            seen_offsets.add(offset_key)
            unique_configs.append(c)
    
    return unique_configs[:top_k]


def _format_cn(cn_val):
    """Format CN value for display."""
    if isinstance(cn_val, list):
        return str(cn_val[0]) if cn_val else '?'
    return str(cn_val)


def _generate_simple_offsets(
    orbit_sizes: List[int],
    M: int,
    bravais_type: str
) -> List[Tuple[float, float, float]]:
    """Generate simple offset positions for a given orbit structure."""
    N = sum(orbit_sizes)
    
    templates = {
        1: [(0.0, 0.0, 0.0)],
        2: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
        3: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)],
        4: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
        6: [(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), 
            (0.0, 0.5, 0.5), (0.5, 0.5, 0.5), (0.0, 0.0, 0.5)],
        8: [(x/2, y/2, z/2) for x in range(2) for y in range(2) for z in range(2)],
    }
    
    if N in templates:
        return templates[N][:N]
    
    # Fallback: evenly spaced
    offsets = []
    for i in range(N):
        t = i / N
        offsets.append((t, t, t))
    
    return offsets


def _run_lattice_search_geometry(
    specs: List[Dict],
    metals: List[Dict],
    top_k: int = 10,
    diagonal_only: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """
    Run lattice search with GEOMETRY-ONLY constraints.
    
    Uses only: min separation, shell gap, uniformity
    Does NOT use CN constraints (those are for cation-anion, not cation-cation).
    """
    if not _ensure_imports():
        return []
    
    all_configs = []
    
    # Import lattice matrices
    from lattice_search import (
        search_and_analyze, 
        SC_PRIMITIVE, BCC_PRIMITIVE, FCC_PRIMITIVE
    )
    
    parent_lattices = {
        'SC': SC_PRIMITIVE,
        'BCC': BCC_PRIMITIVE,
        'FCC': FCC_PRIMITIVE,
    }
    
    for spec in specs:
        orbit_sizes = spec.get('orbit_sizes', [1])
        orbit_species = spec.get('orbit_species', [m['symbol'] for m in metals])
        m_list = spec.get('M_list', [2, 4, 8])
        
        for parent in spec.get('parent_lattices', []):
            parent_name = parent.get('name', 'SC')
            if parent_name not in parent_lattices:
                continue
                
            lattice_matrix = parent_lattices[parent_name]
            
            try:
                raw_configs = search_and_analyze(
                    lattice=lattice_matrix,
                    a=1.0,
                    orbit_sizes=orbit_sizes,
                    M_list=m_list[:3],  # Limit M values
                    diagonal_only=diagonal_only,
                    max_per_hnf=20,
                    verbose=False,
                    parent_basis_name=f'{parent_name}-primitive'
                )
                
                # Filter by GEOMETRY only
                for cfg in raw_configs:
                    # Geometry constraints
                    if cfg.min_distance < 0.25:
                        continue
                    if cfg.shell_gap < 0.05:
                        continue
                    if not cfg.is_uniform:
                        continue
                    
                    # Extract offsets
                    offsets = []
                    for orbit_idx in range(len(orbit_sizes)):
                        coords = cfg.get_fractional_coords(orbit_idx)
                        if len(coords) > 0:
                            offsets.append(tuple(coords[0].tolist()))
                    
                    if not offsets:
                        continue
                    
                    # Get CN info (for display only)
                    cn1_list = [cfg.signatures[j].cn1 for j in range(len(orbit_sizes))]
                    
                    # Map to bravais type
                    bravais_map = {'SC': 'cubic_P', 'BCC': 'cubic_I', 'FCC': 'cubic_F'}
                    bravais_type = bravais_map.get(parent_name, 'cubic_P')
                    
                    config_id = f"ENUM-{parent_name}-M{len(cfg.G)}-{len(all_configs)}"
                    
                    all_configs.append({
                        'id': config_id,
                        'lattice': 'Cubic',
                        'bravais_type': bravais_type,
                        'offsets': list(offsets),
                        'pattern': f"M={len(cfg.G)} geom_CN={cn1_list}",
                        'c_ratio': 1.0,
                        'CN1': sum(cn1_list) // len(cn1_list) if cn1_list else 6,
                        'source': 'lattice_enumeration',
                        'min_distance': cfg.min_distance,
                        'shell_gap': cfg.shell_gap,
                        'orbit_species': orbit_species,
                    })
                    
            except Exception as e:
                if verbose:
                    print(f"Error searching {parent_name}: {e}")
    
    # Sort by min_distance (larger = better separation)
    all_configs.sort(key=lambda x: -x.get('min_distance', 0))
    
    return all_configs[:top_k]


def _run_lattice_search(
    specs: List[Dict],
    metals: List[Dict],
    top_k: int = 10,
    diagonal_only: bool = True,
    verbose: bool = False
) -> List[Dict]:
    """Alias for geometry-based search."""
    return _run_lattice_search_geometry(specs, metals, top_k, diagonal_only, verbose)


class ChemistryPredictor:
    """Wrapper class for chemistry-based lattice prediction."""
    
    def __init__(self):
        self._available = _ensure_imports()
    
    def is_available(self) -> bool:
        return self._available
    
    def get_search_configs(
        self,
        metals: List[Dict],
        anion_symbol: str = 'O',
        anion_charge: int = 2,
        top_k: int = 15,
        always_include_cubic: bool = True,
        always_include_common: bool = True,
        run_lattice_search: bool = False
    ) -> List[Dict]:
        """Get search configurations from chemistry input."""
        if not self._available:
            return []
        
        configs = get_chemistry_search_configs(
            metals=metals,
            anion_symbol=anion_symbol,
            anion_charge=anion_charge,
            top_k=top_k,
            run_lattice_search=run_lattice_search
        )
        
        if always_include_cubic:
            configs = self._ensure_cubic(configs, len(metals))
        
        if always_include_common:
            configs = self._ensure_common(configs, len(metals))
        
        return configs
    
    def _ensure_cubic(self, configs: List[Dict], num_metals: int) -> List[Dict]:
        """Ensure cubic P/I/F are always included."""
        existing = set(c['bravais_type'] for c in configs)
        
        cubic_templates = {
            'cubic_P': 6,
            'cubic_I': 8,
            'cubic_F': 12,
        }
        
        for bravais, cn1 in cubic_templates.items():
            if bravais not in existing:
                offsets = _generate_simple_offsets([1]*num_metals, num_metals, bravais)
                configs.append({
                    'id': f'CUBIC-{bravais}-N{num_metals}',
                    'lattice': 'Cubic',
                    'bravais_type': bravais,
                    'offsets': offsets,
                    'pattern': 'cubic (always)',
                    'c_ratio': 1.0,
                    'CN1': cn1,
                    'source': 'always_cubic',
                })
        
        return configs
    
    def _ensure_common(self, configs: List[Dict], num_metals: int) -> List[Dict]:
        """Ensure common lattice types are included."""
        existing = set(c['bravais_type'] for c in configs)
        
        common_types = [
            ('tetragonal_P', 'Tetragonal', float(num_metals)),
            ('tetragonal_I', 'Tetragonal', float(num_metals)),
            ('hexagonal_P', 'Hexagonal', 1.633),
            ('hexagonal_H', 'Hexagonal', 1.633),
        ]
        
        for bravais, lattice, c_ratio in common_types:
            if bravais not in existing:
                offsets = _generate_simple_offsets([1]*num_metals, num_metals, bravais)
                configs.append({
                    'id': f'COMMON-{bravais}-N{num_metals}',
                    'lattice': lattice,
                    'bravais_type': bravais,
                    'offsets': offsets,
                    'pattern': f'{lattice.lower()} (always)',
                    'c_ratio': c_ratio,
                    'CN1': 6,
                    'source': 'always_common',
                })
        
        return configs


# =============================================================================
# DROP-IN REPLACEMENT FOR get_default_search_configs
# =============================================================================

def get_default_search_configs_chemistry(
    num_metals: int,
    metals: List[Dict] = None,
    anion_symbol: str = 'O',
    anion_charge: int = 2,
    use_predictor: bool = True,
    target_cn: int = None,
    run_lattice_search: bool = False
) -> List[Dict]:
    """
    Drop-in replacement for get_default_search_configs that uses chemistry-based prediction.
    
    This function can be called in two ways:
    1. With full chemistry (metals, anion_symbol, anion_charge) - uses new predictor
    2. With just num_metals (backwards compatible) - falls back to old behavior
    
    Args:
        num_metals: Number of metal types (N)
        metals: List of metal dicts with symbol, charge, ratio, cn, radius
        anion_symbol: Anion symbol (e.g., 'O', 'F')
        anion_charge: Anion charge magnitude (positive)
        use_predictor: Whether to use the chemistry predictor
        target_cn: Target coordination number (for filtering)
        run_lattice_search: Run full HNF enumeration (slower but more accurate)
    
    Returns:
        List of config dicts compatible with the app
    """
    # If we have full chemistry info, use the new predictor
    if metals is not None and use_predictor:
        predictor = get_chemistry_predictor()
        
        if predictor.is_available():
            configs = predictor.get_search_configs(
                metals=metals,
                anion_symbol=anion_symbol,
                anion_charge=anion_charge,
                top_k=20,
                always_include_cubic=True,
                always_include_common=True,
                run_lattice_search=run_lattice_search
            )
            
            if configs:
                return configs
    
    # Fallback: generate basic configs for N metals
    return _generate_fallback_configs(num_metals, target_cn)


def _generate_fallback_configs(num_metals: int, target_cn: int = None) -> List[Dict]:
    """Generate basic fallback configs when chemistry predictor unavailable."""
    configs = []
    
    # Basic templates by N
    offset_templates = {
        1: [[(0.0, 0.0, 0.0)]],
        2: [[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5)]],
        3: [[(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)],
            [(0.0, 0.0, 0.0), (1/3, 2/3, 0.5), (2/3, 1/3, 0.5)]],
        4: [[(0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
            [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]],
    }
    
    lattice_types = [
        ('cubic_P', 'Cubic', 1.0, 6),
        ('cubic_I', 'Cubic', 1.0, 8),
        ('cubic_F', 'Cubic', 1.0, 12),
        ('tetragonal_P', 'Tetragonal', float(num_metals), 6),
        ('hexagonal_P', 'Hexagonal', 1.633, 6),
    ]
    
    templates = offset_templates.get(num_metals, [[(i/num_metals, i/num_metals, i/num_metals) 
                                                    for i in range(num_metals)]])
    
    for bravais, lattice, c_ratio, cn1 in lattice_types:
        for j, offsets in enumerate(templates):
            config_id = f'FALLBACK-{bravais}-N{num_metals}-{j}'
            configs.append({
                'id': config_id,
                'lattice': lattice,
                'bravais_type': bravais,
                'offsets': list(offsets),
                'pattern': f'{lattice} template',
                'c_ratio': c_ratio,
                'CN1': cn1,
                'source': 'fallback',
            })
    
    return configs


# Global instance
_predictor = None


def get_chemistry_predictor() -> ChemistryPredictor:
    """Get or create the global chemistry predictor."""
    global _predictor
    if _predictor is None:
        _predictor = ChemistryPredictor()
    return _predictor


if __name__ == '__main__':
    metals = [
        {'symbol': 'Sr', 'charge': 2, 'ratio': 1, 'cn': 12, 'radius': 1.44},
        {'symbol': 'Ti', 'charge': 4, 'ratio': 1, 'cn': 6, 'radius': 0.605}
    ]
    
    predictor = get_chemistry_predictor()
    print(f"Predictor available: {predictor.is_available()}")
    
    if predictor.is_available():
        configs = predictor.get_search_configs(
            metals=metals,
            anion_symbol='O',
            anion_charge=2,
            top_k=10
        )
        
        print(f"\nGenerated {len(configs)} configs:")
        for c in configs[:5]:
            print(f"  {c['id']}: {c['bravais_type']} - {c['pattern']}")
