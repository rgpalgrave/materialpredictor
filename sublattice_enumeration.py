"""
Equivalent N-Sublattice Enumeration Module
==========================================

Enumerates configurations of N interpenetrating Bravais sublattices where
all sites have identical coordination topology (same CN sequence).

For use upstream of sphere intersection calculations to identify candidate
cation lattice arrangements.

Usage:
------
    from sublattice_enumeration import SublatticeFinder
    
    # Initialize (loads pre-computed database)
    finder = SublatticeFinder('equivalent_configs_full.pkl')
    
    # Query for cubic lattices (instant lookup)
    configs = finder.query(N=4, lattice_type='cubic_F', cn1_min=6)
    
    # Query with c/a scan for tetragonal/hexagonal
    configs = finder.query(N=6, lattice_type='hexagonal_P', 
                           ca_scan=True, cn1_min=4)
    
    # Get offsets for sphere intersection calculator
    for cfg in configs:
        offsets = cfg['offsets']  # List of N fractional coordinates
        # ... run intersection calculator

Theory:
-------
Two methods ensure completeness:
1. HNF cosets: All index-N sublattices of Z³ (works for ANY crystal system)
2. Grid search: Additional non-coset solutions in high-symmetry (cubic)

The topology (CN sequence) depends on lattice parameters (c/a, angles).
For cubic: topology is fixed, pre-computed in database.
For others: topology computed on-the-fly or via parameter scanning.

Author: Claude (Anthropic) with Rob
Date: December 2024
"""

import numpy as np
from itertools import combinations
from collections import defaultdict
import pickle
from pathlib import Path


# =============================================================================
# Lattice Vector Definitions
# =============================================================================

def get_lattice_vectors(lattice_type, c_a=1.0):
    """
    Get lattice vectors for Bravais lattice types.
    
    Args:
        lattice_type: One of 'cubic_P', 'cubic_F', 'cubic_I', 
                      'tetragonal_P', 'tetragonal_I', 'hexagonal_P',
                      'orthorhombic_P', 'rhombohedral'
        c_a: c/a ratio (for tetragonal/hexagonal)
    
    Returns:
        (vectors, basis) where:
            vectors: 3x3 array, rows are lattice vectors
            basis: list of basis positions (fractional coords)
    """
    if lattice_type == 'cubic_P':
        vectors = np.eye(3)
        basis = [np.array([0, 0, 0])]
        
    elif lattice_type == 'cubic_F':  # FCC
        vectors = np.eye(3)
        basis = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0]),
                 np.array([0.5, 0, 0.5]), np.array([0, 0.5, 0.5])]
        
    elif lattice_type == 'cubic_I':  # BCC
        vectors = np.eye(3)
        basis = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        
    elif lattice_type == 'tetragonal_P':
        vectors = np.diag([1, 1, c_a])
        basis = [np.array([0, 0, 0])]
        
    elif lattice_type == 'tetragonal_I':
        vectors = np.diag([1, 1, c_a])
        basis = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
        
    elif lattice_type == 'hexagonal_P':
        a = np.array([1, 0, 0])
        b = np.array([-0.5, np.sqrt(3)/2, 0])
        c = np.array([0, 0, c_a])
        vectors = np.vstack([a, b, c])
        basis = [np.array([0, 0, 0])]
        
    elif lattice_type == 'orthorhombic_P':
        vectors = np.diag([1.0, 1.2, 1.5])
        basis = [np.array([0, 0, 0])]
        
    elif lattice_type == 'rhombohedral':
        # Use c_a as the angle in degrees for rhombohedral
        alpha = np.radians(c_a) if c_a > 10 else np.radians(90)
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        a = np.array([1, 0, 0])
        b = np.array([cos_a, sin_a, 0])
        cx, cy = cos_a, (cos_a - cos_a**2) / sin_a if sin_a > 1e-6 else 0
        cz = np.sqrt(max(0, 1 - cx**2 - cy**2))
        c = np.array([cx, cy, cz])
        vectors = np.vstack([a, b, c])
        basis = [np.array([0, 0, 0])]
        
    else:
        raise ValueError(f"Unknown lattice type: {lattice_type}")
    
    return vectors, basis


# =============================================================================
# HNF Enumeration (Core Algorithm)
# =============================================================================

def generate_hnf_matrices(N):
    """
    Generate all 3x3 upper-triangular Hermite Normal Form matrices with det=N.
    These define all index-N sublattices of Z³.
    """
    matrices = []
    for h11 in range(1, N + 1):
        if N % h11 != 0:
            continue
        remainder = N // h11
        for h22 in range(1, remainder + 1):
            if remainder % h22 != 0:
                continue
            h33 = remainder // h22
            for h12 in range(h22):
                for h13 in range(h33):
                    for h23 in range(h33):
                        H = np.array([[h11, h12, h13],
                                      [0, h22, h23],
                                      [0, 0, h33]], dtype=float)
                        matrices.append(H)
    return matrices


def get_coset_offsets(H):
    """Get N coset representatives from HNF matrix H."""
    h11, h22, h33 = int(H[0,0]), int(H[1,1]), int(H[2,2])
    H_inv = np.linalg.inv(H)
    offsets = []
    for k1 in range(h11):
        for k2 in range(h22):
            for k3 in range(h33):
                t = (H_inv @ np.array([k1, k2, k3])) % 1
                t = np.round(t, 10) % 1
                offsets.append(tuple(t))
    return offsets


# =============================================================================
# Topology Computation
# =============================================================================

def compute_topology(offsets, lattice_type='cubic_P', c_a=1.0, 
                     n_cells=2, n_shells=6):
    """
    Compute coordination topology for a configuration.
    
    Args:
        offsets: List of N fractional coordinate tuples
        lattice_type: Bravais lattice type
        c_a: c/a ratio (or angle for rhombohedral)
        n_cells: Number of unit cells to include
        n_shells: Number of coordination shells to compute
    
    Returns:
        List of (distance, CN) tuples for first n_shells
    """
    vectors, basis = get_lattice_vectors(lattice_type, c_a)
    return compute_topology_for_vectors(offsets, vectors, basis, n_cells, n_shells)


def compute_topology_for_vectors(offsets, vectors, basis=None, 
                                  n_cells=2, n_shells=6):
    """
    Compute topology given explicit lattice vectors.
    
    Args:
        offsets: List of fractional coordinate tuples
        vectors: 3x3 array of lattice vectors (rows)
        basis: Optional basis positions (default: origin only)
        n_cells: Number of unit cells
        n_shells: Number of shells to compute
    
    Returns:
        List of (distance, CN) tuples
    """
    if basis is None:
        basis = [np.array([0, 0, 0])]
    
    offsets = [np.array(o) for o in offsets]
    vectors = np.array(vectors)
    
    # Generate lattice points
    points = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                for b in basis:
                    for off in offsets:
                        frac = np.array([i, j, k]) + b + off
                        cart = frac @ vectors
                        points.append(cart)
    points = np.array(points)
    
    # Reference point
    ref = offsets[0] @ vectors
    
    # Compute distances
    distances = np.linalg.norm(points - ref, axis=1)
    distances = distances[distances > 1e-8]
    distances = np.sort(distances)
    
    # Group into shells
    shells = []
    if len(distances) == 0:
        return shells
    
    current_d = distances[0]
    count = 1
    
    for d in distances[1:]:
        if abs(d - current_d) < 1e-5:
            count += 1
        else:
            shells.append((round(current_d, 6), count))
            if len(shells) >= n_shells:
                break
            current_d = d
            count = 1
    
    if len(shells) < n_shells:
        shells.append((round(current_d, 6), count))
    
    return shells[:n_shells]


# =============================================================================
# Parameter Scanning
# =============================================================================

def scan_ca_ratio(offsets, lattice_type, ca_range=None, cn1_min=None):
    """
    Scan c/a ratio to find optimal topology.
    
    Args:
        offsets: Fractional coordinate tuples
        lattice_type: Base lattice type
        ca_range: Array of c/a values (default: 0.5 to 3.0)
        cn1_min: Minimum CN1 filter
    
    Returns:
        List of dicts: {c_a, topology, CN1, CN_sequence}
    """
    if ca_range is None:
        ca_range = np.linspace(0.5, 3.0, 60)
    
    results = []
    seen = set()
    
    for c_a in ca_range:
        topo = compute_topology(offsets, lattice_type, c_a)
        if not topo:
            continue
        
        cn1 = topo[0][1]
        cn_key = tuple(t[1] for t in topo[:4])
        
        if cn_key in seen:
            continue
        seen.add(cn_key)
        
        if cn1_min is not None and cn1 < cn1_min:
            continue
        
        results.append({
            'c_a': round(c_a, 4),
            'topology': topo,
            'CN1': cn1,
            'CN_sequence': list(cn_key)
        })
    
    results.sort(key=lambda x: -x['CN1'])
    return results


def find_optimal_ca(offsets, lattice_type, ca_range=None):
    """
    Find c/a ratio that maximizes CN1.
    
    Returns:
        Dict with optimal c_a, topology, CN1 (or None)
    """
    if ca_range is None:
        ca_range = np.linspace(0.5, 3.0, 100)
    
    best = None
    best_cn1 = 0
    
    for c_a in ca_range:
        topo = compute_topology(offsets, lattice_type, c_a)
        if not topo:
            continue
        
        cn1 = topo[0][1]
        if cn1 > best_cn1:
            best_cn1 = cn1
            best = {
                'c_a': round(c_a, 4),
                'topology': topo,
                'CN1': cn1
            }
    
    return best


# =============================================================================
# Main Interface Class
# =============================================================================

class SublatticeFinder:
    """
    Main interface for finding equivalent N-sublattice configurations.
    
    Usage:
        finder = SublatticeFinder('equivalent_configs_full.pkl')
        configs = finder.query(N=4, lattice_type='cubic_F', cn1_min=6)
    """
    
    # Lattices where topology is fixed (no c/a dependence)
    CUBIC_LATTICES = {'cubic_P', 'cubic_F', 'cubic_I'}
    
    # Lattices requiring c/a parameter
    VARIABLE_LATTICES = {'tetragonal_P', 'tetragonal_I', 'hexagonal_P', 
                         'orthorhombic_P', 'rhombohedral'}
    
    def __init__(self, database_path=None):
        """
        Initialize finder with pre-computed database.
        
        Args:
            database_path: Path to pickle file (optional)
        """
        self.database = {}
        self._hnf_cache = {}
        
        if database_path and Path(database_path).exists():
            self.load_database(database_path)
    
    def load_database(self, path):
        """Load pre-computed database from pickle file."""
        with open(path, 'rb') as f:
            self.database = pickle.load(f)
    
    def save_database(self, path):
        """Save database to pickle file."""
        with open(path, 'wb') as f:
            pickle.dump(self.database, f)
    
    def get_hnf_offsets(self, N):
        """Get all HNF coset offsets for given N (cached)."""
        if N not in self._hnf_cache:
            hnfs = generate_hnf_matrices(N)
            self._hnf_cache[N] = [get_coset_offsets(H) for H in hnfs]
        return self._hnf_cache[N]
    
    def query(self, N, lattice_type=None, c_a=1.0, 
              cn1_min=None, cn1_max=None, cn1=None, cn2=None, cn3=None,
              cn_sequence=None, ca_scan=False, ca_range=None):
        """
        Query for equivalent N-sublattice configurations.
        
        Args:
            N: Number of sublattices (2-10 typical)
            lattice_type: Specific lattice or None for all cubic
            c_a: c/a ratio for non-cubic lattices
            cn1_min, cn1_max: CN1 range filter
            cn1, cn2, cn3: Exact CN values for specific shells
            cn_sequence: Exact CN sequence to match
            ca_scan: If True, scan c/a to find best values
            ca_range: Custom c/a range for scanning
        
        Returns:
            List of config dicts with keys:
                - offsets: List of N fractional coordinates
                - topology: List of (distance, CN) tuples  
                - CN1: First shell coordination number
                - c_a: c/a ratio (for non-cubic)
                - lattice_type: Lattice type (if querying multiple)
        """
        # Default to cubic lattices
        if lattice_type is None:
            lattice_types = list(self.CUBIC_LATTICES)
        else:
            lattice_types = [lattice_type]
        
        results = []
        
        for lat in lattice_types:
            if lat in self.CUBIC_LATTICES:
                # Use pre-computed database for cubic
                configs = self._query_database(N, lat, cn1_min, cn1_max, 
                                                cn1, cn2, cn3, cn_sequence)
                for cfg in configs:
                    cfg['lattice_type'] = lat
                    cfg['c_a'] = 1.0
                results.extend(configs)
                
            elif ca_scan:
                # Scan c/a range for non-cubic
                configs = self._query_with_ca_scan(N, lat, ca_range,
                                                    cn1_min, cn1_max)
                for cfg in configs:
                    cfg['lattice_type'] = lat
                results.extend(configs)
                
            else:
                # Compute at specific c/a
                configs = self._query_at_ca(N, lat, c_a, cn1_min, cn1_max,
                                            cn1, cn2, cn3, cn_sequence)
                for cfg in configs:
                    cfg['lattice_type'] = lat
                    cfg['c_a'] = c_a
                results.extend(configs)
        
        # Sort by CN1 descending
        results.sort(key=lambda x: -x['CN1'])
        return results
    
    def _query_database(self, N, lattice_type, cn1_min, cn1_max,
                        cn1, cn2, cn3, cn_sequence):
        """Query pre-computed database with filters."""
        if lattice_type not in self.database:
            return []
        if N not in self.database[lattice_type]:
            return []
        
        configs = self.database[lattice_type][N]
        return self._filter_configs(configs, cn1_min, cn1_max,
                                    cn1, cn2, cn3, cn_sequence)
    
    def _query_at_ca(self, N, lattice_type, c_a, cn1_min, cn1_max,
                     cn1, cn2, cn3, cn_sequence):
        """Compute topology at specific c/a."""
        all_offsets = self.get_hnf_offsets(N)
        results = []
        seen = set()
        
        for offsets in all_offsets:
            topo = compute_topology(offsets, lattice_type, c_a)
            if not topo:
                continue
            
            cn_key = tuple(t[1] for t in topo[:4])
            if cn_key in seen:
                continue
            seen.add(cn_key)
            
            cfg = {
                'offsets': offsets,
                'topology': topo,
                'CN1': topo[0][1],
                'is_coset': True
            }
            results.append(cfg)
        
        return self._filter_configs(results, cn1_min, cn1_max,
                                    cn1, cn2, cn3, cn_sequence)
    
    def _query_with_ca_scan(self, N, lattice_type, ca_range, cn1_min, cn1_max):
        """Scan c/a range to find best topology for each config."""
        if ca_range is None:
            ca_range = np.linspace(0.5, 3.0, 50)
        
        all_offsets = self.get_hnf_offsets(N)
        results = []
        seen = set()
        
        for offsets in all_offsets:
            best = find_optimal_ca(offsets, lattice_type, ca_range)
            if best is None:
                continue
            
            cn_key = tuple(t[1] for t in best['topology'][:4])
            if cn_key in seen:
                continue
            seen.add(cn_key)
            
            # Apply CN1 filter
            if cn1_min is not None and best['CN1'] < cn1_min:
                continue
            if cn1_max is not None and best['CN1'] > cn1_max:
                continue
            
            results.append({
                'offsets': offsets,
                'topology': best['topology'],
                'CN1': best['CN1'],
                'c_a': best['c_a'],
                'is_coset': True
            })
        
        return results
    
    def _filter_configs(self, configs, cn1_min, cn1_max,
                        cn1, cn2, cn3, cn_sequence):
        """Apply CN filters to config list."""
        result = []
        
        for cfg in configs:
            cn_seq = [t[1] for t in cfg['topology']]
            
            if cn1_min is not None and cfg['CN1'] < cn1_min:
                continue
            if cn1_max is not None and cfg['CN1'] > cn1_max:
                continue
            if cn_sequence is not None:
                if cn_seq[:len(cn_sequence)] != list(cn_sequence):
                    continue
            if cn1 is not None and (len(cn_seq) < 1 or cn_seq[0] != cn1):
                continue
            if cn2 is not None and (len(cn_seq) < 2 or cn_seq[1] != cn2):
                continue
            if cn3 is not None and (len(cn_seq) < 3 or cn_seq[2] != cn3):
                continue
            
            result.append(cfg)
        
        return result
    
    def scan_parameter(self, offsets, lattice_type, param_range=None, 
                       param_type='c_a'):
        """
        Scan a lattice parameter for a specific offset configuration.
        
        Args:
            offsets: Fractional coordinates
            lattice_type: Lattice type
            param_range: Array of parameter values
            param_type: 'c_a' or 'angle' (for rhombohedral)
        
        Returns:
            List of (param_value, CN1, CN_sequence) tuples
        """
        if param_range is None:
            if param_type == 'angle':
                param_range = np.linspace(50, 110, 50)
            else:
                param_range = np.linspace(0.5, 3.0, 50)
        
        results = []
        for val in param_range:
            if param_type == 'angle' and lattice_type == 'rhombohedral':
                topo = compute_topology(offsets, 'rhombohedral', c_a=val)
            else:
                topo = compute_topology(offsets, lattice_type, c_a=val)
            
            if topo:
                cn_seq = [t[1] for t in topo[:4]]
                results.append((val, topo[0][1], cn_seq))
        
        return results
    
    def get_statistics(self, N=None):
        """Get database statistics."""
        stats = {}
        for lat in self.database:
            stats[lat] = {}
            for n in self.database[lat]:
                if N is not None and n != N:
                    continue
                configs = self.database[lat][n]
                cn1_values = [c['CN1'] for c in configs]
                stats[lat][n] = {
                    'count': len(configs),
                    'cn1_max': max(cn1_values) if cn1_values else 0,
                    'cn1_min': min(cn1_values) if cn1_values else 0
                }
        return stats
    
    def print_summary(self):
        """Print database summary."""
        print("Sublattice Database Summary")
        print("=" * 60)
        
        header = "Lattice".ljust(14)
        for N in range(2, 11):
            header += f"N={N}".ljust(5)
        print(header)
        print("-" * 60)
        
        for lat in sorted(self.database.keys()):
            row = lat.ljust(14)
            for N in range(2, 11):
                count = len(self.database[lat].get(N, []))
                row += f"{count}".ljust(5)
            print(row)


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_query(N, lattice_type='cubic_F', cn1_min=None, database_path=None):
    """
    Quick query without explicit initialization.
    
    Args:
        N: Number of sublattices
        lattice_type: Lattice type
        cn1_min: Minimum CN1 filter
        database_path: Path to database (optional)
    
    Returns:
        List of (offsets, CN1, topology) tuples
    """
    finder = SublatticeFinder(database_path)
    configs = finder.query(N=N, lattice_type=lattice_type, cn1_min=cn1_min)
    
    return [(c['offsets'], c['CN1'], c['topology']) for c in configs]


def enumerate_offsets(N, lattice_type='cubic_P', c_a=1.0, cn1_min=None):
    """
    Enumerate all equivalent N-sublattice offsets (no database required).
    
    Args:
        N: Number of sublattices
        lattice_type: Lattice type
        c_a: c/a ratio
        cn1_min: Minimum CN1 filter
    
    Returns:
        List of offset tuples
    """
    all_offsets = [get_coset_offsets(H) for H in generate_hnf_matrices(N)]
    
    if cn1_min is None:
        return all_offsets
    
    # Filter by CN1
    filtered = []
    for offsets in all_offsets:
        topo = compute_topology(offsets, lattice_type, c_a)
        if topo and topo[0][1] >= cn1_min:
            filtered.append(offsets)
    
    return filtered


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SUBLATTICE ENUMERATION MODULE - DEMO")
    print("=" * 70)
    
    # Try to load database
    db_path = 'equivalent_configs_full.pkl'
    finder = SublatticeFinder(db_path if Path(db_path).exists() else None)
    
    if finder.database:
        print("\nLoaded pre-computed database:")
        finder.print_summary()
    
    # Demo queries
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES")
    print("=" * 70)
    
    # Query 1: Cubic FCC with high CN1
    print("\n1. N=4 FCC with CN1 >= 6:")
    configs = finder.query(N=4, lattice_type='cubic_F', cn1_min=6)
    for cfg in configs[:3]:
        cn_seq = [t[1] for t in cfg['topology'][:4]]
        print(f"   CN1={cfg['CN1']}, CN={cn_seq}")
    
    # Query 2: Hexagonal with c/a scan
    print("\n2. N=6 Hexagonal with c/a scan (CN1 >= 4):")
    configs = finder.query(N=6, lattice_type='hexagonal_P', 
                           ca_scan=True, cn1_min=4)
    for cfg in configs[:3]:
        cn_seq = [t[1] for t in cfg['topology'][:4]]
        print(f"   c/a={cfg['c_a']:.2f}, CN1={cfg['CN1']}, CN={cn_seq}")
    
    # Query 3: Direct enumeration (no database)
    print("\n3. Direct enumeration (no database needed):")
    offsets_list = enumerate_offsets(N=3, lattice_type='cubic_P', cn1_min=6)
    print(f"   Found {len(offsets_list)} N=3 configs with CN1 >= 6")
    
    print("\n" + "=" * 70)
    print("INTEGRATION EXAMPLE")
    print("=" * 70)
    print("""
# In your sphere intersection module:

from sublattice_enumeration import SublatticeFinder

# Initialize once
finder = SublatticeFinder('equivalent_configs_full.pkl')

# For each user query:
def find_structures(N, lattice_type, cn1_target):
    configs = finder.query(N=N, lattice_type=lattice_type, cn1_min=cn1_target)
    
    results = []
    for cfg in configs:
        offsets = cfg['offsets']      # Feed to intersection calculator
        cn_seq = cfg['topology']      # Already know the topology
        
        # Run your intersection calculator here
        # intersection_result = sphere_intersection(offsets, ...)
        
        results.append({
            'offsets': offsets,
            'cn_sequence': [t[1] for t in cn_seq],
            # 'anion_positions': intersection_result
        })
    
    return results
""")
