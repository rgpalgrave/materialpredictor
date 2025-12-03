"""
Interstitial Engine for Sphere Intersection Analysis
Finds minimum scale factors to achieve N-fold sphere intersections
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from functools import lru_cache
from scipy.spatial import KDTree


@dataclass(frozen=True)
class LatticeParams:
    """Lattice parameters for a crystal structure."""
    a: float = 5.0
    b_ratio: float = 1.0
    c_ratio: float = 1.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    
    @property
    def b(self) -> float:
        return self.a * self.b_ratio
    
    @property
    def c(self) -> float:
        return self.a * self.c_ratio


# Bravais lattice basis points (fractional coordinates)
# These are the centering translations that define each Bravais type
BRAVAIS_BASIS = {
    # Cubic
    'cubic_P': [(0, 0, 0)],
    'cubic_I': [(0, 0, 0), (0.5, 0.5, 0.5)],  # BCC
    'cubic_F': [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],  # FCC
    
    # Tetragonal
    'tetragonal_P': [(0, 0, 0)],
    'tetragonal_I': [(0, 0, 0), (0.5, 0.5, 0.5)],  # Body-centered tetragonal
    
    # Orthorhombic
    'orthorhombic_P': [(0, 0, 0)],
    'orthorhombic_I': [(0, 0, 0), (0.5, 0.5, 0.5)],  # Body-centered
    'orthorhombic_F': [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],  # Face-centered
    'orthorhombic_C': [(0, 0, 0), (0.5, 0.5, 0)],  # C-centered (base-centered)
    'orthorhombic_A': [(0, 0, 0), (0, 0.5, 0.5)],  # A-centered
    'orthorhombic_B': [(0, 0, 0), (0.5, 0, 0.5)],  # B-centered
    
    # Hexagonal
    'hexagonal_P': [(0, 0, 0)],
    'hexagonal_H': [(0, 0, 0), (2/3, 1/3, 0.5)],  # HCP - note: requires c/a ~ 1.633 for ideal
    
    # Rhombohedral
    'rhombohedral_P': [(0, 0, 0)],
    
    # Monoclinic
    'monoclinic_P': [(0, 0, 0)],
    'monoclinic_C': [(0, 0, 0), (0.5, 0.5, 0)],  # C-centered
    
    # Triclinic
    'triclinic_P': [(0, 0, 0)],
}


@dataclass(frozen=True)
class Sublattice:
    """A sublattice with positions and sphere parameters."""
    name: str
    offsets: Tuple[Tuple[float, float, float], ...]  # Fractional coordinates
    alpha_ratio: float | Tuple[float, ...] = 0.5  # Sphere radius = alpha_ratio * scale * a (single or per-offset)
    bravais_type: str = 'cubic_P'  # Bravais lattice type for centering
    
    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.offsets, list):
            object.__setattr__(self, 'offsets', tuple(tuple(o) for o in self.offsets))
        # Convert alpha_ratio list to tuple if needed
        if isinstance(self.alpha_ratio, list):
            object.__setattr__(self, 'alpha_ratio', tuple(self.alpha_ratio))
    
    def get_alpha_for_offset(self, offset_idx: int) -> float:
        """Get alpha ratio for a specific offset index."""
        if isinstance(self.alpha_ratio, (int, float)):
            return float(self.alpha_ratio)
        else:
            # It's a tuple - return corresponding value or last value if index out of range
            if offset_idx < len(self.alpha_ratio):
                return float(self.alpha_ratio[offset_idx])
            return float(self.alpha_ratio[-1])
    
    def get_all_positions(self) -> List[Tuple[float, float, float]]:
        """Get all atomic positions including Bravais centering."""
        basis = BRAVAIS_BASIS.get(self.bravais_type, [(0, 0, 0)])
        positions = []
        for offset in self.offsets:
            for b in basis:
                # Add basis translation to offset, wrap to [0, 1)
                pos = tuple((offset[i] + b[i]) % 1.0 for i in range(3))
                # Avoid duplicates (within tolerance)
                is_dup = False
                for existing in positions:
                    if all(abs((pos[i] - existing[i] + 0.5) % 1.0 - 0.5) < 1e-6 for i in range(3)):
                        is_dup = True
                        break
                if not is_dup:
                    positions.append(pos)
        return positions
    
    def get_all_positions_with_alpha(self) -> List[Tuple[Tuple[float, float, float], float]]:
        """Get all atomic positions with their alpha ratios, including Bravais centering."""
        basis = BRAVAIS_BASIS.get(self.bravais_type, [(0, 0, 0)])
        positions = []
        for offset_idx, offset in enumerate(self.offsets):
            alpha = self.get_alpha_for_offset(offset_idx)
            for b in basis:
                # Add basis translation to offset, wrap to [0, 1)
                pos = tuple((offset[i] + b[i]) % 1.0 for i in range(3))
                # Avoid duplicates (within tolerance)
                is_dup = False
                for existing_pos, _ in positions:
                    if all(abs((pos[i] - existing_pos[i] + 0.5) % 1.0 - 0.5) < 1e-6 for i in range(3)):
                        is_dup = True
                        break
                if not is_dup:
                    positions.append((pos, alpha))
        return positions


def lattice_vectors(p: LatticeParams) -> np.ndarray:
    """Generate 3x3 matrix of lattice vectors (row-wise)."""
    alpha_r = np.radians(p.alpha)
    beta_r = np.radians(p.beta)
    gamma_r = np.radians(p.gamma)
    
    a_vec = np.array([p.a, 0, 0])
    b_vec = np.array([p.b * np.cos(gamma_r), p.b * np.sin(gamma_r), 0])
    
    c_x = p.c * np.cos(beta_r)
    c_y = p.c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / np.sin(gamma_r)
    c_z = np.sqrt(max(0, p.c**2 - c_x**2 - c_y**2))
    c_vec = np.array([c_x, c_y, c_z])
    
    return np.array([a_vec, b_vec, c_vec])


def frac_to_cart(frac: np.ndarray, lat_vecs: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian coordinates."""
    return frac @ lat_vecs


def cart_to_frac(cart: np.ndarray, lat_vecs: np.ndarray) -> np.ndarray:
    """Convert Cartesian to fractional coordinates."""
    inv = np.linalg.inv(lat_vecs)
    return cart @ inv


def generate_shifts(lat_vecs: np.ndarray) -> np.ndarray:
    """Generate 27 shift vectors for periodic boundary conditions."""
    shifts = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                shifts.append(np.array([i, j, k]) @ lat_vecs)
    return np.array(shifts)


def pair_circle_samples(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, 
                        k: int = 12) -> np.ndarray:
    """
    Sample points on the intersection circle of two spheres.
    Returns empty array if spheres don't intersect or one contains the other.
    """
    d = np.linalg.norm(c2 - c1)
    if d < 1e-10:
        return np.empty((0, 3))
    if d > r1 + r2:  # Too far apart
        return np.empty((0, 3))
    if d < abs(r1 - r2):  # One inside the other
        return np.empty((0, 3))
    
    # Distance from c1 to the plane of intersection
    h = (d**2 + r1**2 - r2**2) / (2 * d)
    
    # Radius of intersection circle
    r_circle_sq = r1**2 - h**2
    if r_circle_sq < 0:
        return np.empty((0, 3))
    r_circle = np.sqrt(r_circle_sq)
    
    # Center of intersection circle
    axis = (c2 - c1) / d
    center = c1 + h * axis
    
    # Find two perpendicular vectors to the axis
    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, np.array([1, 0, 0]))
    else:
        perp1 = np.cross(axis, np.array([0, 1, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(axis, perp1)
    
    # Sample points
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    samples = np.zeros((k, 3))
    for i, theta in enumerate(angles):
        samples[i] = center + r_circle * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    
    return samples


def build_centers_and_radii(sublattices: List[Sublattice], p: LatticeParams, 
                            scale_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build arrays of sphere centers and radii from sublattices.
    
    The alpha value is interpreted as the coordination radius in Å (typically r_metal + r_anion).
    The actual sphere radius is: scale_s × coord_radius
    """
    lat_vecs = lattice_vectors(p)
    centers = []
    radii = []
    
    for sub in sublattices:
        # Get all positions including Bravais centering, with per-offset coord_radius
        all_positions_with_alpha = sub.get_all_positions_with_alpha()
        for pos, coord_radius in all_positions_with_alpha:
            cart = frac_to_cart(np.array(pos), lat_vecs)
            centers.append(cart)
            # coord_radius is in Å (r_metal + r_anion), scale_s is the common scale factor
            radii.append(coord_radius * scale_s)
    
    return np.array(centers), np.array(radii)


def periodic_candidate_pairs(centers: np.ndarray, shifts: np.ndarray, 
                             cutoff: float) -> List[Tuple[int, int, int]]:
    """
    Find all pairs of spheres that could intersect within cutoff distance.
    Returns list of (i, j, shift_idx) tuples.
    Includes self-interactions with periodic images (i with i in neighboring cells).
    """
    n = len(centers)
    pairs = []
    zero_shift_idx = 13  # Index of (0,0,0) shift in 3x3x3 grid
    
    for s_idx, shift in enumerate(shifts):
        for i in range(n):
            for j in range(n):
                # In central cell: only i < j to avoid duplicates
                if s_idx == zero_shift_idx:
                    if i >= j:
                        continue
                # In neighboring cells: include all pairs including i == j (periodic self)
                c1 = centers[i]
                c2 = centers[j] + shift
                d = np.linalg.norm(c2 - c1)
                if d < cutoff and d > 1e-10:
                    pairs.append((i, j, s_idx))
    
    return pairs


def max_multiplicity_for_scale(sublattices: List[Sublattice], p: LatticeParams,
                               scale_s: float, k_samples: int = 16,
                               tol_inside: float = 1e-3,
                               early_stop_at: Optional[int] = None) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find maximum intersection multiplicity for a given scale factor.
    
    Returns:
        max_mult: Maximum multiplicity found
        positions: Array of sample positions with high multiplicity
        multiplicities: Multiplicity at each position
    """
    lat_vecs = lattice_vectors(p)
    centers, radii = build_centers_and_radii(sublattices, p, scale_s)
    
    if len(centers) == 0:
        return 0, np.empty((0, 3)), np.empty(0)
    
    rmax = float(np.max(radii))
    shifts = generate_shifts(lat_vecs)
    
    # Find candidate pairs
    pairs = periodic_candidate_pairs(centers, shifts, cutoff=2.0 * rmax)
    if not pairs:
        return 0, np.empty((0, 3)), np.empty(0)
    
    # Build KDTree for fast neighbor lookup
    tree = KDTree(centers.copy())
    
    samples = []
    counts = []
    
    for (i, j, s_idx) in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = radii[i]
        r2 = radii[j]
        
        pts = pair_circle_samples(c1, r1, c2, r2, k=k_samples)
        if pts.size == 0:
            continue
        
        # Count multiplicity for each sample point
        total = np.zeros(len(pts), dtype=int)
        for shift in shifts:
            P = pts - shift
            # Query nearby spheres
            idxs = tree.query_ball_point(P, r=rmax + tol_inside)
            for p_idx, neigh in enumerate(idxs):
                if not neigh:
                    continue
                d = np.linalg.norm(centers[neigh] - P[p_idx], axis=1)
                total[p_idx] += int(np.sum(d <= (radii[neigh] + tol_inside)))
        
        # Early stopping
        if early_stop_at is not None and np.any(total >= early_stop_at):
            keep = np.where(total >= 2)[0][:5]
            samples.extend(pts[keep])
            counts.extend([int(x) for x in total[keep]])
            mmax = int(np.max(total))
            return mmax, np.array(samples), np.array(counts)
        
        good = np.where(total >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in total[good]])
    
    if not samples:
        return 0, np.empty((0, 3)), np.empty(0)
    
    # Cluster nearby points
    samples = np.array(samples)
    counts = np.array(counts)
    
    mmax = int(counts.max()) if len(counts) else 0
    return mmax, samples, counts


def find_threshold_s_for_N(sublattices: List[Sublattice], p: LatticeParams,
                           target_N: int, s_min: float = 0.01, s_max: float = 1.5,
                           k_samples_coarse: int = 8, k_samples_fine: int = 16,
                           tol_inside: float = 1e-3, max_iter: int = 25) -> Optional[float]:
    """
    Find minimum scale factor s* such that max_multiplicity >= target_N.
    Uses coarse sweep followed by bisection.
    
    Returns None if target cannot be achieved within s_max.
    """
    # Coarse sweep to find initial bounds
    s_hi = None
    for s in np.linspace(s_min, s_max, 20):
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, s,
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            early_stop_at=target_N
        )
        if m >= target_N:
            s_hi = s
            break
    
    if s_hi is None:
        return None
    
    # Refine lower bound
    s_lo = s_min
    for s in np.linspace(s_min, s_hi, 15):
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, s,
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            early_stop_at=target_N
        )
        if m < target_N:
            s_lo = s
        else:
            s_hi = s
            break
    
    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (s_lo + s_hi)
        if (s_hi - s_lo) < 1e-5:
            break
        
        ks = k_samples_coarse if (s_hi - s_lo) > 0.02 else k_samples_fine
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, mid,
            k_samples=ks,
            tol_inside=tol_inside,
            early_stop_at=target_N
        )
        if m >= target_N:
            s_hi = mid
        else:
            s_lo = mid
    
    return s_hi


def compute_min_scale_for_cn(config_offsets: List[Tuple[float, float, float]],
                             target_cn: int,
                             lattice_type: str,
                             alpha_ratio: float | Tuple[float, ...] | List[float] = 0.5,
                             bravais_type: Optional[str] = None,
                             lattice_params: Optional[dict] = None) -> Optional[float]:
    """
    Compute minimum scale factor for a specific lattice configuration to achieve target CN.
    
    Args:
        config_offsets: List of fractional coordinate offsets
        target_cn: Target coordination number (intersection multiplicity)
        lattice_type: 'Cubic', 'Tetragonal', 'Hexagonal', etc.
        alpha_ratio: r = alpha * s * a (single value for all, or tuple/list of per-offset values)
        bravais_type: Specific Bravais type (e.g., 'cubic_F', 'tetragonal_I')
        lattice_params: Optional dict with 'b_ratio', 'c_ratio', 'alpha', 'beta', 'gamma'
    
    Returns:
        Minimum scale factor s*, or None if not achievable
    """
    # Set up lattice parameters based on type
    params = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0, 
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
    
    if lattice_type == 'Hexagonal':
        params['gamma'] = 120.0
        # For HCP, use ideal c/a ratio
        if bravais_type == 'hexagonal_H':
            params['c_ratio'] = 1.633
    elif lattice_type == 'Rhombohedral':
        params['alpha'] = params['beta'] = params['gamma'] = 80.0
    elif lattice_type == 'Monoclinic':
        params['beta'] = 100.0
    
    # Override with user params
    if lattice_params:
        params.update(lattice_params)
    
    p = LatticeParams(**params)
    
    # Determine Bravais type if not specified
    if bravais_type is None:
        bravais_type = lattice_type.lower() + '_P'
    
    # Normalize alpha_ratio to tuple if it's a list
    if isinstance(alpha_ratio, list):
        alpha_ratio = tuple(alpha_ratio)
    
    # Create sublattice
    sub = Sublattice(
        name='M',
        offsets=tuple(tuple(o) for o in config_offsets),
        alpha_ratio=alpha_ratio,
        bravais_type=bravais_type
    )
    
    # Find threshold
    s_star = find_threshold_s_for_N([sub], p, target_cn)
    return s_star


def batch_compute_min_scales(configs: List[dict], target_cn: int,
                             alpha_ratio: float = 0.5,
                             lattice_params: Optional[dict] = None) -> Dict[str, Optional[float]]:
    """
    Compute minimum scale factors for multiple configurations.
    
    Args:
        configs: List of dicts with 'id', 'lattice', 'offsets', optionally 'bravais_type'
        target_cn: Target coordination number
        alpha_ratio: Sphere radius ratio
        lattice_params: Optional additional lattice parameters
    
    Returns:
        Dict mapping config ID to minimum scale factor (or None)
    """
    results = {}
    for config in configs:
        if config.get('offsets') is None:  # Parametric config
            results[config['id']] = None
            continue
        
        s_star = compute_min_scale_for_cn(
            config['offsets'],
            target_cn,
            config['lattice'],
            alpha_ratio,
            config.get('bravais_type'),
            lattice_params
        )
        results[config['id']] = s_star
    
    return results


def scan_c_ratio_for_min_scale(
    config_offsets: List[Tuple[float, float, float]],
    target_cn: int,
    lattice_type: str,
    alpha_ratio: float | Tuple[float, ...] | List[float] = 0.5,
    bravais_type: Optional[str] = None,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.0,
    scan_level: str = 'fine',  # 'coarse', 'medium', 'fine', 'ultrafine'
    lattice_params: Optional[dict] = None,
    optimize_metric: str = 's3_over_volume',  # 's_star' or 's3_over_volume'
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Scan c/a ratio to find the minimum scale factor for a target CN.
    
    Uses hierarchical scanning:
    - Coarse: 5 points
    - Medium: 5 points between best two from coarse
    - Fine: 10 points between best two from medium
    - Ultrafine: 10 additional points between best two from fine
    
    Args:
        config_offsets: List of fractional coordinate offsets
        target_cn: Target coordination number (intersection multiplicity)
        lattice_type: 'Tetragonal', 'Hexagonal', etc.
        alpha_ratio: r = alpha * s * a (single value for all, or tuple/list of per-offset values)
        bravais_type: Specific Bravais type
        c_ratio_min: Minimum c/a ratio to scan
        c_ratio_max: Maximum c/a ratio to scan
        scan_level: 'coarse', 'medium', 'fine', or 'ultrafine'
        lattice_params: Optional dict with other lattice parameters
        optimize_metric: 's_star' to minimize s*, or 's3_over_volume' to minimize s³/V
                        (s³/V is proportional to packing fraction)
        progress_callback: Optional callback(current, total, message) for progress updates
    
    Returns:
        Dict with:
            'best_c_ratio': Optimal c/a ratio
            'best_s_star': Minimum scale factor at optimal c/a
            'best_volume': Unit cell volume at optimal c/a (a²c for tetragonal)
            'best_metric': The optimized metric value (s* or s³/V)
            'scan_results': List of (c_ratio, s_star, volume, metric) tuples from all scans
            'scan_history': Dict with results from each scan level
    """
    # Set up base lattice parameters
    params = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0, 
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
    
    if lattice_type == 'Hexagonal':
        params['gamma'] = 120.0
    elif lattice_type == 'Rhombohedral':
        params['alpha'] = params['beta'] = params['gamma'] = 80.0
    elif lattice_type == 'Monoclinic':
        params['beta'] = 100.0
    
    if lattice_params:
        params.update(lattice_params)
    
    if bravais_type is None:
        bravais_type = lattice_type.lower() + '_P'
    
    all_results = []
    scan_history = {}
    
    def compute_volume(c_ratio: float, b_ratio: float = 1.0) -> float:
        """Compute unit cell volume. For tetragonal/hex: a²c, for ortho: abc."""
        a = params['a']
        b = a * b_ratio
        c = a * c_ratio
        
        if lattice_type == 'Hexagonal':
            # Hexagonal: V = (√3/2) * a² * c
            return (np.sqrt(3) / 2) * a * a * c
        elif lattice_type == 'Tetragonal':
            # Tetragonal: V = a² * c
            return a * a * c
        elif lattice_type == 'Orthorhombic':
            # Orthorhombic: V = a * b * c
            return a * b * c
        else:
            # Default cubic-like
            return a * a * c
    
    def evaluate_c_ratio(c_ratio: float) -> Tuple[Optional[float], float, Optional[float]]:
        """Evaluate s* for a given c/a ratio. Returns (s_star, volume, metric)."""
        params['c_ratio'] = c_ratio
        p = LatticeParams(**params)
        
        sub = Sublattice(
            name='M',
            offsets=tuple(tuple(o) for o in config_offsets),
            alpha_ratio=alpha_ratio,
            bravais_type=bravais_type
        )
        
        s_star = find_threshold_s_for_N([sub], p, target_cn)
        volume = compute_volume(c_ratio, params.get('b_ratio', 1.0))
        
        if s_star is not None:
            if optimize_metric == 's3_over_volume':
                # s³/V is proportional to packing fraction
                metric = (s_star ** 3) / volume
            else:
                metric = s_star
        else:
            metric = None
        
        return s_star, volume, metric
    
    def scan_range(c_min: float, c_max: float, n_points: int, level_name: str) -> List[Tuple[float, Optional[float], float, Optional[float]]]:
        """Scan a range of c/a ratios. Returns (c_ratio, s_star, volume, metric)."""
        results = []
        c_ratios = np.linspace(c_min, c_max, n_points)
        
        for i, c_ratio in enumerate(c_ratios):
            if progress_callback:
                progress_callback(i + 1, n_points, f"{level_name}: c/a = {c_ratio:.3f}")
            
            s_star, volume, metric = evaluate_c_ratio(c_ratio)
            results.append((c_ratio, s_star, volume, metric))
            all_results.append((c_ratio, s_star, volume, metric))
        
        return results
    
    def find_best_two(results: List[Tuple[float, Optional[float], float, Optional[float]]]) -> Tuple[float, float]:
        """Find the two c/a values with the lowest metric values."""
        valid = [(c, s, v, m) for c, s, v, m in results if m is not None]
        if len(valid) < 2:
            return c_ratio_min, c_ratio_max
        
        # Sort by metric (s* or s³/V)
        sorted_results = sorted(valid, key=lambda x: x[3])
        
        best_c = sorted_results[0][0]
        second_c = sorted_results[1][0] if len(sorted_results) > 1 else sorted_results[0][0]
        
        return (min(best_c, second_c), max(best_c, second_c))
    
    def get_best_result():
        """Get the best result from all scans."""
        valid = [(c, s, v, m) for c, s, v, m in all_results if m is not None]
        if valid:
            best = min(valid, key=lambda x: x[3])
            return {
                'best_c_ratio': best[0],
                'best_s_star': best[1],
                'best_volume': best[2],
                'best_metric': best[3],
                'scan_results': all_results,
                'scan_history': scan_history,
                'optimize_metric': optimize_metric
            }
        return {
            'best_c_ratio': None, 
            'best_s_star': None, 
            'best_volume': None,
            'best_metric': None,
            'scan_results': all_results, 
            'scan_history': scan_history,
            'optimize_metric': optimize_metric
        }
    
    # Coarse scan (5 points)
    coarse_results = scan_range(c_ratio_min, c_ratio_max, 5, "Coarse")
    scan_history['coarse'] = coarse_results
    
    if scan_level == 'coarse':
        return get_best_result()
    
    # Medium scan (5 points between best two from coarse)
    c_min_med, c_max_med = find_best_two(coarse_results)
    margin = (c_max_med - c_min_med) * 0.2
    c_min_med = max(c_ratio_min, c_min_med - margin)
    c_max_med = min(c_ratio_max, c_max_med + margin)
    
    medium_results = scan_range(c_min_med, c_max_med, 5, "Medium")
    scan_history['medium'] = medium_results
    
    if scan_level == 'medium':
        return get_best_result()
    
    # Fine scan (10 points between best two from medium)
    c_min_fine, c_max_fine = find_best_two(medium_results)
    margin = (c_max_fine - c_min_fine) * 0.1
    c_min_fine = max(c_ratio_min, c_min_fine - margin)
    c_max_fine = min(c_ratio_max, c_max_fine + margin)
    
    fine_results = scan_range(c_min_fine, c_max_fine, 10, "Fine")
    scan_history['fine'] = fine_results
    
    if scan_level == 'fine':
        return get_best_result()
    
    # Ultrafine scan (10 additional points)
    c_min_uf, c_max_uf = find_best_two(fine_results)
    margin = (c_max_uf - c_min_uf) * 0.05
    c_min_uf = max(c_ratio_min, c_min_uf - margin)
    c_max_uf = min(c_ratio_max, c_max_uf + margin)
    
    ultrafine_results = scan_range(c_min_uf, c_max_uf, 10, "Ultrafine")
    scan_history['ultrafine'] = ultrafine_results
    
    return get_best_result()


def batch_scan_c_ratio(
    configs: List[dict],
    target_cn: int,
    alpha_ratio: float | Tuple[float, ...] | List[float] = 0.5,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.0,
    scan_level: str = 'fine',
    lattice_params: Optional[dict] = None,
    progress_callback: Optional[callable] = None
) -> Dict[str, Dict]:
    """
    Scan c/a ratio for multiple configurations.
    Only scans configs where c/a is relevant (Tetragonal, Hexagonal, Orthorhombic).
    
    Args:
        configs: List of config dicts with 'id', 'lattice', 'offsets', 'bravais_type'
        target_cn: Target coordination number
        alpha_ratio: Sphere radius ratio (single value or per-offset tuple/list)
        c_ratio_min: Minimum c/a ratio
        c_ratio_max: Maximum c/a ratio
        scan_level: Scan resolution level
        lattice_params: Optional additional parameters
        progress_callback: Optional callback(config_id, current, total)
    
    Returns:
        Dict mapping config ID to scan results
    """
    # Lattices where c/a scanning is relevant
    scannable_lattices = {'Tetragonal', 'Hexagonal', 'Orthorhombic'}
    
    results = {}
    scannable_configs = [c for c in configs 
                        if c.get('lattice') in scannable_lattices 
                        and c.get('offsets') is not None]
    
    for i, config in enumerate(scannable_configs):
        if progress_callback:
            progress_callback(config['id'], i + 1, len(scannable_configs))
        
        scan_result = scan_c_ratio_for_min_scale(
            config['offsets'],
            target_cn,
            config['lattice'],
            alpha_ratio,
            config.get('bravais_type'),
            c_ratio_min,
            c_ratio_max,
            scan_level,
            lattice_params
        )
        
        results[config['id']] = scan_result
    
    return results
