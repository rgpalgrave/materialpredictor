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


@dataclass(frozen=True)
class Sublattice:
    """A sublattice with positions and sphere parameters."""
    name: str
    offsets: Tuple[Tuple[float, float, float], ...]  # Fractional coordinates
    alpha_ratio: float = 0.5  # Sphere radius = alpha_ratio * scale * a
    
    def __post_init__(self):
        # Convert list to tuple if needed
        if isinstance(self.offsets, list):
            object.__setattr__(self, 'offsets', tuple(tuple(o) for o in self.offsets))


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
    """Build arrays of sphere centers and radii from sublattices."""
    lat_vecs = lattice_vectors(p)
    centers = []
    radii = []
    
    for sub in sublattices:
        for offset in sub.offsets:
            cart = frac_to_cart(np.array(offset), lat_vecs)
            centers.append(cart)
            radii.append(sub.alpha_ratio * scale_s * p.a)
    
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
                             alpha_ratio: float = 0.5,
                             lattice_params: Optional[dict] = None) -> Optional[float]:
    """
    Compute minimum scale factor for a specific lattice configuration to achieve target CN.
    
    Args:
        config_offsets: List of fractional coordinate offsets
        target_cn: Target coordination number (intersection multiplicity)
        lattice_type: 'Cubic', 'Tetragonal', 'Hexagonal', etc.
        alpha_ratio: r = alpha * s * a
        lattice_params: Optional dict with 'b_ratio', 'c_ratio', 'alpha', 'beta', 'gamma'
    
    Returns:
        Minimum scale factor s*, or None if not achievable
    """
    # Set up lattice parameters based on type
    params = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0, 
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
    
    if lattice_type == 'Hexagonal':
        params['gamma'] = 120.0
    elif lattice_type == 'Rhombohedral':
        params['alpha'] = params['beta'] = params['gamma'] = 80.0  # Default rhombohedral angle
    elif lattice_type == 'Monoclinic':
        params['beta'] = 100.0  # Default monoclinic angle
    
    # Override with user params
    if lattice_params:
        params.update(lattice_params)
    
    p = LatticeParams(**params)
    
    # Create sublattice
    sub = Sublattice(
        name='M',
        offsets=tuple(tuple(o) for o in config_offsets),
        alpha_ratio=alpha_ratio
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
        configs: List of dicts with 'id', 'lattice', 'offsets'
        target_cn: Target coordination number
        alpha_ratio: Sphere radius ratio
        lattice_params: Optional additional lattice parameters
    
    Returns:
        Dict mapping config ID to minimum scale factor (or None)
    """
    results = {}
    for config in configs:
        if config['offsets'] is None:  # Parametric config
            results[config['id']] = None
            continue
        
        s_star = compute_min_scale_for_cn(
            config['offsets'],
            target_cn,
            config['lattice'],
            alpha_ratio,
            lattice_params
        )
        results[config['id']] = s_star
    
    return results
