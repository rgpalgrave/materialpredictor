"""
Interstitial Engine for Sphere Intersection Analysis (Optimized)
Finds threshold lattice parameter 'a' (Å) where N-fold sphere intersections occur.

Key optimizations:
- scipy.spatial.cKDTree (C-optimized) instead of KDTree
- KDTree neighbor queries O(n log n) instead of O(n²) pair generation
- LRU cache for lattice vectors and periodic shifts
- Hash-based duplicate elimination for sublattice positions
- Batch neighbor queries for sample points
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Callable, Union
from functools import lru_cache

from scipy.spatial import cKDTree


# -----------------------------
# Data models
# -----------------------------

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
BRAVAIS_BASIS: Dict[str, List[Tuple[float, float, float]]] = {
    # Cubic
    'cubic_P': [(0, 0, 0)],
    'cubic_I': [(0, 0, 0), (0.5, 0.5, 0.5)],  # BCC
    'cubic_F': [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],  # FCC

    # Tetragonal
    'tetragonal_P': [(0, 0, 0)],
    'tetragonal_I': [(0, 0, 0), (0.5, 0.5, 0.5)],

    # Orthorhombic
    'orthorhombic_P': [(0, 0, 0)],
    'orthorhombic_I': [(0, 0, 0), (0.5, 0.5, 0.5)],
    'orthorhombic_F': [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    'orthorhombic_C': [(0, 0, 0), (0.5, 0.5, 0)],
    'orthorhombic_A': [(0, 0, 0), (0, 0.5, 0.5)],
    'orthorhombic_B': [(0, 0, 0), (0.5, 0, 0.5)],

    # Hexagonal
    'hexagonal_P': [(0, 0, 0)],
    'hexagonal_H': [(0, 0, 0), (2/3, 1/3, 0.5)],  # HCP centering translation

    # Rhombohedral
    'rhombohedral_P': [(0, 0, 0)],

    # Monoclinic
    'monoclinic_P': [(0, 0, 0)],
    'monoclinic_C': [(0, 0, 0), (0.5, 0.5, 0)],

    # Triclinic
    'triclinic_P': [(0, 0, 0)],
}


AlphaLike = Union[float, Tuple[float, ...], List[float]]


@dataclass(frozen=True)
class Sublattice:
    """A sublattice with positions and sphere parameters."""
    name: str
    offsets: Tuple[Tuple[float, float, float], ...]  # Fractional coordinates
    alpha_ratio: AlphaLike = 0.5  # In NEW MODEL: coordination radius in Å
    bravais_type: str = 'cubic_P'  # Bravais lattice type for centering

    def __post_init__(self):
        if isinstance(self.offsets, list):
            object.__setattr__(self, 'offsets', tuple(tuple(o) for o in self.offsets))
        if isinstance(self.alpha_ratio, list):
            object.__setattr__(self, 'alpha_ratio', tuple(self.alpha_ratio))

    def get_alpha_for_offset(self, offset_idx: int) -> float:
        if isinstance(self.alpha_ratio, (int, float)):
            return float(self.alpha_ratio)
        if offset_idx < len(self.alpha_ratio):
            return float(self.alpha_ratio[offset_idx])
        return float(self.alpha_ratio[-1])

    @staticmethod
    def _pos_key(pos: Tuple[float, float, float], tol: float = 1e-6) -> Tuple[int, int, int]:
        """Quantize position for hash-based deduplication."""
        return (int(round(pos[0] / tol)), int(round(pos[1] / tol)), int(round(pos[2] / tol)))

    def get_all_positions_with_alpha(self, tol: float = 1e-6) -> List[Tuple[Tuple[float, float, float], float]]:
        """Get all atomic positions with their alpha ratios, including Bravais centering."""
        basis = BRAVAIS_BASIS.get(self.bravais_type, [(0, 0, 0)])
        out: List[Tuple[Tuple[float, float, float], float]] = []
        seen = set()

        for offset_idx, offset in enumerate(self.offsets):
            alpha = self.get_alpha_for_offset(offset_idx)
            ox, oy, oz = offset
            for bx, by, bz in basis:
                pos = ((ox + bx) % 1.0, (oy + by) % 1.0, (oz + bz) % 1.0)
                k = self._pos_key(pos, tol=tol)
                if k in seen:
                    continue
                seen.add(k)
                out.append((pos, alpha))
        return out


# -----------------------------
# Lattice utilities (cached)
# -----------------------------

@lru_cache(maxsize=2048)
def _lattice_vectors_cached(a: float, b_ratio: float, c_ratio: float,
                            alpha: float, beta: float, gamma: float) -> Tuple[Tuple[float, float, float], ...]:
    """Return lattice vectors as a tuple-of-tuples for caching."""
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    b = a * b_ratio
    c = a * c_ratio

    a_vec = np.array([a, 0.0, 0.0], dtype=float)
    b_vec = np.array([b * np.cos(gamma_r), b * np.sin(gamma_r), 0.0], dtype=float)

    c_x = c * np.cos(beta_r)
    sg = np.sin(gamma_r)
    if abs(sg) < 1e-14:
        c_y = 0.0
    else:
        c_y = c * (np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r)) / sg
    c_z_sq = max(0.0, c**2 - c_x**2 - c_y**2)
    c_vec = np.array([c_x, c_y, np.sqrt(c_z_sq)], dtype=float)

    lat = np.array([a_vec, b_vec, c_vec], dtype=float)
    return tuple(tuple(float(x) for x in row) for row in lat)


def lattice_vectors(p: LatticeParams) -> np.ndarray:
    """Generate 3x3 matrix of lattice vectors (row-wise)."""
    return np.array(_lattice_vectors_cached(p.a, p.b_ratio, p.c_ratio, p.alpha, p.beta, p.gamma), dtype=float)


@lru_cache(maxsize=2048)
def _shifts_cached(lat_key: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
    """Cache the 27 periodic shift vectors."""
    lat_vecs = np.array(lat_key, dtype=float)
    shifts = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                shifts.append(tuple((np.array([i, j, k], dtype=float) @ lat_vecs).tolist()))
    return tuple(shifts)


def generate_shifts(lat_vecs: np.ndarray) -> np.ndarray:
    """Generate 27 shift vectors for periodic boundary conditions."""
    lat_key = tuple(tuple(float(x) for x in row) for row in lat_vecs)
    return np.array(_shifts_cached(lat_key), dtype=float)


def frac_to_cart(frac: np.ndarray, lat_vecs: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian coordinates."""
    return frac @ lat_vecs


# -----------------------------
# Core geometry
# -----------------------------

def pair_circle_samples(c1: np.ndarray, r1: float, c2: np.ndarray, r2: float, k: int = 12) -> np.ndarray:
    """
    Sample points on the intersection circle of two spheres.
    Returns empty array if spheres don't intersect or one contains the other.
    """
    d = float(np.linalg.norm(c2 - c1))
    if d < 1e-10:
        return np.empty((0, 3), dtype=float)
    if d > r1 + r2:
        return np.empty((0, 3), dtype=float)
    if d < abs(r1 - r2):
        return np.empty((0, 3), dtype=float)

    h = (d*d + r1*r1 - r2*r2) / (2.0 * d)
    r_circle_sq = r1*r1 - h*h
    if r_circle_sq <= 0.0:
        return np.empty((0, 3), dtype=float)
    r_circle = float(np.sqrt(r_circle_sq))

    axis = (c2 - c1) / d
    center = c1 + h * axis

    if abs(axis[0]) < 0.9:
        perp1 = np.cross(axis, np.array([1.0, 0.0, 0.0]))
    else:
        perp1 = np.cross(axis, np.array([0.0, 1.0, 0.0]))
    n1 = float(np.linalg.norm(perp1))
    if n1 < 1e-12:
        return np.empty((0, 3), dtype=float)
    perp1 = perp1 / n1
    perp2 = np.cross(axis, perp1)

    angles = np.linspace(0.0, 2.0*np.pi, int(k), endpoint=False)
    ct = np.cos(angles)[:, None]
    st = np.sin(angles)[:, None]
    pts = center[None, :] + r_circle * (ct * perp1[None, :] + st * perp2[None, :])
    return pts.astype(float, copy=False)


def build_centers_and_radii(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build arrays of sphere centers and radii from sublattices.

    NEW MODEL (scale_s is None):
        p.a is the lattice parameter (Å).
        alpha_ratio is the actual sphere radius in Å.

    LEGACY MODEL (scale_s provided):
        radius = alpha_ratio * scale_s
    """
    lat_vecs = lattice_vectors(p)
    centers = []
    radii = []

    for sub in sublattices:
        for pos, coord_radius in sub.get_all_positions_with_alpha():
            cart = frac_to_cart(np.array(pos, dtype=float), lat_vecs)
            centers.append(cart)
            if scale_s is None:
                radii.append(float(coord_radius))
            else:
                radii.append(float(coord_radius) * float(scale_s))

    if not centers:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)

    return np.vstack(centers).astype(float, copy=False), np.array(radii, dtype=float)


def periodic_candidate_pairs_fast(
    centers: np.ndarray,
    shifts: np.ndarray,
    cutoff: float
) -> List[Tuple[int, int, int]]:
    """
    Find all pairs of spheres that could intersect within cutoff distance.
    Uses cKDTree neighbor queries - O(n log n) instead of O(n²).

    Returns list of (i, j, shift_idx) for pairs where center_j + shift is within cutoff of center_i.
    """
    n = int(len(centers))
    if n == 0:
        return []

    tree = cKDTree(centers)
    pairs: List[Tuple[int, int, int]] = []
    zero_shift_idx = 13  # Index of (0,0,0) shift in the 27 shifts

    for s_idx, shift in enumerate(shifts):
        shifted_centers = centers + shift
        neighs = tree.query_ball_point(shifted_centers, r=cutoff)
        
        if s_idx == zero_shift_idx:
            # Same cell - only take i < j to avoid duplicates
            for j, iset in enumerate(neighs):
                for i in iset:
                    if i < j:
                        pairs.append((i, j, s_idx))
        else:
            # Different cells - take all pairs
            for j, iset in enumerate(neighs):
                for i in iset:
                    pairs.append((i, j, s_idx))

    return pairs


def _build_periodic_image_tree(
    centers: np.ndarray, 
    radii: np.ndarray, 
    shifts: np.ndarray
) -> Tuple[cKDTree, np.ndarray, np.ndarray]:
    """
    Build a cKDTree over all 27 periodic images (27*n points).
    Returns (tree, centers_img, radii_img).
    """
    centers_img = np.vstack([centers + sh for sh in shifts]).astype(float, copy=False)
    radii_img = np.tile(radii, len(shifts))
    tree_img = cKDTree(centers_img)
    return tree_img, centers_img, radii_img


def max_multiplicity_for_scale(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    k_samples: int = 16,
    tol_inside: float = 1e-3,
    early_stop_at: Optional[int] = None,
    use_new_model: bool = True
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find maximum intersection multiplicity for a given scale factor.

    NEW MODEL (use_new_model=True):
        scale_s is the lattice parameter 'a' in Å.
        Sphere radii are fixed at alpha_ratio (coordination radii).

    Returns:
        max_mult, sample_positions, multiplicities
    """
    if use_new_model:
        p_scaled = LatticeParams(
            a=float(scale_s),
            b_ratio=p.b_ratio,
            c_ratio=p.c_ratio,
            alpha=p.alpha,
            beta=p.beta,
            gamma=p.gamma
        )
        lat_vecs = lattice_vectors(p_scaled)
        centers, radii = build_centers_and_radii(sublattices, p_scaled, scale_s=None)
    else:
        lat_vecs = lattice_vectors(p)
        centers, radii = build_centers_and_radii(sublattices, p, scale_s=float(scale_s))

    if len(centers) == 0:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    shifts = generate_shifts(lat_vecs)
    rmax = float(np.max(radii))
    cutoff = 2.0 * rmax

    # Find candidate pairs using cKDTree
    pairs = periodic_candidate_pairs_fast(centers, shifts, cutoff=cutoff)
    if not pairs:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    # Build tree over all periodic images for fast multiplicity counting
    tree_img, centers_img, radii_img = _build_periodic_image_tree(centers, radii, shifts)

    samples: List[np.ndarray] = []
    counts: List[int] = []
    search_r = rmax + float(tol_inside)

    for (i, j, s_idx) in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = float(radii[i])
        r2 = float(radii[j])

        pts = pair_circle_samples(c1, r1, c2, r2, k=int(k_samples))
        if pts.size == 0:
            continue

        # Batch query neighbors for all sample points
        neigh_lists = tree_img.query_ball_point(pts, r=search_r)

        # Compute multiplicity for each point
        total = np.zeros(len(pts), dtype=int)
        for p_idx, neigh in enumerate(neigh_lists):
            if not neigh:
                continue
            neigh = np.asarray(neigh, dtype=int)
            d = np.linalg.norm(centers_img[neigh] - pts[p_idx], axis=1)
            total[p_idx] = int(np.sum(d <= (radii_img[neigh] + tol_inside)))

        if early_stop_at is not None and int(total.max()) >= int(early_stop_at):
            keep = np.where(total >= 2)[0][:5]
            if keep.size:
                samples.extend(pts[keep])
                counts.extend([int(x) for x in total[keep]])
            return int(total.max()), np.array(samples) if samples else pts[keep], np.array(counts, dtype=int) if counts else total[keep].astype(int)

        good = np.where(total >= 2)[0]
        if good.size:
            samples.extend(pts[good])
            counts.extend([int(x) for x in total[good]])

    if not samples:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    samples_arr = np.array(samples, dtype=float)
    counts_arr = np.array(counts, dtype=int)
    return int(counts_arr.max()), samples_arr, counts_arr


def find_threshold_s_for_N(
    sublattices: List[Sublattice],
    p: LatticeParams,
    target_N: int,
    s_min: float = 2.0,
    s_max: float = 15.0,
    k_samples_coarse: int = 6,
    k_samples_fine: int = 14,
    tol_inside: float = 1e-3,
    max_iter: int = 22,
    use_new_model: bool = True
) -> Optional[float]:
    """
    Find threshold lattice parameter for target coordination number.
    
    NEW MODEL (default):
        Find MAXIMUM lattice parameter 'a' where multiplicity >= target_N.
        Larger 'a' -> less overlap -> lower multiplicity.

    Returns None if target cannot be achieved within [s_min, s_max].
    """
    target_N = int(target_N)

    if use_new_model:
        # Find bracket where multiplicity crosses target
        s_lo = None
        for s in np.linspace(s_max, s_min, 14):
            m, _, _ = max_multiplicity_for_scale(
                sublattices, p, float(s),
                k_samples=k_samples_coarse,
                tol_inside=tol_inside,
                early_stop_at=target_N,
                use_new_model=True
            )
            if m >= target_N:
                s_lo = float(s)
                break
        if s_lo is None:
            return None

        # Find upper bound where multiplicity drops
        s_hi = float(s_max)
        for s in np.linspace(s_lo, s_max, 10):
            m, _, _ = max_multiplicity_for_scale(
                sublattices, p, float(s),
                k_samples=k_samples_coarse,
                tol_inside=tol_inside,
                early_stop_at=target_N,
                use_new_model=True
            )
            if m >= target_N:
                s_lo = float(s)
            else:
                s_hi = float(s)
                break

        # Bisection refinement
        for _ in range(int(max_iter)):
            if (s_hi - s_lo) < 1e-4:
                break
            mid = 0.5 * (s_lo + s_hi)
            ks = k_samples_coarse if (s_hi - s_lo) > 0.12 else k_samples_fine
            m, _, _ = max_multiplicity_for_scale(
                sublattices, p, float(mid),
                k_samples=int(ks),
                tol_inside=tol_inside,
                early_stop_at=target_N,
                use_new_model=True
            )
            if m >= target_N:
                s_lo = float(mid)
            else:
                s_hi = float(mid)
        return s_lo

    # Legacy mode
    s_hi = None
    for s in np.linspace(s_min, s_max, 18):
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, float(s),
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            early_stop_at=target_N,
            use_new_model=False
        )
        if m >= target_N:
            s_hi = float(s)
            break
    if s_hi is None:
        return None

    s_lo = float(s_min)
    for s in np.linspace(s_min, s_hi, 12):
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, float(s),
            k_samples=k_samples_coarse,
            tol_inside=tol_inside,
            early_stop_at=target_N,
            use_new_model=False
        )
        if m < target_N:
            s_lo = float(s)
        else:
            s_hi = float(s)
            break

    for _ in range(int(max_iter)):
        if (s_hi - s_lo) < 1e-4:
            break
        mid = 0.5 * (s_lo + s_hi)
        ks = k_samples_coarse if (s_hi - s_lo) > 0.12 else k_samples_fine
        m, _, _ = max_multiplicity_for_scale(
            sublattices, p, float(mid),
            k_samples=int(ks),
            tol_inside=tol_inside,
            early_stop_at=target_N,
            use_new_model=False
        )
        if m >= target_N:
            s_hi = float(mid)
        else:
            s_lo = float(mid)
    return s_hi


# -----------------------------
# Public API
# -----------------------------

def compute_min_scale_for_cn(
    config_offsets: List[Tuple[float, float, float]],
    target_cn: int,
    lattice_type: str,
    alpha_ratio: AlphaLike = 0.5,
    bravais_type: Optional[str] = None,
    lattice_params: Optional[dict] = None
) -> Optional[float]:
    """
    Compute threshold lattice parameter 'a' (Å) for target CN.
    alpha_ratio is interpreted as coordination radius (Å) in the NEW MODEL.
    """
    params = {'a': 1.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}

    if lattice_type == 'Hexagonal':
        params['gamma'] = 120.0
        if bravais_type == 'hexagonal_H':
            params['c_ratio'] = 1.633
    elif lattice_type == 'Rhombohedral':
        params['alpha'] = params['beta'] = params['gamma'] = 80.0
    elif lattice_type == 'Monoclinic':
        params['beta'] = 100.0

    if lattice_params:
        params.update(lattice_params)

    p = LatticeParams(**params)

    if bravais_type is None:
        bravais_type = lattice_type.lower() + '_P'

    if isinstance(alpha_ratio, list):
        alpha_ratio = tuple(alpha_ratio)

    sub = Sublattice(
        name='M',
        offsets=tuple(tuple(o) for o in config_offsets),
        alpha_ratio=alpha_ratio,
        bravais_type=bravais_type
    )

    return find_threshold_s_for_N([sub], p, int(target_cn), use_new_model=True)


def batch_compute_min_scales(
    configs: List[dict],
    target_cn: int,
    alpha_ratio: float = 0.5,
    lattice_params: Optional[dict] = None
) -> Dict[str, Optional[float]]:
    """Batch compute minimum scales for multiple configurations."""
    results: Dict[str, Optional[float]] = {}
    for config in configs:
        if config.get('offsets') is None:
            results[config['id']] = None
            continue
        results[config['id']] = compute_min_scale_for_cn(
            config['offsets'],
            target_cn,
            config['lattice'],
            alpha_ratio,
            config.get('bravais_type'),
            lattice_params
        )
    return results


def scan_c_ratio_for_min_scale(
    config_offsets: List[Tuple[float, float, float]],
    target_cn: int,
    lattice_type: str,
    alpha_ratio: AlphaLike = 0.5,
    bravais_type: Optional[str] = None,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.0,
    scan_level: str = 'fine',
    lattice_params: Optional[dict] = None,
    optimize_metric: str = 's3_over_volume',
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict:
    """
    Scan c/a ratio to find optimal value minimizing s³/V or s*.
    Uses hierarchical scanning: coarse -> medium -> fine -> ultrafine.
    """
    params = {'a': 1.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
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

    if isinstance(alpha_ratio, list):
        alpha_ratio = tuple(alpha_ratio)

    all_results: List[Tuple[float, Optional[float], float, Optional[float]]] = []
    scan_history: Dict[str, List[Tuple[float, Optional[float], float, Optional[float]]]] = {}

    def compute_volume(a: float, c_ratio: float, b_ratio: float = 1.0) -> float:
        b = a * b_ratio
        c = a * c_ratio
        if lattice_type == 'Hexagonal':
            return (np.sqrt(3) / 2.0) * a * a * c
        elif lattice_type == 'Tetragonal':
            return a * a * c
        elif lattice_type == 'Orthorhombic':
            return a * b * c
        else:
            return a * a * c

    def evaluate_c_ratio(c_ratio: float) -> Tuple[Optional[float], float, Optional[float]]:
        params['c_ratio'] = float(c_ratio)
        p = LatticeParams(**params)
        sub = Sublattice(
            name='M',
            offsets=tuple(tuple(o) for o in config_offsets),
            alpha_ratio=alpha_ratio,
            bravais_type=bravais_type
        )
        s_star = find_threshold_s_for_N([sub], p, int(target_cn))
        if s_star is None:
            return None, 0.0, None
        volume = float(compute_volume(float(s_star), float(c_ratio), params.get('b_ratio', 1.0)))
        if optimize_metric == 's3_over_volume':
            metric = (float(s_star) ** 3) / volume if volume > 0 else None
        else:
            metric = float(s_star)
        return float(s_star), volume, metric

    def scan_range(c_min: float, c_max: float, n_points: int, level_name: str):
        results = []
        c_vals = np.linspace(float(c_min), float(c_max), int(n_points))
        for i, c in enumerate(c_vals):
            if progress_callback:
                progress_callback(i + 1, int(n_points), f"{level_name}: c/a = {c:.3f}")
            s_star, vol, met = evaluate_c_ratio(float(c))
            tup = (float(c), s_star, float(vol), met)
            results.append(tup)
            all_results.append(tup)
        return results

    def find_best_two(results):
        valid = [r for r in results if r[3] is not None]
        if len(valid) < 2:
            return float(c_ratio_min), float(c_ratio_max)
        valid.sort(key=lambda x: x[3])
        c1 = valid[0][0]
        c2 = valid[1][0]
        return (min(c1, c2), max(c1, c2))

    def get_best_result():
        valid = [r for r in all_results if r[3] is not None]
        if not valid:
            return {
                'best_c_ratio': None, 'best_s_star': None, 'best_volume': None, 'best_metric': None,
                'scan_results': all_results, 'scan_history': scan_history, 'optimize_metric': optimize_metric
            }
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

    # Hierarchical scanning
    coarse = scan_range(c_ratio_min, c_ratio_max, 5, "Coarse")
    scan_history['coarse'] = coarse
    if scan_level == 'coarse':
        return get_best_result()

    cmin, cmax = find_best_two(coarse)
    margin = (cmax - cmin) * 0.2
    cmin = max(float(c_ratio_min), cmin - margin)
    cmax = min(float(c_ratio_max), cmax + margin)

    medium = scan_range(cmin, cmax, 5, "Medium")
    scan_history['medium'] = medium
    if scan_level == 'medium':
        return get_best_result()

    cmin, cmax = find_best_two(medium)
    margin = (cmax - cmin) * 0.1
    cmin = max(float(c_ratio_min), cmin - margin)
    cmax = min(float(c_ratio_max), cmax + margin)

    fine = scan_range(cmin, cmax, 10, "Fine")
    scan_history['fine'] = fine
    if scan_level == 'fine':
        return get_best_result()

    cmin, cmax = find_best_two(fine)
    margin = (cmax - cmin) * 0.05
    cmin = max(float(c_ratio_min), cmin - margin)
    cmax = min(float(c_ratio_max), cmax + margin)

    ultra = scan_range(cmin, cmax, 10, "Ultrafine")
    scan_history['ultrafine'] = ultra
    return get_best_result()


def batch_scan_c_ratio(
    configs: List[dict],
    target_cn: int,
    alpha_ratio: AlphaLike = 0.5,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.0,
    scan_level: str = 'fine',
    lattice_params: Optional[dict] = None,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Dict]:
    """Batch scan c/a ratio for multiple configurations."""
    scannable = {'Tetragonal', 'Hexagonal', 'Orthorhombic'}
    results: Dict[str, Dict] = {}
    scannable_configs = [c for c in configs if c.get('lattice') in scannable and c.get('offsets') is not None]

    for i, config in enumerate(scannable_configs):
        if progress_callback:
            progress_callback(config['id'], i + 1, len(scannable_configs))
        results[config['id']] = scan_c_ratio_for_min_scale(
            config['offsets'],
            int(target_cn),
            config['lattice'],
            alpha_ratio,
            config.get('bravais_type'),
            float(c_ratio_min),
            float(c_ratio_max),
            scan_level,
            lattice_params
        )
    return results
