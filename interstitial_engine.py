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
    'rhombohedral_R': [(0, 0, 0)],  # R = rhombohedral setting (same as P for primitive)

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


class CachedGeometry:
    """
    Cache static geometry for efficient repeated multiplicity calculations.
    
    For a fixed configuration (sublattices, ratios, angles), the fractional
    positions and radii don't change - only the scale `a` changes.
    
    This caches:
    - Fractional positions (computed once)
    - Radii (computed once)
    - Fractional shifts (computed once for unit ratios)
    
    For each scale `a`:
    - Recompute lat_vecs = ratios * a
    - Convert fractional → Cartesian by simple matrix multiply
    """
    
    def __init__(self, sublattices: List[Sublattice], p_template: LatticeParams):
        """
        Initialize with sublattices and a template LatticeParams.
        The template's ratios and angles are used; `a` will be varied.
        """
        self.sublattices = sublattices
        self.b_ratio = p_template.b_ratio
        self.c_ratio = p_template.c_ratio
        self.alpha = p_template.alpha
        self.beta = p_template.beta
        self.gamma = p_template.gamma
        
        # Compute fractional positions and radii ONCE
        self._frac_positions = []
        self._radii = []
        
        for sub in sublattices:
            for pos, coord_radius in sub.get_all_positions_with_alpha():
                self._frac_positions.append(pos)
                self._radii.append(float(coord_radius))
        
        if self._frac_positions:
            self._frac_positions = np.array(self._frac_positions, dtype=float)
            self._radii = np.array(self._radii, dtype=float)
        else:
            self._frac_positions = np.empty((0, 3), dtype=float)
            self._radii = np.empty((0,), dtype=float)
        
        # Precompute unit lattice vectors (for a=1.0) - will scale by `a` later
        p_unit = LatticeParams(
            a=1.0, b_ratio=self.b_ratio, c_ratio=self.c_ratio,
            alpha=self.alpha, beta=self.beta, gamma=self.gamma
        )
        self._unit_lat_vecs = lattice_vectors(p_unit)
        
        # Precompute fractional shifts (for unit cell neighbors)
        # These are the same regardless of scale
        self._frac_shifts = []
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    self._frac_shifts.append([i, j, k])
        self._frac_shifts = np.array(self._frac_shifts, dtype=float)
        
        # Precompute squared radii thresholds (with small tolerance)
        self._rmax = float(np.max(self._radii)) if len(self._radii) > 0 else 0.0
    
    def get_cartesian_for_scale(self, a: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get Cartesian centers, radii, and shifts for a given scale `a`.
        This is fast - just matrix multiply, no iteration.
        """
        lat_vecs = self._unit_lat_vecs * a
        centers = self._frac_positions @ lat_vecs
        shifts = self._frac_shifts @ lat_vecs
        return centers, self._radii, shifts
    
    @property
    def radii(self) -> np.ndarray:
        return self._radii
    
    @property
    def rmax(self) -> float:
        return self._rmax
    
    @property
    def n_centers(self) -> int:
        return len(self._frac_positions)


def max_multiplicity_for_scale_cached(
    geom: CachedGeometry,
    scale_a: float,
    k_samples: int = 16,
    tol_inside: float = 1e-3,
    early_stop_at: Optional[int] = None
) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Find maximum intersection multiplicity using cached geometry.
    Much faster than max_multiplicity_for_scale for repeated calls.
    
    OPTIMIZED: 
    - Batches all sample points for single KDTree query
    - FULLY VECTORIZED multiplicity counting using flattened arrays + bincount
    """
    if geom.n_centers == 0:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    # Get Cartesian coordinates for this scale (fast - just matrix multiply)
    centers, radii, shifts = geom.get_cartesian_for_scale(scale_a)
    
    rmax = geom.rmax
    cutoff = 2.0 * rmax
    
    # Find candidate pairs (using canonical shifts)
    pairs = periodic_candidate_pairs_fast(centers, shifts, cutoff=cutoff)
    if not pairs:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    # Build tree over all periodic images
    tree_img, centers_img, radii_img = _build_periodic_image_tree(centers, radii, shifts)
    
    # Precompute squared thresholds
    radii_thresh_sq = (radii_img + tol_inside) ** 2
    search_r = rmax + float(tol_inside)
    
    # BATCH ALL SAMPLES
    all_pts_list = []
    for i, j, s_idx in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = float(radii[i])
        r2 = float(radii[j])
        
        pts = pair_circle_samples(c1, r1, c2, r2, k=int(k_samples))
        if pts.size > 0:
            all_pts_list.append(pts)
    
    if not all_pts_list:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    # Stack all points and do ONE batch query
    all_pts = np.vstack(all_pts_list)
    n_pts = len(all_pts)
    neigh_lists = tree_img.query_ball_point(all_pts, r=search_r)
    
    # FULLY VECTORIZED: Flatten all neighbors with point indices
    all_neigh = []
    all_pt_idx = []
    for p_idx, neigh in enumerate(neigh_lists):
        if neigh:
            all_neigh.extend(neigh)
            all_pt_idx.extend([p_idx] * len(neigh))
    
    if not all_neigh:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    all_neigh = np.asarray(all_neigh, dtype=np.int32)
    all_pt_idx = np.asarray(all_pt_idx, dtype=np.int32)
    
    # Vectorized squared distance for ALL neighbor pairs at once
    diff = centers_img[all_neigh] - all_pts[all_pt_idx]
    d_sq = np.einsum('ij,ij->i', diff, diff)
    
    # Vectorized threshold check
    inside = d_sq <= radii_thresh_sq[all_neigh]
    
    # Use bincount to sum up counts per point (fully vectorized!)
    total = np.bincount(all_pt_idx[inside], minlength=n_pts).astype(np.int32)
    
    # Early stop check
    max_mult = int(total.max()) if n_pts > 0 else 0
    if early_stop_at is not None and max_mult >= int(early_stop_at):
        keep = np.where(total >= 2)[0][:10]
        return max_mult, all_pts[keep], total[keep]
    
    # Filter to valid samples
    good = np.where(total >= 2)[0]
    if len(good) == 0:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    return int(total[good].max()), all_pts[good], total[good]


def periodic_candidate_pairs_fast(
    centers: np.ndarray,
    shifts: np.ndarray,
    cutoff: float
) -> List[Tuple[int, int, int]]:
    """
    Find all pairs of spheres that could intersect within cutoff distance.
    Uses cKDTree neighbor queries - O(n log n) instead of O(n²).
    
    OPTIMIZED: Uses canonical half of shifts to avoid duplicate pairs.
    For shift (dx, dy, dz), only process if:
    - It's the zero shift (0,0,0) and i < j
    - OR it's "lexicographically positive" (first non-zero component is positive)
    
    This eliminates symmetric duplicates like (i,j,+shift) vs (j,i,-shift).

    Returns list of (i, j, shift_idx) for unique pairs.
    """
    n = int(len(centers))
    if n == 0:
        return []

    tree = cKDTree(centers)
    pairs: List[Tuple[int, int, int]] = []
    
    # Determine which shifts are in the canonical half
    # A shift is canonical if:
    # - It's (0,0,0), OR
    # - The first non-zero component is positive
    canonical_shift_indices = []
    zero_shift_idx = None
    
    for s_idx, shift in enumerate(shifts):
        sx, sy, sz = shift[0], shift[1], shift[2]
        
        if abs(sx) < 1e-10 and abs(sy) < 1e-10 and abs(sz) < 1e-10:
            # Zero shift
            zero_shift_idx = s_idx
            canonical_shift_indices.append(s_idx)
        elif sx > 1e-10:
            # First component positive -> canonical
            canonical_shift_indices.append(s_idx)
        elif abs(sx) < 1e-10 and sy > 1e-10:
            # First is zero, second is positive -> canonical
            canonical_shift_indices.append(s_idx)
        elif abs(sx) < 1e-10 and abs(sy) < 1e-10 and sz > 1e-10:
            # First two zero, third positive -> canonical
            canonical_shift_indices.append(s_idx)
        # Otherwise skip (negative canonical counterpart exists)

    for s_idx in canonical_shift_indices:
        shift = shifts[s_idx]
        shifted_centers = centers + shift
        neighs = tree.query_ball_point(shifted_centers, r=cutoff)
        
        if s_idx == zero_shift_idx:
            # Same cell - only take i < j to avoid duplicates
            for j, iset in enumerate(neighs):
                for i in iset:
                    if i < j:
                        pairs.append((i, j, s_idx))
        else:
            # Canonical non-zero shift - take all pairs (no duplicates possible)
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

    OPTIMIZED: 
    - Uses squared distances (no sqrt)
    - Batches ALL sample points into single query
    - FULLY VECTORIZED multiplicity counting (no Python loops over neighbors)
    - Uses canonical shift half to avoid duplicate pairs

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

    # Find candidate pairs using cKDTree (with canonical shifts)
    pairs = periodic_candidate_pairs_fast(centers, shifts, cutoff=cutoff)
    if not pairs:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    # Build tree over all periodic images
    tree_img, centers_img, radii_img = _build_periodic_image_tree(centers, radii, shifts)
    
    # Precompute squared thresholds
    radii_thresh_sq = (radii_img + tol_inside) ** 2

    # BATCH ALL SAMPLES: collect all sample points first
    all_pts_list = []
    
    for i, j, s_idx in pairs:
        c1 = centers[i]
        c2 = centers[j] + shifts[s_idx]
        r1 = float(radii[i])
        r2 = float(radii[j])

        pts = pair_circle_samples(c1, r1, c2, r2, k=int(k_samples))
        if pts.size > 0:
            all_pts_list.append(pts)
    
    if not all_pts_list:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    # Stack all points and do ONE batch query
    all_pts = np.vstack(all_pts_list)
    n_pts = len(all_pts)
    search_r = rmax + float(tol_inside)
    neigh_lists = tree_img.query_ball_point(all_pts, r=search_r)
    
    # FULLY VECTORIZED: Flatten all neighbors with point indices
    # This eliminates the Python loop over neighbor lists
    all_neigh = []
    all_pt_idx = []
    for p_idx, neigh in enumerate(neigh_lists):
        if neigh:
            all_neigh.extend(neigh)
            all_pt_idx.extend([p_idx] * len(neigh))
    
    if not all_neigh:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)
    
    all_neigh = np.asarray(all_neigh, dtype=np.int32)
    all_pt_idx = np.asarray(all_pt_idx, dtype=np.int32)
    
    # Vectorized squared distance for ALL neighbor pairs at once
    diff = centers_img[all_neigh] - all_pts[all_pt_idx]
    d_sq = np.einsum('ij,ij->i', diff, diff)
    
    # Vectorized threshold check
    inside = d_sq <= radii_thresh_sq[all_neigh]
    
    # Use bincount to sum up counts per point (fully vectorized!)
    total = np.bincount(all_pt_idx[inside], minlength=n_pts).astype(np.int32)

    # Early stop check
    max_mult = int(total.max()) if n_pts > 0 else 0
    if early_stop_at is not None and max_mult >= int(early_stop_at):
        keep = np.where(total >= 2)[0][:10]
        return max_mult, all_pts[keep], total[keep]

    # Filter to valid samples
    good = np.where(total >= 2)[0]
    if len(good) == 0:
        return 0, np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    return int(total[good].max()), all_pts[good], total[good]


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

    OPTIMIZED: Uses CachedGeometry to avoid rebuilding positions on each iteration.

    Returns None if target cannot be achieved within [s_min, s_max].
    """
    target_N = int(target_N)

    if use_new_model:
        # Create cached geometry ONCE - reuse for all bisection iterations
        geom = CachedGeometry(sublattices, p)
        
        if geom.n_centers == 0:
            return None
        
        # Find bracket where multiplicity crosses target
        s_lo = None
        for s in np.linspace(s_max, s_min, 14):
            m, _, _ = max_multiplicity_for_scale_cached(
                geom, float(s),
                k_samples=k_samples_coarse,
                tol_inside=tol_inside,
                early_stop_at=target_N
            )
            if m >= target_N:
                s_lo = float(s)
                break
        if s_lo is None:
            return None

        # Find upper bound where multiplicity drops
        s_hi = float(s_max)
        for s in np.linspace(s_lo, s_max, 10):
            m, _, _ = max_multiplicity_for_scale_cached(
                geom, float(s),
                k_samples=k_samples_coarse,
                tol_inside=tol_inside,
                early_stop_at=target_N
            )
            if m >= target_N:
                s_lo = float(s)
            else:
                s_hi = float(s)
                break

        # Bisection refinement (using cached geometry)
        for _ in range(int(max_iter)):
            if (s_hi - s_lo) < 1e-4:
                break
            mid = 0.5 * (s_lo + s_hi)
            ks = k_samples_coarse if (s_hi - s_lo) > 0.12 else k_samples_fine
            m, _, _ = max_multiplicity_for_scale_cached(
                geom, float(mid),
                k_samples=int(ks),
                tol_inside=tol_inside,
                early_stop_at=target_N
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


# -----------------------------
# Monoclinic Structure Search
# -----------------------------

def scan_monoclinic_structure(
    config_offsets: List[Tuple[float, float, float]],
    target_cn: int,
    alpha_ratio: AlphaLike = 0.5,
    bravais_type: str = 'monoclinic_P',
    beta_min: float = 85.0,
    beta_max: float = 125.0,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.5,
    b_ratio_min: float = 0.7,
    b_ratio_max: float = 1.4,
    n_beta_coarse: int = 9,
    n_ca_coarse: int = 11,
    n_ba_fine: int = 7,
    n_refine: int = 5,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Dict]:
    """
    Hierarchical search for monoclinic structures.
    
    Search strategy:
    1. Phase 1: Scan beta angle with a=b=c (cubic-like ratios)
    2. Phase 2: At best beta, scan c/a ratio with a=b
    3. Phase 3: At best (beta, c/a), scan b/a ratio
    4. Phase 4: Local refinement around best point
    
    Args:
        config_offsets: Metal sublattice offset positions
        target_cn: Target coordination number
        alpha_ratio: Coordination radius (metal + anion radii)
        bravais_type: 'monoclinic_P' or 'monoclinic_C'
        beta_min/max: Beta angle range (degrees)
        c_ratio_min/max: c/a ratio range
        b_ratio_min/max: b/a ratio range
        n_beta_coarse: Number of beta points in coarse scan
        n_ca_coarse: Number of c/a points in coarse scan
        n_ba_fine: Number of b/a points in fine scan
        n_refine: Number of points in refinement
        progress_callback: Optional callback for progress updates
    
    Returns:
        Dict with best parameters, or None if no valid structure found
    """
    if isinstance(alpha_ratio, list):
        alpha_ratio = tuple(alpha_ratio)
    
    def report(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    def compute_volume(a: float, b_ratio: float, c_ratio: float, beta: float) -> float:
        """Monoclinic volume = a * b * c * sin(beta)"""
        b = a * b_ratio
        c = a * c_ratio
        beta_r = np.radians(beta)
        return a * b * c * np.sin(beta_r)
    
    def evaluate_params(beta: float, c_ratio: float, b_ratio: float) -> Tuple[Optional[float], float, Optional[float]]:
        """Evaluate structure at given monoclinic parameters."""
        params = {
            'a': 1.0,
            'b_ratio': float(b_ratio),
            'c_ratio': float(c_ratio),
            'alpha': 90.0,
            'beta': float(beta),
            'gamma': 90.0
        }
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
        volume = compute_volume(float(s_star), b_ratio, c_ratio, beta)
        # Optimize s³/V (lower is better - more compact structure)
        metric = (float(s_star) ** 3) / volume if volume > 0 else None
        return float(s_star), volume, metric
    
    all_results = []
    best_result = None
    best_metric = float('inf')
    
    # ========================================
    # Phase 1: Scan beta angle with a=b=c
    # ========================================
    report(f"Phase 1: Scanning beta angle ({beta_min}°-{beta_max}°) with a=b=c...")
    
    beta_values = np.linspace(beta_min, beta_max, n_beta_coarse)
    phase1_results = []
    
    for beta in beta_values:
        s_star, vol, metric = evaluate_params(beta, c_ratio=1.0, b_ratio=1.0)
        result = {
            'beta': float(beta),
            'c_ratio': 1.0,
            'b_ratio': 1.0,
            's_star': s_star,
            'volume': vol,
            'metric': metric
        }
        phase1_results.append(result)
        all_results.append(result)
        
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = result
    
    if best_result is None:
        report("Phase 1: No valid structures found")
        return None
    
    best_beta = best_result['beta']
    report(f"Phase 1: Best beta = {best_beta:.1f}° (s³/V = {best_metric:.4f})")
    
    # Narrow beta range around best
    valid_betas = [r['beta'] for r in phase1_results if r['metric'] is not None]
    if len(valid_betas) >= 3:
        valid_betas.sort(key=lambda b: next(r['metric'] for r in phase1_results if r['beta'] == b))
        beta_window = abs(valid_betas[0] - valid_betas[min(2, len(valid_betas)-1)])
        beta_lo = max(beta_min, best_beta - beta_window)
        beta_hi = min(beta_max, best_beta + beta_window)
    else:
        beta_lo, beta_hi = best_beta - 5, best_beta + 5
    
    # ========================================
    # Phase 2: Scan c/a ratio at best beta
    # ========================================
    report(f"Phase 2: Scanning c/a ratio ({c_ratio_min}-{c_ratio_max}) at beta={best_beta:.1f}°...")
    
    ca_values = np.linspace(c_ratio_min, c_ratio_max, n_ca_coarse)
    phase2_results = []
    
    for ca in ca_values:
        s_star, vol, metric = evaluate_params(best_beta, c_ratio=ca, b_ratio=1.0)
        result = {
            'beta': best_beta,
            'c_ratio': float(ca),
            'b_ratio': 1.0,
            's_star': s_star,
            'volume': vol,
            'metric': metric
        }
        phase2_results.append(result)
        all_results.append(result)
        
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = result
    
    best_ca = best_result['c_ratio']
    report(f"Phase 2: Best c/a = {best_ca:.3f} (s³/V = {best_metric:.4f})")
    
    # Narrow c/a range
    valid_cas = [r['c_ratio'] for r in phase2_results if r['metric'] is not None]
    if len(valid_cas) >= 3:
        valid_cas.sort(key=lambda c: next(r['metric'] for r in phase2_results if r['c_ratio'] == c))
        ca_window = abs(valid_cas[0] - valid_cas[min(2, len(valid_cas)-1)])
        ca_lo = max(c_ratio_min, best_ca - ca_window)
        ca_hi = min(c_ratio_max, best_ca + ca_window)
    else:
        ca_lo, ca_hi = max(c_ratio_min, best_ca - 0.3), min(c_ratio_max, best_ca + 0.3)
    
    # ========================================
    # Phase 3: Scan b/a ratio at best (beta, c/a)
    # ========================================
    report(f"Phase 3: Scanning b/a ratio ({b_ratio_min}-{b_ratio_max}) at beta={best_beta:.1f}°, c/a={best_ca:.3f}...")
    
    ba_values = np.linspace(b_ratio_min, b_ratio_max, n_ba_fine)
    phase3_results = []
    
    for ba in ba_values:
        s_star, vol, metric = evaluate_params(best_beta, c_ratio=best_ca, b_ratio=ba)
        result = {
            'beta': best_beta,
            'c_ratio': best_ca,
            'b_ratio': float(ba),
            's_star': s_star,
            'volume': vol,
            'metric': metric
        }
        phase3_results.append(result)
        all_results.append(result)
        
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = result
    
    best_ba = best_result['b_ratio']
    report(f"Phase 3: Best b/a = {best_ba:.3f} (s³/V = {best_metric:.4f})")
    
    # ========================================
    # Phase 4: Local refinement around best point
    # ========================================
    report(f"Phase 4: Refining around (beta={best_result['beta']:.1f}°, c/a={best_result['c_ratio']:.3f}, b/a={best_result['b_ratio']:.3f})...")
    
    # Refine beta
    beta_refine = np.linspace(
        max(beta_min, best_result['beta'] - 3),
        min(beta_max, best_result['beta'] + 3),
        n_refine
    )
    for beta in beta_refine:
        s_star, vol, metric = evaluate_params(beta, best_result['c_ratio'], best_result['b_ratio'])
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = {
                'beta': float(beta),
                'c_ratio': best_result['c_ratio'],
                'b_ratio': best_result['b_ratio'],
                's_star': s_star,
                'volume': vol,
                'metric': metric
            }
    
    # Refine c/a
    ca_refine = np.linspace(
        max(c_ratio_min, best_result['c_ratio'] - 0.15),
        min(c_ratio_max, best_result['c_ratio'] + 0.15),
        n_refine
    )
    for ca in ca_refine:
        s_star, vol, metric = evaluate_params(best_result['beta'], ca, best_result['b_ratio'])
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = {
                'beta': best_result['beta'],
                'c_ratio': float(ca),
                'b_ratio': best_result['b_ratio'],
                's_star': s_star,
                'volume': vol,
                'metric': metric
            }
    
    # Refine b/a
    ba_refine = np.linspace(
        max(b_ratio_min, best_result['b_ratio'] - 0.1),
        min(b_ratio_max, best_result['b_ratio'] + 0.1),
        n_refine
    )
    for ba in ba_refine:
        s_star, vol, metric = evaluate_params(best_result['beta'], best_result['c_ratio'], ba)
        if metric is not None and metric < best_metric:
            best_metric = metric
            best_result = {
                'beta': best_result['beta'],
                'c_ratio': best_result['c_ratio'],
                'b_ratio': float(ba),
                's_star': s_star,
                'volume': vol,
                'metric': metric
            }
    
    report(f"Phase 4: Final result - beta={best_result['beta']:.1f}°, c/a={best_result['c_ratio']:.3f}, b/a={best_result['b_ratio']:.3f}")
    
    return {
        'best': best_result,
        'all_results': all_results,
        'bravais_type': bravais_type,
        'phases': {
            'phase1_beta': phase1_results,
            'phase2_ca': phase2_results,
            'phase3_ba': phase3_results
        }
    }


def scan_monoclinic_for_stoichiometry(
    config: Dict,
    metals: List[Dict],
    anion_symbol: str,
    anion_radius: float,
    anion_charge: int,
    target_cn: int,
    expected_metal_counts: Dict[str, int],
    expected_anion_count: int,
    num_metals: int,
    beta_min: float = 85.0,
    beta_max: float = 125.0,
    n_beta: int = 9,
    n_ca: int = 11,
    n_ba: int = 7,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Optional[Dict]:
    """
    Scan monoclinic parameter space to find structures matching target stoichiometry.
    
    Similar to run_ca_scan_for_stoichiometry but for monoclinic with 3 parameters.
    """
    from position_calculator import (
        calculate_stoichiometry_for_config,
        check_stoichiometry_match
    )
    
    coord_radii = tuple(m['radius'] + anion_radius for m in metals)
    if len(coord_radii) == 1:
        coord_radii = coord_radii[0]
    
    def report(msg: str):
        if progress_callback:
            progress_callback(msg)
    
    def evaluate_and_check_stoich(beta: float, c_ratio: float, b_ratio: float) -> Optional[Dict]:
        """Evaluate structure and check stoichiometry match."""
        # First get s* for this configuration
        params = {
            'a': 1.0,
            'b_ratio': float(b_ratio),
            'c_ratio': float(c_ratio),
            'alpha': 90.0,
            'beta': float(beta),
            'gamma': 90.0
        }
        p = LatticeParams(**params)
        sub = Sublattice(
            name='M',
            offsets=tuple(tuple(o) for o in config['offsets']),
            alpha_ratio=coord_radii,
            bravais_type=config['bravais_type']
        )
        s_star = find_threshold_s_for_N([sub], p, int(target_cn))
        if s_star is None:
            return None
        
        # Calculate stoichiometry
        stoich = calculate_stoichiometry_for_config(
            config_id=config['id'],
            offsets=config['offsets'],
            bravais_type=config['bravais_type'],
            lattice_type='Monoclinic',
            metals=metals,
            anion_symbol=anion_symbol,
            scale_s=s_star,
            target_cn=target_cn,
            anion_radius=anion_radius,
            c_ratio=c_ratio,
            b_ratio=b_ratio,
            beta=beta
        )
        
        if not stoich.success:
            return None
        
        # Check match
        matches, match_type = check_stoichiometry_match(
            stoich.metal_counts,
            stoich.anion_count,
            expected_metal_counts,
            expected_anion_count
        )
        
        if not matches:
            return None
        
        # Compute volume and metric
        b = s_star * b_ratio
        c = s_star * c_ratio
        volume = s_star * b * c * np.sin(np.radians(beta))
        metric = (s_star ** 3) / volume if volume > 0 else float('inf')
        
        return {
            'beta': float(beta),
            'c_ratio': float(c_ratio),
            'b_ratio': float(b_ratio),
            's_star': float(s_star),
            'volume': float(volume),
            'metric': float(metric),
            'match_type': match_type,
            'stoich_result': stoich,
            'regularity': stoich.regularity if hasattr(stoich, 'regularity') else 0.0
        }
    
    best_result = None
    best_metric = float('inf')
    all_matches = []
    
    # Phase 1: Scan beta with a=b=c
    report(f"Monoclinic Phase 1: Scanning beta ({beta_min}°-{beta_max}°)...")
    beta_values = np.linspace(beta_min, beta_max, n_beta)
    
    for beta in beta_values:
        result = evaluate_and_check_stoich(beta, c_ratio=1.0, b_ratio=1.0)
        if result is not None:
            all_matches.append(result)
            if result['metric'] < best_metric:
                best_metric = result['metric']
                best_result = result
    
    if best_result is None:
        # No matches at a=b=c, try scanning c/a at different betas
        report("Phase 1: No matches at a=b=c, expanding search...")
        for beta in beta_values[::2]:  # Every other beta
            for ca in np.linspace(0.6, 2.0, 8):
                result = evaluate_and_check_stoich(beta, c_ratio=ca, b_ratio=1.0)
                if result is not None:
                    all_matches.append(result)
                    if result['metric'] < best_metric:
                        best_metric = result['metric']
                        best_result = result
    
    if best_result is None:
        return None
    
    best_beta = best_result['beta']
    report(f"Phase 1: Found match at beta={best_beta:.1f}°")
    
    # Phase 2: Scan c/a at best beta
    report(f"Monoclinic Phase 2: Scanning c/a at beta={best_beta:.1f}°...")
    ca_values = np.linspace(0.5, 2.5, n_ca)
    
    for ca in ca_values:
        result = evaluate_and_check_stoich(best_beta, c_ratio=ca, b_ratio=1.0)
        if result is not None:
            all_matches.append(result)
            if result['metric'] < best_metric:
                best_metric = result['metric']
                best_result = result
    
    best_ca = best_result['c_ratio']
    report(f"Phase 2: Best c/a={best_ca:.3f}")
    
    # Phase 3: Scan b/a at best (beta, c/a)
    report(f"Monoclinic Phase 3: Scanning b/a at beta={best_beta:.1f}°, c/a={best_ca:.3f}...")
    ba_values = np.linspace(0.7, 1.4, n_ba)
    
    for ba in ba_values:
        result = evaluate_and_check_stoich(best_beta, c_ratio=best_ca, b_ratio=ba)
        if result is not None:
            all_matches.append(result)
            if result['metric'] < best_metric:
                best_metric = result['metric']
                best_result = result
    
    best_ba = best_result['b_ratio']
    report(f"Phase 3: Best b/a={best_ba:.3f}")
    
    # Phase 4: Local refinement
    report("Monoclinic Phase 4: Refining...")
    for beta in np.linspace(best_beta - 2, best_beta + 2, 5):
        for ca in np.linspace(best_ca - 0.1, best_ca + 0.1, 3):
            for ba in np.linspace(best_ba - 0.05, best_ba + 0.05, 3):
                result = evaluate_and_check_stoich(beta, c_ratio=ca, b_ratio=ba)
                if result is not None:
                    all_matches.append(result)
                    if result['metric'] < best_metric:
                        best_metric = result['metric']
                        best_result = result
    
    report(f"Final: beta={best_result['beta']:.1f}°, c/a={best_result['c_ratio']:.3f}, b/a={best_result['b_ratio']:.3f}")
    
    return {
        'best': best_result,
        'all_matches': all_matches,
        's_star': best_result['s_star'],
        'beta': best_result['beta'],
        'c_ratio': best_result['c_ratio'],
        'b_ratio': best_result['b_ratio'],
        'match_type': best_result['match_type'],
        'stoich_result': best_result['stoich_result'],
        'regularity': best_result.get('regularity', 0.0)
    }
