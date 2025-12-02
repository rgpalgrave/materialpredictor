"""
Position Calculator for Crystal Coordination Analysis
Calculates exact positions of metal atoms and intersection sites in a unit cell.

Key features:
- Correct handling of translational symmetry (PBC)
- Sites on unit cell boundaries appear at all equivalent positions
- Export to JSON, CSV, XYZ formats
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import numpy as np
from interstitial_engine import (
    LatticeParams,
    Sublattice,
    lattice_vectors,
    max_multiplicity_for_scale,
    BRAVAIS_BASIS,
    generate_shifts,
)


# -----------------
# Coordinate conversion utilities
# -----------------

def frac_to_cart(frac: np.ndarray, lat_vecs: np.ndarray) -> np.ndarray:
    """Convert fractional to Cartesian coordinates."""
    return frac @ lat_vecs


def cart_to_frac(cart: np.ndarray, lat_vecs: np.ndarray) -> np.ndarray:
    """Convert Cartesian to fractional coordinates."""
    inv = np.linalg.inv(lat_vecs)
    if cart.ndim == 1:
        return cart @ inv
    return cart @ inv


def wrap_to_unit_cell(frac: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """
    Wrap fractional coordinates to [0, 1).
    Values very close to 0 or 1 are snapped.
    """
    wrapped = frac - np.floor(frac)
    # Snap values very close to 0 or 1
    wrapped = np.where(np.abs(wrapped) < tol, 0.0, wrapped)
    wrapped = np.where(np.abs(wrapped - 1.0) < tol, 0.0, wrapped)
    return wrapped


def is_on_boundary(frac: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Check which coordinates are on unit cell boundaries (close to 0).
    Returns boolean array of shape (N, 3) indicating boundary positions.
    """
    if frac.ndim == 1:
        frac = frac.reshape(1, -1)
    return np.abs(frac) < tol


def generate_boundary_equivalents(frac: np.ndarray, tol: float = 1e-6) -> List[np.ndarray]:
    """
    Generate all equivalent positions for a site considering unit cell boundaries.
    
    If a site is at (0, 0.5, 0), it should also appear at (1, 0.5, 0), (0, 0.5, 1), (1, 0.5, 1).
    
    Args:
        frac: Single fractional coordinate (3,)
        tol: Tolerance for boundary detection
        
    Returns:
        List of all equivalent fractional positions
    """
    on_boundary = is_on_boundary(frac.reshape(1, -1), tol)[0]
    
    # Generate all combinations of 0 and 1 for boundary coordinates
    equivalents = []
    n_boundaries = np.sum(on_boundary)
    
    if n_boundaries == 0:
        return [frac.copy()]
    
    # Generate 2^n combinations
    for i in range(2 ** n_boundaries):
        new_pos = frac.copy()
        bit_idx = 0
        for dim in range(3):
            if on_boundary[dim]:
                # Use bit i to decide if this should be 0 or 1
                if (i >> bit_idx) & 1:
                    new_pos[dim] = 1.0
                else:
                    new_pos[dim] = 0.0
                bit_idx += 1
        equivalents.append(new_pos)
    
    return equivalents


# -----------------
# Metal atom position generation
# -----------------

@dataclass
class MetalAtomData:
    """Complete information about metal atoms in the structure"""
    fractional: np.ndarray      # (N, 3) fractional coordinates
    cartesian: np.ndarray       # (N, 3) Cartesian coordinates
    sublattice_id: np.ndarray   # (N,) which sublattice each atom belongs to
    sublattice_name: List[str]  # Names of sublattices
    radius: np.ndarray          # (N,) actual radius in Angstroms
    alpha_ratio: np.ndarray     # (N,) alpha ratio for each atom
    offset_idx: np.ndarray      # (N,) which offset within sublattice (for multi-metal)


def generate_metal_positions(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    include_boundary_equivalents: bool = True
) -> MetalAtomData:
    """
    Generate metal atom positions in one unit cell.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value (sphere radius parameter)
        include_boundary_equivalents: Include equivalent positions at cell boundaries
    
    Returns:
        MetalAtomData with all metal atom information
    """
    lat_vecs = lattice_vectors(p)
    
    all_frac = []
    all_cart = []
    all_sublattice_id = []
    all_sublattice_names = []
    all_radius = []
    all_alpha = []
    all_offset_idx = []
    
    for sub_idx, sub in enumerate(sublattices):
        # Get basis positions for this Bravais lattice type
        basis = BRAVAIS_BASIS.get(sub.bravais_type, [(0, 0, 0)])
        
        # Get all offsets (default to origin if none specified)
        offsets = sub.offsets if sub.offsets and len(sub.offsets) > 0 else [(0, 0, 0)]
        
        # Loop over all offsets AND all basis positions
        for offset_idx, offset in enumerate(offsets):
            offset_arr = np.array(offset, dtype=float)
            # Get alpha for this specific offset
            alpha = sub.get_alpha_for_offset(offset_idx)
            
            for basis_pos in basis:
                # Fractional position
                frac = np.array(basis_pos, dtype=float) + offset_arr
                frac = wrap_to_unit_cell(frac)
                
                # Generate boundary equivalents if requested
                if include_boundary_equivalents:
                    positions = generate_boundary_equivalents(frac)
                else:
                    positions = [frac]
                
                for pos in positions:
                    # Cartesian position
                    cart = frac_to_cart(pos, lat_vecs)
                    
                    all_frac.append(pos)
                    all_cart.append(cart)
                    all_sublattice_id.append(sub_idx)
                    all_sublattice_names.append(sub.name)
                    all_radius.append(alpha * scale_s * p.a)
                    all_alpha.append(alpha)
                    all_offset_idx.append(offset_idx)
    
    if not all_frac:
        return MetalAtomData(
            fractional=np.empty((0, 3)),
            cartesian=np.empty((0, 3)),
            sublattice_id=np.empty((0,), dtype=int),
            sublattice_name=[],
            radius=np.empty((0,)),
            alpha_ratio=np.empty((0,)),
            offset_idx=np.empty((0,), dtype=int)
        )
    
    return MetalAtomData(
        fractional=np.array(all_frac),
        cartesian=np.array(all_cart),
        sublattice_id=np.array(all_sublattice_id, dtype=int),
        sublattice_name=all_sublattice_names,
        radius=np.array(all_radius),
        alpha_ratio=np.array(all_alpha),
        offset_idx=np.array(all_offset_idx, dtype=int)
    )


# -----------------
# Intersection position calculation
# -----------------

@dataclass
class IntersectionData:
    """Complete information about intersection sites"""
    fractional: np.ndarray          # (N, 3) fractional coordinates
    cartesian: np.ndarray           # (N, 3) Cartesian coordinates
    multiplicity: np.ndarray        # (N,) number of spheres intersecting
    contributing_atoms: List[List[int]]  # Indices of atoms forming each intersection


def cluster_intersections_pbc(
    positions_frac: np.ndarray,
    multiplicities: np.ndarray,
    eps_frac: float = 0.02
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster intersection positions accounting for PBC.
    
    Args:
        positions_frac: Fractional coordinates (N, 3)
        multiplicities: Multiplicity values (N,)
        eps_frac: Clustering threshold in fractional coordinates
    
    Returns:
        unique_positions: Fractional coordinates of cluster centers (wrapped to [0,1))
        unique_multiplicities: Maximum multiplicity in each cluster
    """
    if len(positions_frac) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    
    # Wrap to unit cell
    wrapped = wrap_to_unit_cell(positions_frac)
    
    used = np.zeros(len(wrapped), dtype=bool)
    unique_pos = []
    unique_mult = []
    
    eps2 = eps_frac ** 2
    
    for i in range(len(wrapped)):
        if used[i]:
            continue
        
        # Find all positions within eps of this one (considering PBC)
        cluster = [i]
        for j in range(i + 1, len(wrapped)):
            if used[j]:
                continue
            
            # Compute distance considering PBC wrapping
            diff = wrapped[j] - wrapped[i]
            # Handle periodic wrapping
            diff = diff - np.round(diff)
            dist2 = np.sum(diff ** 2)
            
            if dist2 < eps2:
                cluster.append(j)
        
        # Mark as used
        used[cluster] = True
        
        # Compute cluster representative
        cluster_positions = wrapped[cluster]
        
        # Handle PBC in averaging: unwrap relative to first point
        ref = cluster_positions[0]
        unwrapped = cluster_positions.copy()
        for k in range(1, len(unwrapped)):
            diff = unwrapped[k] - ref
            diff = diff - np.round(diff)
            unwrapped[k] = ref + diff
        
        # Average and wrap back
        mean_pos = np.mean(unwrapped, axis=0)
        mean_pos = wrap_to_unit_cell(mean_pos)
        
        # Snap to high-symmetry positions if close
        mean_pos = snap_to_symmetry(mean_pos)
        
        # Maximum multiplicity in cluster
        max_mult = int(np.max(multiplicities[cluster]))
        
        unique_pos.append(mean_pos)
        unique_mult.append(max_mult)
    
    return np.array(unique_pos), np.array(unique_mult, dtype=int)


def snap_to_symmetry(frac: np.ndarray, tol: float = 0.02) -> np.ndarray:
    """
    Snap fractional coordinates to common high-symmetry values.
    
    Common values: 0, 0.25, 0.333, 0.5, 0.667, 0.75, 1
    """
    symmetry_values = [0.0, 0.25, 1/3, 0.5, 2/3, 0.75, 1.0]
    
    result = frac.copy()
    for i in range(3):
        for sym_val in symmetry_values:
            if abs(result[i] - sym_val) < tol:
                result[i] = sym_val
                break
    
    # Wrap 1.0 to 0.0
    result = np.where(np.abs(result - 1.0) < 1e-9, 0.0, result)
    
    return result


def calculate_intersections(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    target_N: Optional[int] = None,
    k_samples: int = 24,
    cluster_eps_frac: float = 0.02,
    include_boundary_equivalents: bool = True
) -> IntersectionData:
    """
    Calculate intersection positions in one unit cell.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value
        target_N: Optional minimum multiplicity filter
        k_samples: Number of samples per sphere pair
        cluster_eps_frac: Clustering epsilon in fractional coordinates
        include_boundary_equivalents: Include equivalent positions at cell boundaries
    
    Returns:
        IntersectionData with all intersection information
    """
    lat_vecs = lattice_vectors(p)
    
    # Use existing engine to get raw intersection samples
    max_mult, sample_positions, sample_counts = max_multiplicity_for_scale(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        k_samples=k_samples,
        tol_inside=1e-3,
        early_stop_at=None
    )
    
    if len(sample_positions) == 0:
        return IntersectionData(
            fractional=np.empty((0, 3)),
            cartesian=np.empty((0, 3)),
            multiplicity=np.empty((0,), dtype=int),
            contributing_atoms=[]
        )
    
    # Convert to fractional coordinates
    frac_positions = cart_to_frac(sample_positions, lat_vecs)
    
    # Cluster in fractional space with PBC
    unique_frac, unique_mult = cluster_intersections_pbc(
        frac_positions,
        sample_counts,
        eps_frac=cluster_eps_frac
    )
    
    # Filter by target multiplicity if specified
    if target_N is not None:
        mask = unique_mult >= target_N
        unique_frac = unique_frac[mask]
        unique_mult = unique_mult[mask]
    
    # Filter to unit cell [0, 1)
    mask = np.all((unique_frac >= 0) & (unique_frac < 1.0 - 1e-9), axis=1)
    unique_frac = unique_frac[mask]
    unique_mult = unique_mult[mask]
    
    # Now expand boundary equivalents for visualization
    final_frac = []
    final_mult = []
    
    for i in range(len(unique_frac)):
        if include_boundary_equivalents:
            equivalents = generate_boundary_equivalents(unique_frac[i])
        else:
            equivalents = [unique_frac[i]]
        
        for eq in equivalents:
            final_frac.append(eq)
            final_mult.append(unique_mult[i])
    
    if len(final_frac) == 0:
        return IntersectionData(
            fractional=np.empty((0, 3)),
            cartesian=np.empty((0, 3)),
            multiplicity=np.empty((0,), dtype=int),
            contributing_atoms=[]
        )
    
    final_frac = np.array(final_frac)
    final_mult = np.array(final_mult, dtype=int)
    
    # Convert to Cartesian
    final_cart = np.array([frac_to_cart(f, lat_vecs) for f in final_frac])
    
    # Contributing atoms - compute for unique sites
    contributing = identify_contributing_atoms(
        final_cart, sublattices, p, scale_s
    )
    
    return IntersectionData(
        fractional=final_frac,
        cartesian=final_cart,
        multiplicity=final_mult,
        contributing_atoms=contributing
    )


def identify_contributing_atoms(
    intersection_points: np.ndarray,
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    tol: float = 0.05
) -> List[List[int]]:
    """
    Identify which metal atoms contribute to each intersection.
    """
    if len(intersection_points) == 0:
        return []
    
    lat_vecs = lattice_vectors(p)
    shifts = generate_shifts(lat_vecs)
    
    # Generate metal positions (without boundary equivalents for cleaner indexing)
    metals = generate_metal_positions(sublattices, p, scale_s, include_boundary_equivalents=False)
    
    if len(metals.cartesian) == 0:
        return [[] for _ in range(len(intersection_points))]
    
    contributing = []
    for pt in intersection_points:
        atoms = []
        for atom_idx in range(len(metals.cartesian)):
            center = metals.cartesian[atom_idx]
            radius = metals.radius[atom_idx]
            
            # Check all periodic images
            for shift in shifts:
                dist = np.linalg.norm(pt - (center + shift))
                if abs(dist - radius) < tol * radius:
                    if atom_idx not in atoms:
                        atoms.append(atom_idx)
                    break
        
        contributing.append(atoms)
    
    return contributing


# -----------------
# Complete structure calculation
# -----------------

@dataclass
class CompleteStructureData:
    """Complete structure with metal atoms and intersections"""
    metal_atoms: MetalAtomData
    intersections: IntersectionData
    lattice_params: LatticeParams
    scale_s: float
    lattice_vectors: np.ndarray


def calculate_complete_structure(
    sublattices: List[Sublattice],
    p: LatticeParams,
    scale_s: float,
    target_N: Optional[int] = None,
    k_samples: int = 24,
    cluster_eps_frac: float = 0.05,
    include_boundary_equivalents: bool = True
) -> CompleteStructureData:
    """
    Calculate complete structure: metal atoms + intersections for one unit cell.
    
    Args:
        sublattices: List of sublattice definitions
        p: Lattice parameters
        scale_s: Current s value
        target_N: Optional minimum multiplicity for intersections
        k_samples: Sampling density for intersections
        cluster_eps_frac: Clustering epsilon in fractional coords (default 0.05)
        include_boundary_equivalents: Show equivalent positions at cell boundaries
    
    Returns:
        CompleteStructureData with all structural information
    """
    # Generate metal positions
    metal_atoms = generate_metal_positions(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        include_boundary_equivalents=include_boundary_equivalents
    )
    
    # Calculate intersections
    intersections = calculate_intersections(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        target_N=target_N,
        k_samples=k_samples,
        cluster_eps_frac=cluster_eps_frac,
        include_boundary_equivalents=include_boundary_equivalents
    )
    
    # Get lattice vectors
    vecs = lattice_vectors(p)
    
    return CompleteStructureData(
        metal_atoms=metal_atoms,
        intersections=intersections,
        lattice_params=p,
        scale_s=scale_s,
        lattice_vectors=vecs
    )


# -----------------
# Export utilities
# -----------------

def format_position_dict(data: CompleteStructureData) -> Dict:
    """Format structure data as dictionary for JSON export."""
    p = data.lattice_params
    lat_vecs = data.lattice_vectors
    
    result = {
        'lattice_parameters': {
            'a': float(p.a),
            'b': float(p.a * p.b_ratio),
            'c': float(p.a * p.c_ratio),
            'alpha': float(p.alpha),
            'beta': float(p.beta),
            'gamma': float(p.gamma),
            'b_ratio': float(p.b_ratio),
            'c_ratio': float(p.c_ratio),
        },
        'lattice_vectors': {
            'a': lat_vecs[0].tolist(),
            'b': lat_vecs[1].tolist(),
            'c': lat_vecs[2].tolist(),
        },
        'scale_s': float(data.scale_s),
        'metal_atoms': {
            'count': len(data.metal_atoms.fractional),
            'positions': {
                'fractional': data.metal_atoms.fractional.tolist() if len(data.metal_atoms.fractional) > 0 else [],
                'cartesian': data.metal_atoms.cartesian.tolist() if len(data.metal_atoms.cartesian) > 0 else [],
            },
            'sublattice_id': data.metal_atoms.sublattice_id.tolist() if len(data.metal_atoms.sublattice_id) > 0 else [],
            'sublattice_name': data.metal_atoms.sublattice_name,
            'radius_angstrom': data.metal_atoms.radius.tolist() if len(data.metal_atoms.radius) > 0 else [],
            'alpha_ratio': data.metal_atoms.alpha_ratio.tolist() if len(data.metal_atoms.alpha_ratio) > 0 else [],
        },
        'intersections': {
            'count': len(data.intersections.fractional),
            'positions': {
                'fractional': data.intersections.fractional.tolist() if len(data.intersections.fractional) > 0 else [],
                'cartesian': data.intersections.cartesian.tolist() if len(data.intersections.cartesian) > 0 else [],
            },
            'multiplicity': data.intersections.multiplicity.tolist() if len(data.intersections.multiplicity) > 0 else [],
            'contributing_atoms': data.intersections.contributing_atoms,
        }
    }
    
    return result


def format_metal_atoms_csv(data: CompleteStructureData) -> str:
    """Format metal atoms as CSV string."""
    lines = []
    lines.append("atom_index,sublattice_name,sublattice_id,frac_x,frac_y,frac_z,cart_x,cart_y,cart_z,radius_angstrom,alpha_ratio")
    
    for i in range(len(data.metal_atoms.fractional)):
        frac = data.metal_atoms.fractional[i]
        cart = data.metal_atoms.cartesian[i]
        lines.append(
            f"{i},{data.metal_atoms.sublattice_name[i]},{data.metal_atoms.sublattice_id[i]},"
            f"{frac[0]:.6f},{frac[1]:.6f},{frac[2]:.6f},"
            f"{cart[0]:.6f},{cart[1]:.6f},{cart[2]:.6f},"
            f"{data.metal_atoms.radius[i]:.6f},{data.metal_atoms.alpha_ratio[i]:.6f}"
        )
    
    return "\n".join(lines)


def format_intersections_csv(data: CompleteStructureData) -> str:
    """Format intersections as CSV string."""
    lines = []
    lines.append("intersection_index,multiplicity,frac_x,frac_y,frac_z,cart_x,cart_y,cart_z,contributing_atom_indices")
    
    for i in range(len(data.intersections.fractional)):
        frac = data.intersections.fractional[i]
        cart = data.intersections.cartesian[i]
        mult = data.intersections.multiplicity[i]
        contrib = ";".join(map(str, data.intersections.contributing_atoms[i]))
        
        lines.append(
            f"{i},{mult},"
            f"{frac[0]:.6f},{frac[1]:.6f},{frac[2]:.6f},"
            f"{cart[0]:.6f},{cart[1]:.6f},{cart[2]:.6f},"
            f"{contrib}"
        )
    
    return "\n".join(lines)


def format_xyz(data: CompleteStructureData, include_intersections: bool = True) -> str:
    """Format structure as XYZ file for visualization."""
    lines = []
    
    # Count atoms
    n_atoms = len(data.metal_atoms.cartesian)
    n_intersections = len(data.intersections.cartesian) if include_intersections else 0
    total = n_atoms + n_intersections
    
    lines.append(str(total))
    lines.append(f"s={data.scale_s:.6f} a={data.lattice_params.a:.6f}")
    
    # Metal atoms - use element symbol from sublattice name
    for i in range(len(data.metal_atoms.cartesian)):
        cart = data.metal_atoms.cartesian[i]
        name = data.metal_atoms.sublattice_name[i]
        # Use first 2 characters as element symbol
        symbol = name[:2] if len(name) >= 2 else name
        lines.append(f"{symbol}  {cart[0]:.6f}  {cart[1]:.6f}  {cart[2]:.6f}")
    
    # Intersections (as X with multiplicity)
    if include_intersections:
        for i in range(len(data.intersections.cartesian)):
            cart = data.intersections.cartesian[i]
            mult = data.intersections.multiplicity[i]
            lines.append(f"X{mult}  {cart[0]:.6f}  {cart[1]:.6f}  {cart[2]:.6f}")
    
    return "\n".join(lines)


def get_unique_intersections(data: IntersectionData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get unique intersection sites (excluding boundary duplicates).
    
    Returns only sites in [0, 1) for each coordinate.
    """
    if len(data.fractional) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=int)
    
    # Filter to sites strictly in [0, 1)
    mask = np.all((data.fractional >= 0) & (data.fractional < 1.0 - 1e-6), axis=1)
    
    return data.fractional[mask], data.multiplicity[mask]


def calculate_site_weight(frac_coord: np.ndarray, tol: float = 1e-6) -> float:
    """
    Calculate the fractional weight of a site based on its position.
    
    Sites on boundaries are shared between unit cells:
    - Interior (no coords at 0 or 1): weight = 1
    - Face (1 coord at 0 or 1): weight = 1/2
    - Edge (2 coords at 0 or 1): weight = 1/4
    - Corner (3 coords at 0 or 1): weight = 1/8
    
    Args:
        frac_coord: Fractional coordinate (3,)
        tol: Tolerance for detecting boundary positions
    
    Returns:
        Fractional weight (1, 0.5, 0.25, or 0.125)
    """
    # Count how many coordinates are on boundaries (0 or 1)
    on_boundary = 0
    for x in frac_coord:
        if abs(x) < tol or abs(x - 1.0) < tol:
            on_boundary += 1
    
    # Weight is 1/2^(number of boundary coordinates)
    return 1.0 / (2 ** on_boundary)


def calculate_weighted_counts(structure: CompleteStructureData, tol: float = 1e-6) -> dict:
    """
    Calculate weighted atom and site counts for correct stoichiometry.
    
    Sites on boundaries are weighted fractionally:
    - Interior: weight 1
    - Face (1 coord at 0 or 1): weight 1/2  
    - Edge (2 coords at 0 or 1): weight 1/4
    - Corner (3 coords at 0 or 1): weight 1/8
    
    Args:
        structure: Complete structure data
        tol: Tolerance for boundary detection
    
    Returns:
        Dictionary with:
        - metal_count: Weighted count of metal atoms per unit cell
        - intersection_count: Weighted count of intersection sites per unit cell
        - metal_weights: Array of individual weights for each metal atom
        - intersection_weights: Array of individual weights for each intersection
    """
    # Calculate metal weights - sum over all positions including boundary equivalents
    metal_weights = np.array([
        calculate_site_weight(frac, tol) 
        for frac in structure.metal_atoms.fractional
    ])
    
    # Calculate intersection weights - sum over all positions including boundary equivalents
    # This gives correct stoichiometry (e.g., 4 octahedral sites in FCC)
    intersection_weights = np.array([
        calculate_site_weight(frac, tol) 
        for frac in structure.intersections.fractional
    ])
    
    # Also compute unique site mask for reference
    unique_mask = np.all(
        (structure.intersections.fractional >= 0) & 
        (structure.intersections.fractional < 1.0 - tol), 
        axis=1
    )
    
    return {
        'metal_count': float(np.sum(metal_weights)),
        'intersection_count': float(np.sum(intersection_weights)),
        'metal_weights': metal_weights,
        'intersection_weights': intersection_weights,
        'unique_intersection_mask': unique_mask
    }


# -----------------
# Stoichiometry calculation
# -----------------

@dataclass
class StoichiometryResult:
    """Result of stoichiometry calculation for a configuration."""
    config_id: str
    metal_counts: Dict[str, float]  # Symbol -> weighted count per unit cell
    anion_count: float  # Weighted intersection count per unit cell
    formula: str  # e.g., "LaAl₂O₄"
    ratio_formula: str  # Simplified ratio, e.g., "1:2:4"
    success: bool
    error: Optional[str] = None


def calculate_stoichiometry_for_config(
    config_id: str,
    offsets: List[Tuple[float, float, float]],
    bravais_type: str,
    lattice_type: str,
    metals: List[Dict],  # List of {'symbol': str, 'radius': float, ...}
    anion_symbol: str,
    scale_s: float,
    target_cn: int,
    base_alpha: float = 0.5,
    cluster_eps_frac: float = 0.05
) -> StoichiometryResult:
    """
    Calculate stoichiometry for a single configuration.
    
    Args:
        config_id: Configuration identifier
        offsets: List of fractional coordinate offsets for metal positions
        bravais_type: Bravais lattice type (e.g., 'cubic_F')
        lattice_type: Lattice system (e.g., 'Cubic', 'Tetragonal')
        metals: List of metal definitions with 'symbol' and 'radius'
        anion_symbol: Anion symbol (e.g., 'O')
        scale_s: Scale factor to use (typically s*)
        target_cn: Target coordination number for filtering intersections
        base_alpha: Base alpha ratio
        cluster_eps_frac: Clustering tolerance
    
    Returns:
        StoichiometryResult with formula and counts
    """
    try:
        # Compute per-offset alpha ratios based on metal radii
        metal_radii = [m['radius'] for m in metals]
        if len(metal_radii) > 1:
            max_radius = max(metal_radii)
            alpha_ratios = tuple(base_alpha * (r / max_radius) for r in metal_radii)
        else:
            alpha_ratios = base_alpha
        
        # Set up lattice parameters
        p_dict = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
                  'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
        
        if lattice_type == 'Hexagonal':
            p_dict['gamma'] = 120.0
            p_dict['c_ratio'] = 1.633
        elif lattice_type == 'Rhombohedral':
            p_dict['alpha'] = p_dict['beta'] = p_dict['gamma'] = 80.0
        elif lattice_type == 'Monoclinic':
            p_dict['beta'] = 100.0
        
        p = LatticeParams(**p_dict)
        
        # Create sublattice
        sublattice = Sublattice(
            name='M',
            offsets=tuple(tuple(o) for o in offsets),
            alpha_ratio=alpha_ratios,
            bravais_type=bravais_type
        )
        
        # Calculate structure
        structure = calculate_complete_structure(
            sublattices=[sublattice],
            p=p,
            scale_s=scale_s,
            target_N=target_cn,
            k_samples=24,
            cluster_eps_frac=cluster_eps_frac,
            include_boundary_equivalents=True
        )
        
        # Calculate weighted counts per metal type
        # We need to track which atoms came from which offset
        metal_counts = {}
        lat_vecs = lattice_vectors(p)
        basis = BRAVAIS_BASIS.get(bravais_type, [(0, 0, 0)])
        
        for offset_idx, offset in enumerate(offsets):
            # Get metal symbol for this offset
            if offset_idx < len(metals):
                symbol = metals[offset_idx]['symbol']
            else:
                symbol = f"M{offset_idx+1}"
            
            if symbol not in metal_counts:
                metal_counts[symbol] = 0.0
            
            offset_arr = np.array(offset, dtype=float)
            
            # Count atoms from this offset (with all basis positions)
            for basis_pos in basis:
                frac = np.array(basis_pos, dtype=float) + offset_arr
                frac = wrap_to_unit_cell(frac)
                
                # Generate boundary equivalents and sum weights
                equivalents = generate_boundary_equivalents(frac)
                for equiv in equivalents:
                    weight = calculate_site_weight(equiv)
                    metal_counts[symbol] += weight
        
        # Calculate anion count from intersections
        weighted = calculate_weighted_counts(structure)
        anion_count = weighted['intersection_count']
        
        # Build formula string
        formula = build_formula_string(metal_counts, anion_symbol, anion_count)
        ratio_formula = build_ratio_string(metal_counts, anion_count)
        
        return StoichiometryResult(
            config_id=config_id,
            metal_counts=metal_counts,
            anion_count=anion_count,
            formula=formula,
            ratio_formula=ratio_formula,
            success=True
        )
        
    except Exception as e:
        return StoichiometryResult(
            config_id=config_id,
            metal_counts={},
            anion_count=0.0,
            formula="",
            ratio_formula="",
            success=False,
            error=str(e)
        )


def build_formula_string(metal_counts: Dict[str, float], anion_symbol: str, anion_count: float) -> str:
    """
    Build a chemical formula string like "LaAl₂O₄".
    
    Subscript digits: ₀₁₂₃₄₅₆₇₈₉
    """
    subscript_map = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    
    # Find GCD to simplify if all are integers
    all_counts = list(metal_counts.values()) + [anion_count]
    
    # Check if we can express as simple integers
    # Try multiplying by small factors to get integers
    best_multiplier = 1
    for mult in [1, 2, 3, 4, 6, 8, 12]:
        scaled = [c * mult for c in all_counts]
        if all(abs(s - round(s)) < 0.05 for s in scaled):
            best_multiplier = mult
            break
    
    # Apply multiplier and round
    metal_counts_int = {k: round(v * best_multiplier) for k, v in metal_counts.items()}
    anion_count_int = round(anion_count * best_multiplier)
    
    # Find GCD of all counts
    from math import gcd
    from functools import reduce
    all_int_counts = list(metal_counts_int.values()) + [anion_count_int]
    all_int_counts = [c for c in all_int_counts if c > 0]
    
    if all_int_counts:
        common_divisor = reduce(gcd, all_int_counts)
        metal_counts_int = {k: v // common_divisor for k, v in metal_counts_int.items()}
        anion_count_int = anion_count_int // common_divisor
    
    # Build formula
    parts = []
    for symbol, count in metal_counts_int.items():
        if count == 1:
            parts.append(symbol)
        elif count > 0:
            parts.append(f"{symbol}{str(count).translate(subscript_map)}")
    
    if anion_count_int == 1:
        parts.append(anion_symbol)
    elif anion_count_int > 0:
        parts.append(f"{anion_symbol}{str(anion_count_int).translate(subscript_map)}")
    
    return ''.join(parts)


def build_ratio_string(metal_counts: Dict[str, float], anion_count: float) -> str:
    """Build a ratio string like "1:2:4"."""
    all_counts = list(metal_counts.values()) + [anion_count]
    
    # Try to express as simple integers
    best_multiplier = 1
    for mult in [1, 2, 3, 4, 6, 8, 12]:
        scaled = [c * mult for c in all_counts]
        if all(abs(s - round(s)) < 0.05 for s in scaled):
            best_multiplier = mult
            break
    
    scaled = [round(c * best_multiplier) for c in all_counts]
    
    # Simplify by GCD
    from math import gcd
    from functools import reduce
    non_zero = [s for s in scaled if s > 0]
    if non_zero:
        common_divisor = reduce(gcd, non_zero)
        scaled = [s // common_divisor for s in scaled]
    
    return ':'.join(str(s) for s in scaled)


# -----------------
# Stoichiometry-based c/a scanning
# -----------------

@dataclass
class StoichiometryScanResult:
    """Result of scanning c/a ratios for target stoichiometry."""
    config_id: str
    target_mx_ratio: float  # Target M/X ratio
    best_c_ratio: Optional[float]
    best_s_star: Optional[float]
    best_mx_ratio: Optional[float]  # Achieved M/X ratio
    best_mx_error: Optional[float]  # |achieved - target|
    matching_ranges: List[Tuple[float, float]]  # c/a ranges where stoichiometry matches
    scan_data: List[Tuple[float, float, float, float]]  # (c/a, s*, M/X, error)
    success: bool
    error: Optional[str] = None


def scan_ca_for_stoichiometry(
    config_id: str,
    offsets: List[Tuple[float, float, float]],
    bravais_type: str,
    lattice_type: str,
    metals: List[Dict],
    target_mx_ratio: float,
    target_cn: int,
    base_alpha: float = 0.5,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.0,
    n_points: int = 50,
    cluster_eps_frac: float = 0.05,
    tolerance: float = 0.1,
    check_half_filling: bool = True
) -> StoichiometryScanResult:
    """
    Scan c/a ratios to find regions where stoichiometry matches target.
    
    Args:
        config_id: Configuration identifier
        offsets: Metal position offsets
        bravais_type: Bravais lattice type
        lattice_type: Lattice system
        metals: List of metal definitions
        target_mx_ratio: Target M/X ratio (total metals / anions)
        target_cn: Target coordination number
        base_alpha: Base alpha ratio
        c_ratio_min/max: c/a scan range
        n_points: Number of scan points
        cluster_eps_frac: Clustering tolerance
        tolerance: Relative tolerance for M/X match
        check_half_filling: Also check if half the anions gives correct stoichiometry
    
    Returns:
        StoichiometryScanResult with scan data and matching ranges
    """
    try:
        from interstitial_engine import (
            LatticeParams, Sublattice, compute_min_scale_for_cn
        )
        
        # Compute per-offset alpha ratios
        metal_radii = [m['radius'] for m in metals]
        if len(metal_radii) > 1:
            max_radius = max(metal_radii)
            alpha_ratios = tuple(base_alpha * (r / max_radius) for r in metal_radii)
        else:
            alpha_ratios = base_alpha
        
        scan_data = []
        matching_ranges = []
        in_matching_range = False
        range_start = None
        
        best_c_ratio = None
        best_s_star = None
        best_mx_ratio = None
        best_mx_error = float('inf')
        
        c_ratios = np.linspace(c_ratio_min, c_ratio_max, n_points)
        
        for c_ratio in c_ratios:
            # Set up lattice parameters
            p_dict = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': c_ratio,
                      'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
            
            if lattice_type == 'Hexagonal':
                p_dict['gamma'] = 120.0
            
            p = LatticeParams(**p_dict)
            
            # Get s* for this c/a
            s_star = compute_min_scale_for_cn(
                config_offsets=offsets,
                target_cn=target_cn,
                lattice_type=lattice_type,
                alpha_ratio=alpha_ratios,
                bravais_type=bravais_type,
                c_ratio=c_ratio
            )
            
            if s_star is None:
                scan_data.append((c_ratio, None, None, None))
                # End matching range if we were in one
                if in_matching_range:
                    matching_ranges.append((range_start, c_ratios[list(c_ratios).index(c_ratio) - 1]))
                    in_matching_range = False
                continue
            
            # Create sublattice and calculate structure
            sublattice = Sublattice(
                name='M',
                offsets=tuple(tuple(o) for o in offsets),
                alpha_ratio=alpha_ratios,
                bravais_type=bravais_type
            )
            
            structure = calculate_complete_structure(
                sublattices=[sublattice],
                p=p,
                scale_s=s_star,
                target_N=target_cn,
                k_samples=24,
                cluster_eps_frac=cluster_eps_frac,
                include_boundary_equivalents=True
            )
            
            # Calculate weighted counts
            weighted = calculate_weighted_counts(structure)
            metal_count = weighted['metal_count']
            anion_count = weighted['intersection_count']
            
            if anion_count <= 0:
                scan_data.append((c_ratio, s_star, None, None))
                if in_matching_range:
                    matching_ranges.append((range_start, c_ratios[list(c_ratios).index(c_ratio) - 1]))
                    in_matching_range = False
                continue
            
            # Calculate M/X ratio (also check half-filling)
            mx_ratio = metal_count / anion_count
            mx_error = abs(mx_ratio - target_mx_ratio) / target_mx_ratio if target_mx_ratio > 0 else float('inf')
            
            # Check half-filling case
            if check_half_filling:
                mx_ratio_half = metal_count / (anion_count / 2.0)
                mx_error_half = abs(mx_ratio_half - target_mx_ratio) / target_mx_ratio if target_mx_ratio > 0 else float('inf')
                
                if mx_error_half < mx_error:
                    mx_ratio = mx_ratio_half
                    mx_error = mx_error_half
            
            scan_data.append((c_ratio, s_star, mx_ratio, mx_error))
            
            # Track best result
            if mx_error < best_mx_error:
                best_mx_error = mx_error
                best_c_ratio = c_ratio
                best_s_star = s_star
                best_mx_ratio = mx_ratio
            
            # Track matching ranges
            is_match = mx_error <= tolerance
            if is_match and not in_matching_range:
                range_start = c_ratio
                in_matching_range = True
            elif not is_match and in_matching_range:
                matching_ranges.append((range_start, c_ratios[list(c_ratios).index(c_ratio) - 1]))
                in_matching_range = False
        
        # Close any open range
        if in_matching_range:
            matching_ranges.append((range_start, c_ratios[-1]))
        
        return StoichiometryScanResult(
            config_id=config_id,
            target_mx_ratio=target_mx_ratio,
            best_c_ratio=best_c_ratio,
            best_s_star=best_s_star,
            best_mx_ratio=best_mx_ratio,
            best_mx_error=best_mx_error if best_mx_error != float('inf') else None,
            matching_ranges=matching_ranges,
            scan_data=scan_data,
            success=best_c_ratio is not None,
            error=None
        )
        
    except Exception as e:
        return StoichiometryScanResult(
            config_id=config_id,
            target_mx_ratio=target_mx_ratio,
            best_c_ratio=None,
            best_s_star=None,
            best_mx_ratio=None,
            best_mx_error=None,
            matching_ranges=[],
            scan_data=[],
            success=False,
            error=str(e)
        )
