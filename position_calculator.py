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
                    # alpha is coord_radius in Å, scale_s=1.0 when p.a is already physical
                    all_radius.append(alpha * scale_s)
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
    # When p.a is already the physical lattice parameter and scale_s=1.0,
    # we use legacy model: radius = coord_radius * scale_s = coord_radius * 1.0
    max_mult, sample_positions, sample_counts = max_multiplicity_for_scale(
        sublattices=sublattices,
        p=p,
        scale_s=scale_s,
        k_samples=k_samples,
        tol_inside=1e-3,
        early_stop_at=None,
        use_new_model=False  # Use p.a directly, scale_s multiplies radius
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
    scale_s: float,  # Now interpreted as lattice parameter 'a' in Å
    target_cn: int,
    anion_radius: float = 1.40,
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
        scale_s: Lattice parameter 'a' in Å (what was previously s*)
        target_cn: Target coordination number for filtering intersections
        anion_radius: Anion ionic radius in Å
        cluster_eps_frac: Clustering tolerance
    
    Returns:
        StoichiometryResult with formula and counts
    """
    try:
        # Compute coordination radii as (r_metal + r_anion) for each metal
        metal_radii = [m['radius'] for m in metals]
        if len(metal_radii) > 1:
            coord_radii = tuple(r + anion_radius for r in metal_radii)
        else:
            coord_radii = metal_radii[0] + anion_radius
        
        # Set up lattice parameters with real 'a' value
        # Use 0.99 factor to ensure intersections occur (s* is the max where CN >= target)
        a_real = scale_s * 0.99
        p_dict = {'a': a_real, 'b_ratio': 1.0, 'c_ratio': 1.0,
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
            alpha_ratio=coord_radii,  # Coordination radius in Å
            bravais_type=bravais_type
        )
        
        # Calculate structure with scale = 1 since 'a' is already real
        structure = calculate_complete_structure(
            sublattices=[sublattice],
            p=p,
            scale_s=1.0,
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
    is_half_filling_match: bool  # Whether best match uses half the anion sites
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
    anion_radius: float = 1.40,
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
        anion_radius: Anion ionic radius in Å
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
        
        # Compute coordination radii as (r_metal + r_anion)
        metal_radii = [m['radius'] for m in metals]
        if len(metal_radii) > 1:
            coord_radii = tuple(r + anion_radius for r in metal_radii)
        else:
            coord_radii = metal_radii[0] + anion_radius
        
        scan_data = []
        matching_ranges = []
        in_matching_range = False
        range_start = None
        
        best_c_ratio = None
        best_s_star = None
        best_mx_ratio = None
        best_mx_error = float('inf')
        best_is_half_filling = False
        
        c_ratios = np.linspace(c_ratio_min, c_ratio_max, n_points)
        
        for c_ratio in c_ratios:
            # Get s* for this c/a - this is now the lattice parameter 'a' in Å
            s_star = compute_min_scale_for_cn(
                config_offsets=offsets,
                target_cn=target_cn,
                lattice_type=lattice_type,
                alpha_ratio=coord_radii,
                bravais_type=bravais_type,
                lattice_params={'c_ratio': c_ratio}
            )
            
            if s_star is None:
                scan_data.append((c_ratio, None, None, None))
                # End matching range if we were in one
                if in_matching_range:
                    matching_ranges.append((range_start, c_ratios[list(c_ratios).index(c_ratio) - 1]))
                    in_matching_range = False
                continue
            
            # Set up lattice parameters with real 'a' value
            # s_star is the MAX lattice param where CN >= target
            # Use slightly smaller 'a' to ensure intersections occur
            a_real = s_star * 0.99
            p_dict = {'a': a_real, 'b_ratio': 1.0, 'c_ratio': c_ratio,
                      'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
            
            if lattice_type == 'Hexagonal':
                p_dict['gamma'] = 120.0
            
            p = LatticeParams(**p_dict)
            
            # Create sublattice and calculate structure
            sublattice = Sublattice(
                name='M',
                offsets=tuple(tuple(o) for o in offsets),
                alpha_ratio=coord_radii,
                bravais_type=bravais_type
            )
            
            structure = calculate_complete_structure(
                sublattices=[sublattice],
                p=p,
                scale_s=1.0,  # Scale = 1 since 'a' is already real
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
            is_half_filling = False
            
            # Check half-filling case
            if check_half_filling:
                mx_ratio_half = metal_count / (anion_count / 2.0)
                mx_error_half = abs(mx_ratio_half - target_mx_ratio) / target_mx_ratio if target_mx_ratio > 0 else float('inf')
                
                if mx_error_half < mx_error:
                    mx_ratio = mx_ratio_half
                    mx_error = mx_error_half
                    is_half_filling = True
            
            scan_data.append((c_ratio, s_star, mx_ratio, mx_error))
            
            # Track best result
            if mx_error < best_mx_error:
                best_mx_error = mx_error
                best_c_ratio = c_ratio
                best_s_star = s_star
                best_mx_ratio = mx_ratio
                best_is_half_filling = is_half_filling
            
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
            is_half_filling_match=best_is_half_filling,
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
            is_half_filling_match=False,
            matching_ranges=[],
            scan_data=[],
            success=False,
            error=str(e)
        )


@dataclass
class RegularityScanResult:
    """Result of scanning c/a ratios for best coordination regularity."""
    config_id: str
    best_c_ratio: Optional[float]
    best_s_star: Optional[float]
    best_regularity: float
    per_metal_scores: List[Dict]
    scan_data: List[Tuple]  # (c_ratio, s_star, regularity)
    success: bool
    error: Optional[str]
    is_half_filling: bool = False  # Whether half-filling optimization was applied


def scan_ca_for_best_regularity(
    config_id: str,
    offsets: List,
    bravais_type: str,
    lattice_type: str,
    metals: List[Dict],
    target_cn: int,
    coord_radii,
    c_ratio_min: float = 0.5,
    c_ratio_max: float = 2.5,
    n_points: int = 50,
    is_half_filling: bool = False,
    target_mx_ratio: Optional[float] = None,
    stoich_tolerance: float = 0.15
) -> RegularityScanResult:
    """
    Scan c/a ratios to find the one with best coordination regularity.
    
    Args:
        config_id: Configuration identifier
        offsets: Metal site offsets
        bravais_type: Bravais lattice type
        lattice_type: Lattice system (Tetragonal, Hexagonal, etc.)
        metals: List of metal dictionaries with symbol, cn, radius
        target_cn: Target coordination number
        coord_radii: Coordination radius (r_metal + r_anion) - scalar or tuple
        c_ratio_min: Minimum c/a ratio to scan
        c_ratio_max: Maximum c/a ratio to scan
        n_points: Number of points in scan
        is_half_filling: If True, always use half-filling (legacy behavior)
        target_mx_ratio: If provided, auto-detect whether full or half-filling 
                        is needed at each c/a point based on stoichiometry match.
                        This overrides is_half_filling.
        stoich_tolerance: Tolerance for M/X ratio matching (default 0.15 = 15%)
    
    Returns:
        RegularityScanResult with best c/a and regularity scores
    """
    from interstitial_engine import compute_min_scale_for_cn
    
    best_c_ratio = None
    best_s_star = None
    best_regularity = -1.0
    best_per_metal = []
    best_is_half = False
    scan_data = []
    
    c_ratios = np.linspace(c_ratio_min, c_ratio_max, n_points)
    
    try:
        for c_ratio in c_ratios:
            try:
                # Get s* for this c/a
                s_star = compute_min_scale_for_cn(
                    config_offsets=offsets,
                    target_cn=target_cn,
                    lattice_type=lattice_type,
                    alpha_ratio=coord_radii,
                    bravais_type=bravais_type,
                    lattice_params={'c_ratio': c_ratio}
                )
                
                if s_star is None:
                    scan_data.append((c_ratio, None, None))
                    continue
                
                # Build structure with real lattice parameter
                # s_star is the MAX lattice param where CN >= target
                # Use slightly smaller 'a' to ensure intersections occur
                sublattice = Sublattice(
                    name='M',
                    offsets=tuple(tuple(o) for o in offsets),
                    alpha_ratio=coord_radii,
                    bravais_type=bravais_type
                )
                
                a_real = s_star * 0.99  # Slightly smaller to ensure sphere overlap
                p_dict = {'a': a_real, 'b_ratio': 1.0, 'c_ratio': c_ratio,
                         'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
                if lattice_type == 'Hexagonal':
                    p_dict['gamma'] = 120.0
                
                p = LatticeParams(**p_dict)
                
                structure = calculate_complete_structure(
                    sublattices=[sublattice],
                    p=p,
                    scale_s=1.0,  # Scale = 1 since 'a' is already real
                    target_N=target_cn,
                    k_samples=24,
                    cluster_eps_frac=0.05,
                    include_boundary_equivalents=True
                )
                
                # Determine if we should use half-filling for this c/a point
                use_half_filling = is_half_filling  # Default to explicit parameter
                
                if target_mx_ratio is not None:
                    # Auto-detect based on stoichiometry
                    weighted = calculate_weighted_counts(structure)
                    metal_count = weighted['metal_count']
                    anion_count = weighted['intersection_count']
                    
                    if anion_count > 0 and metal_count > 0:
                        # Check full occupancy match
                        mx_full = metal_count / anion_count
                        error_full = abs(mx_full - target_mx_ratio) / target_mx_ratio
                        
                        # Check half occupancy match
                        mx_half = metal_count / (anion_count / 2.0)
                        error_half = abs(mx_half - target_mx_ratio) / target_mx_ratio
                        
                        # Decide which to use
                        if error_full <= stoich_tolerance and error_full <= error_half:
                            use_half_filling = False
                        elif error_half <= stoich_tolerance:
                            use_half_filling = True
                        else:
                            # Neither matches well - skip this c/a for stoichiometry-constrained search
                            scan_data.append((c_ratio, s_star, None))
                            continue
                
                # Evaluate regularity with chosen occupancy mode
                if use_half_filling:
                    half_result = find_optimal_half_filling(
                        structure=structure,
                        metals=metals,
                        max_coord_sites=target_cn,
                        target_fraction=0.5
                    )
                    
                    if half_result.success:
                        mean_reg = half_result.mean_regularity_after
                        scan_data.append((c_ratio, s_star, mean_reg))
                        
                        if mean_reg > best_regularity:
                            best_regularity = mean_reg
                            best_c_ratio = c_ratio
                            best_s_star = s_star
                            best_per_metal = half_result.per_metal_scores
                            best_is_half = True
                    else:
                        scan_data.append((c_ratio, s_star, None))
                else:
                    # Standard regularity analysis with all sites
                    coord_result = analyze_all_coordination_environments(
                        structure=structure,
                        metals=metals,
                        max_sites=target_cn
                    )
                    
                    if coord_result.success:
                        mean_reg = coord_result.summary.get('mean_overall_regularity', 0)
                        scan_data.append((c_ratio, s_star, mean_reg))
                        
                        if mean_reg > best_regularity:
                            best_regularity = mean_reg
                            best_c_ratio = c_ratio
                            best_s_star = s_star
                            best_is_half = False
                            # Store per-metal scores
                            best_per_metal = []
                            for env in coord_result.environments:
                                best_per_metal.append({
                                    'symbol': env.metal_symbol,
                                    'cn': len(env.coordination_sites),
                                    'regularity': env.overall_regularity
                                })
                    else:
                        scan_data.append((c_ratio, s_star, None))
            except Exception:
                scan_data.append((c_ratio, None, None))
                continue
        
        if best_c_ratio is not None:
            return RegularityScanResult(
                config_id=config_id,
                best_c_ratio=best_c_ratio,
                best_s_star=best_s_star,
                best_regularity=best_regularity,
                per_metal_scores=best_per_metal,
                scan_data=scan_data,
                success=True,
                error=None,
                is_half_filling=best_is_half
            )
        else:
            return RegularityScanResult(
                config_id=config_id,
                best_c_ratio=None,
                best_s_star=None,
                best_regularity=0.0,
                per_metal_scores=[],
                scan_data=scan_data,
                success=False,
                error='No valid c/a found in range',
                is_half_filling=False
            )
    except Exception as e:
        return RegularityScanResult(
            config_id=config_id,
            best_c_ratio=None,
            best_s_star=None,
            best_regularity=0.0,
            per_metal_scores=[],
            scan_data=[],
            success=False,
            error=str(e),
            is_half_filling=False
        )


# -----------------
# Coordination Environment Analysis
# -----------------

@dataclass
class CoordinationSite:
    """Information about a single coordination site (intersection) around a metal."""
    fractional: np.ndarray          # Fractional coordinates
    cartesian: np.ndarray           # Cartesian coordinates  
    distance: float                 # Distance from metal center
    multiplicity: int               # Intersection multiplicity (N value)
    image: Tuple[int, int, int]     # Periodic image offset (0,0,0 = primary cell)


@dataclass 
class CoordinationEnvironment:
    """Complete coordination environment for a single metal site."""
    metal_index: int                        # Index in metal atoms array
    metal_symbol: str                       # Element symbol
    metal_fractional: np.ndarray            # Metal fractional position
    metal_cartesian: np.ndarray             # Metal Cartesian position
    coordination_sites: List[CoordinationSite]  # List of coordinating intersection sites
    
    # Distance metrics
    distances: np.ndarray                   # Array of distances
    mean_distance: float                    # Mean coordination distance
    std_distance: float                     # Standard deviation of distances
    min_distance: float                     # Minimum distance
    max_distance: float                     # Maximum distance
    distance_range: float                   # max - min
    cv_distance: float                      # Coefficient of variation (std/mean)
    
    # Angular metrics
    angles: np.ndarray                      # All unique angles between coordination vectors
    mean_angle: float                       # Mean angle (degrees)
    std_angle: float                        # Standard deviation of angles
    
    # Regularity scores
    distance_regularity: float              # 0-1 score (1 = perfectly regular)
    angular_regularity: float               # 0-1 score based on ideal polyhedra
    overall_regularity: float               # Combined score
    
    # Ideal polyhedron comparison
    ideal_polyhedron: str                   # Name of closest ideal polyhedron
    ideal_angles: List[float]               # Expected angles for ideal polyhedron
    angle_deviation: float                  # RMS deviation from ideal angles


@dataclass
class CoordinationAnalysisResult:
    """Results of coordination environment analysis for entire structure."""
    environments: List[CoordinationEnvironment]  # One per unique metal type
    summary: Dict                                 # Summary statistics
    success: bool
    error: Optional[str] = None


# Ideal polyhedra definitions: CN -> (name, characteristic_angles)
IDEAL_POLYHEDRA = {
    2: ('linear', [180.0]),
    3: ('trigonal_planar', [120.0]),
    4: ('tetrahedron', [109.47]),  # arccos(-1/3)
    5: ('trigonal_bipyramid', [90.0, 120.0, 180.0]),
    6: ('octahedron', [90.0, 180.0]),
    7: ('pentagonal_bipyramid', [72.0, 90.0, 144.0]),
    8: ('cube', [70.53, 109.47]),  # face diagonal, body diagonal
    9: ('tricapped_trigonal_prism', [70.0, 82.0, 118.0, 136.0]),
    10: ('bicapped_square_antiprism', [65.0, 75.0, 115.0, 140.0]),
    12: ('cuboctahedron', [60.0, 90.0, 120.0, 180.0]),
}


def find_nearest_intersections_pbc(
    metal_pos_cart: np.ndarray,
    intersection_frac: np.ndarray,
    intersection_cart: np.ndarray,
    intersection_mult: np.ndarray,
    lat_vecs: np.ndarray,
    max_sites: int = 12,
    dedup_tol: float = 0.01
) -> List[CoordinationSite]:
    """
    Find the N nearest intersection sites to a metal position using PBC.
    
    Args:
        metal_pos_cart: Cartesian position of metal atom
        intersection_frac: Fractional coordinates of all intersections (N, 3)
        intersection_cart: Cartesian coordinates of all intersections (N, 3)
        intersection_mult: Multiplicities of intersections (N,)
        lat_vecs: Lattice vectors (3, 3)
        max_sites: Maximum number of coordination sites to return
        dedup_tol: Tolerance for considering two sites as duplicates (Angstrom)
    
    Returns:
        List of CoordinationSite objects, sorted by distance
    """
    if len(intersection_frac) == 0:
        return []
    
    # Generate all periodic images within a reasonable range
    # Use 27 images (3x3x3 supercell) centered on origin
    images = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                images.append((i, j, k))
    
    # Compute all distances considering periodic images
    all_sites = []
    
    for int_idx in range(len(intersection_frac)):
        frac = intersection_frac[int_idx]
        mult = intersection_mult[int_idx]
        
        for img in images:
            # Shift fractional coordinate by image
            shifted_frac = frac + np.array(img)
            # Convert to Cartesian
            shifted_cart = shifted_frac @ lat_vecs
            
            # Calculate distance
            dist = np.linalg.norm(shifted_cart - metal_pos_cart)
            
            # Skip if too close (on top of metal) or very far
            if dist < 0.1:  # Skip if essentially at the metal position
                continue
            
            all_sites.append(CoordinationSite(
                fractional=shifted_frac,
                cartesian=shifted_cart,
                distance=dist,
                multiplicity=int(mult),
                image=img
            ))
    
    # Sort by distance
    all_sites.sort(key=lambda s: s.distance)
    
    # Remove duplicates: sites at effectively the same Cartesian position
    # This happens when periodic images of different fractional sites overlap
    unique_sites = []
    for site in all_sites:
        is_duplicate = False
        for existing in unique_sites:
            # Check if Cartesian positions are essentially the same
            if np.linalg.norm(site.cartesian - existing.cartesian) < dedup_tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_sites.append(site)
        
        # Stop early if we have enough
        if len(unique_sites) >= max_sites:
            break
    
    return unique_sites[:max_sites]


def calculate_angles(metal_pos: np.ndarray, coord_positions: np.ndarray) -> np.ndarray:
    """
    Calculate all unique angles between coordination vectors from metal center.
    
    Args:
        metal_pos: Cartesian position of metal atom
        coord_positions: Cartesian positions of coordinating sites (N, 3)
    
    Returns:
        Array of unique angles in degrees
    """
    if len(coord_positions) < 2:
        return np.array([])
    
    # Calculate unit vectors from metal to each coordination site
    vectors = coord_positions - metal_pos
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)  # Avoid division by zero
    unit_vectors = vectors / norms
    
    # Calculate all unique angles
    angles = []
    n = len(unit_vectors)
    for i in range(n):
        for j in range(i + 1, n):
            dot = np.clip(np.dot(unit_vectors[i], unit_vectors[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(dot))
            angles.append(angle)
    
    return np.array(angles)


def find_closest_ideal_polyhedron(cn: int, observed_angles: np.ndarray) -> Tuple[str, List[float], float]:
    """
    Find the closest ideal polyhedron for a given coordination number.
    
    Args:
        cn: Coordination number
        observed_angles: Observed angles in degrees
    
    Returns:
        Tuple of (polyhedron_name, ideal_angles, rms_deviation)
    """
    if cn not in IDEAL_POLYHEDRA:
        # Find closest CN
        available_cns = list(IDEAL_POLYHEDRA.keys())
        cn = min(available_cns, key=lambda x: abs(x - cn))
    
    name, ideal_angles = IDEAL_POLYHEDRA[cn]
    
    if len(observed_angles) == 0:
        return name, ideal_angles, 0.0
    
    # Calculate RMS deviation: for each observed angle, find nearest ideal angle
    deviations = []
    for obs in observed_angles:
        min_dev = min(abs(obs - ideal) for ideal in ideal_angles)
        deviations.append(min_dev ** 2)
    
    rms = np.sqrt(np.mean(deviations)) if deviations else 0.0
    
    return name, ideal_angles, rms


def calculate_coordination_environment(
    metal_idx: int,
    metal_symbol: str,
    metal_frac: np.ndarray,
    metal_cart: np.ndarray,
    intersection_frac: np.ndarray,
    intersection_cart: np.ndarray,
    intersection_mult: np.ndarray,
    lat_vecs: np.ndarray,
    max_sites: int = 12
) -> CoordinationEnvironment:
    """
    Calculate complete coordination environment for a single metal site.
    
    Args:
        metal_idx: Index of metal in the metal atoms array
        metal_symbol: Element symbol
        metal_frac: Fractional coordinates of metal
        metal_cart: Cartesian coordinates of metal
        intersection_frac: All intersection fractional coordinates
        intersection_cart: All intersection Cartesian coordinates
        intersection_mult: All intersection multiplicities
        lat_vecs: Lattice vectors
        max_sites: Maximum coordination sites to consider
    
    Returns:
        CoordinationEnvironment with all metrics calculated
    """
    # Find nearest coordination sites
    coord_sites = find_nearest_intersections_pbc(
        metal_cart, intersection_frac, intersection_cart, 
        intersection_mult, lat_vecs, max_sites
    )
    
    if len(coord_sites) == 0:
        # Return empty environment
        return CoordinationEnvironment(
            metal_index=metal_idx,
            metal_symbol=metal_symbol,
            metal_fractional=metal_frac,
            metal_cartesian=metal_cart,
            coordination_sites=[],
            distances=np.array([]),
            mean_distance=0.0,
            std_distance=0.0,
            min_distance=0.0,
            max_distance=0.0,
            distance_range=0.0,
            cv_distance=0.0,
            angles=np.array([]),
            mean_angle=0.0,
            std_angle=0.0,
            distance_regularity=0.0,
            angular_regularity=0.0,
            overall_regularity=0.0,
            ideal_polyhedron='none',
            ideal_angles=[],
            angle_deviation=0.0
        )
    
    # Extract distances
    distances = np.array([s.distance for s in coord_sites])
    
    # Distance metrics
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    dist_range = max_dist - min_dist
    cv_dist = std_dist / mean_dist if mean_dist > 0 else 0.0
    
    # Calculate angles
    coord_positions = np.array([s.cartesian for s in coord_sites])
    angles = calculate_angles(metal_cart, coord_positions)
    
    mean_angle = np.mean(angles) if len(angles) > 0 else 0.0
    std_angle = np.std(angles) if len(angles) > 0 else 0.0
    
    # Find ideal polyhedron and compare
    cn = len(coord_sites)
    ideal_name, ideal_angles, angle_deviation = find_closest_ideal_polyhedron(cn, angles)
    
    # Calculate regularity scores
    # Distance regularity: 1 - CV (capped at 0)
    distance_regularity = max(0.0, 1.0 - cv_dist * 2)  # Scale CV for sensitivity
    
    # Angular regularity: based on deviation from ideal (max deviation ~60° gives 0)
    angular_regularity = max(0.0, 1.0 - angle_deviation / 30.0)
    
    # Overall regularity: weighted combination
    overall_regularity = 0.5 * distance_regularity + 0.5 * angular_regularity
    
    return CoordinationEnvironment(
        metal_index=metal_idx,
        metal_symbol=metal_symbol,
        metal_fractional=metal_frac,
        metal_cartesian=metal_cart,
        coordination_sites=coord_sites,
        distances=distances,
        mean_distance=mean_dist,
        std_distance=std_dist,
        min_distance=min_dist,
        max_distance=max_dist,
        distance_range=dist_range,
        cv_distance=cv_dist,
        angles=angles,
        mean_angle=mean_angle,
        std_angle=std_angle,
        distance_regularity=distance_regularity,
        angular_regularity=angular_regularity,
        overall_regularity=overall_regularity,
        ideal_polyhedron=ideal_name,
        ideal_angles=ideal_angles,
        angle_deviation=angle_deviation
    )


def get_unique_metal_sites(metal_atoms: MetalAtomData) -> List[int]:
    """
    Get indices of unique metal sites (one per offset type).
    
    For structures with multiple metals, we want one representative
    site per metal type (offset_idx).
    """
    unique_indices = []
    seen_offsets = set()
    
    for i in range(len(metal_atoms.fractional)):
        offset_idx = metal_atoms.offset_idx[i]
        if offset_idx not in seen_offsets:
            seen_offsets.add(offset_idx)
            unique_indices.append(i)
    
    return unique_indices


def analyze_all_coordination_environments(
    structure: CompleteStructureData,
    metals: List[Dict],
    max_sites: int = 12
) -> CoordinationAnalysisResult:
    """
    Analyze coordination environments for all unique metal types in a structure.
    
    Args:
        structure: Complete structure data with metal atoms and intersections
        metals: List of metal dictionaries with 'symbol' and 'cn' keys
        max_sites: Default maximum coordination sites (used if metal has no 'cn' key)
    
    Returns:
        CoordinationAnalysisResult with all environments and summary
    """
    try:
        metal_atoms = structure.metal_atoms
        intersections = structure.intersections
        lat_vecs = structure.lattice_vectors
        
        if len(metal_atoms.fractional) == 0:
            return CoordinationAnalysisResult(
                environments=[],
                summary={'error': 'No metal atoms in structure'},
                success=False,
                error='No metal atoms in structure'
            )
        
        if len(intersections.fractional) == 0:
            return CoordinationAnalysisResult(
                environments=[],
                summary={'error': 'No intersection sites found'},
                success=False,
                error='No intersection sites found'
            )
        
        # Get unique metal sites (one per type)
        unique_indices = get_unique_metal_sites(metal_atoms)
        
        environments = []
        for idx in unique_indices:
            # Get metal info
            offset_idx = metal_atoms.offset_idx[idx]
            
            if offset_idx < len(metals):
                symbol = metals[offset_idx]['symbol']
                # Use the metal's specific CN, not the anion CN
                metal_cn = metals[offset_idx].get('cn', max_sites)
            else:
                symbol = f'M{offset_idx+1}'
                metal_cn = max_sites
            
            # Calculate coordination environment using this metal's CN
            env = calculate_coordination_environment(
                metal_idx=idx,
                metal_symbol=symbol,
                metal_frac=metal_atoms.fractional[idx],
                metal_cart=metal_atoms.cartesian[idx],
                intersection_frac=intersections.fractional,
                intersection_cart=intersections.cartesian,
                intersection_mult=intersections.multiplicity,
                lat_vecs=lat_vecs,
                max_sites=metal_cn  # Use metal's CN, not anion CN
            )
            environments.append(env)
        
        # Generate summary statistics
        summary = {
            'num_metal_types': len(environments),
            'environments': []
        }
        
        for env in environments:
            env_summary = {
                'symbol': env.metal_symbol,
                'cn': len(env.coordination_sites),
                'mean_distance': env.mean_distance,
                'cv_distance': env.cv_distance,
                'ideal_polyhedron': env.ideal_polyhedron,
                'angle_deviation': env.angle_deviation,
                'distance_regularity': env.distance_regularity,
                'angular_regularity': env.angular_regularity,
                'overall_regularity': env.overall_regularity
            }
            summary['environments'].append(env_summary)
        
        # Overall structure regularity
        if environments:
            summary['mean_overall_regularity'] = np.mean([e.overall_regularity for e in environments])
        else:
            summary['mean_overall_regularity'] = 0.0
        
        return CoordinationAnalysisResult(
            environments=environments,
            summary=summary,
            success=True,
            error=None
        )
        
    except Exception as e:
        return CoordinationAnalysisResult(
            environments=[],
            summary={'error': str(e)},
            success=False,
            error=str(e)
        )


# -----------------
# Optimized Half-Filling
# -----------------

from itertools import combinations


def calculate_regularity_for_sites(
    metal_cart: np.ndarray,
    coord_sites: List[CoordinationSite]
) -> float:
    """
    Calculate regularity score for a given set of coordination sites.
    
    Args:
        metal_cart: Cartesian position of metal
        coord_sites: List of coordination sites
    
    Returns:
        Overall regularity score (0-1)
    """
    if len(coord_sites) < 2:
        return 0.0
    
    # Distance metrics
    distances = np.array([s.distance for s in coord_sites])
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    cv_dist = std_dist / mean_dist if mean_dist > 0 else 0.0
    
    # Angular metrics
    coord_positions = np.array([s.cartesian for s in coord_sites])
    angles = calculate_angles(metal_cart, coord_positions)
    
    # Find ideal polyhedron
    cn = len(coord_sites)
    _, _, angle_deviation = find_closest_ideal_polyhedron(cn, angles)
    
    # Calculate regularity scores
    distance_regularity = max(0.0, 1.0 - cv_dist * 2)
    angular_regularity = max(0.0, 1.0 - angle_deviation / 30.0)
    
    return 0.5 * distance_regularity + 0.5 * angular_regularity


def optimize_half_filling_for_metal(
    metal_cart: np.ndarray,
    all_coord_sites: List[CoordinationSite],
    target_cn: int = None
) -> Tuple[List[CoordinationSite], float]:
    """
    Find the optimal half of coordination sites for maximum regularity.
    
    Args:
        metal_cart: Cartesian position of metal
        all_coord_sites: Full list of coordination sites
        target_cn: Target coordination number (default: half of total)
    
    Returns:
        Tuple of (optimal sites, regularity score)
    """
    n_sites = len(all_coord_sites)
    
    if n_sites == 0:
        return [], 0.0
    
    if target_cn is None:
        target_cn = n_sites // 2
    
    target_cn = min(target_cn, n_sites)
    
    if target_cn == n_sites:
        reg = calculate_regularity_for_sites(metal_cart, all_coord_sites)
        return all_coord_sites, reg
    
    # For small numbers, try all combinations
    if n_sites <= 14:
        best_score = -1.0
        best_subset = None
        
        for subset_indices in combinations(range(n_sites), target_cn):
            subset = [all_coord_sites[i] for i in subset_indices]
            score = calculate_regularity_for_sites(metal_cart, subset)
            
            if score > best_score:
                best_score = score
                best_subset = subset
        
        return best_subset if best_subset else [], best_score
    
    else:
        # For larger numbers, use greedy removal
        # Start with all sites sorted by distance
        remaining = list(all_coord_sites)
        
        while len(remaining) > target_cn:
            best_removal_score = -1.0
            best_removal_idx = 0
            
            for i in range(len(remaining)):
                # Try removing site i
                test_subset = remaining[:i] + remaining[i+1:]
                score = calculate_regularity_for_sites(metal_cart, test_subset)
                
                if score > best_removal_score:
                    best_removal_score = score
                    best_removal_idx = i
            
            remaining.pop(best_removal_idx)
        
        final_score = calculate_regularity_for_sites(metal_cart, remaining)
        return remaining, final_score


@dataclass
class HalfFillingResult:
    """Result of optimized half-filling analysis."""
    kept_site_indices: List[int]          # Indices of sites to keep (in unique intersections)
    kept_site_fractions: np.ndarray       # Fractional coords of kept sites
    original_count: int                    # Original number of unique sites
    kept_count: int                        # Number of sites kept
    mean_regularity_before: float          # Mean regularity with all sites
    mean_regularity_after: float           # Mean regularity after half-filling
    per_metal_scores: List[Dict]           # Per-metal regularity scores
    success: bool
    error: Optional[str] = None


def find_optimal_half_filling(
    structure: CompleteStructureData,
    metals: List[Dict],
    max_coord_sites: int = 12,
    target_fraction: float = 0.5
) -> HalfFillingResult:
    """
    Find the optimal set of intersection sites to keep for half-filling.
    
    Optimizes for maximum average coordination regularity across all metal types.
    
    Args:
        structure: Complete structure data
        metals: List of metal dictionaries with 'cn' keys for per-metal coordination
        max_coord_sites: Default max coordination sites (used if metal has no 'cn' key)
        target_fraction: Fraction of sites to keep (default 0.5)
    
    Returns:
        HalfFillingResult with optimal site selection
    """
    
    def get_metal_cn(offset_idx: int) -> int:
        """Get the CN for a metal by its offset index."""
        if offset_idx < len(metals):
            return metals[offset_idx].get('cn', max_coord_sites)
        return max_coord_sites
    
    try:
        metal_atoms = structure.metal_atoms
        intersections = structure.intersections
        lat_vecs = structure.lattice_vectors
        
        if len(intersections.fractional) == 0:
            return HalfFillingResult(
                kept_site_indices=[],
                kept_site_fractions=np.empty((0, 3)),
                original_count=0,
                kept_count=0,
                mean_regularity_before=0.0,
                mean_regularity_after=0.0,
                per_metal_scores=[],
                success=False,
                error="No intersection sites found"
            )
        
        # Get unique intersection sites (without boundary equivalents)
        unique_frac, unique_mult = get_unique_intersections(intersections)
        n_unique = len(unique_frac)
        
        if n_unique == 0:
            return HalfFillingResult(
                kept_site_indices=[],
                kept_site_fractions=np.empty((0, 3)),
                original_count=0,
                kept_count=0,
                mean_regularity_before=0.0,
                mean_regularity_after=0.0,
                per_metal_scores=[],
                success=False,
                error="No unique intersection sites"
            )
        
        # Convert unique sites to Cartesian
        unique_cart = unique_frac @ lat_vecs
        
        # Get unique metal sites
        unique_metal_indices = get_unique_metal_sites(metal_atoms)
        
        # Calculate regularity with all sites for baseline
        baseline_scores = []
        for metal_idx in unique_metal_indices:
            offset_idx = metal_atoms.offset_idx[metal_idx]
            metal_cart = metal_atoms.cartesian[metal_idx]
            metal_cn = get_metal_cn(offset_idx)  # Use this metal's CN
            
            # Find coordination sites for this metal
            coord_sites = find_nearest_intersections_pbc(
                metal_cart, unique_frac, unique_cart, unique_mult,
                lat_vecs, metal_cn
            )
            
            if coord_sites:
                score = calculate_regularity_for_sites(metal_cart, coord_sites)
                baseline_scores.append(score)
        
        mean_baseline = np.mean(baseline_scores) if baseline_scores else 0.0
        
        # Target number of sites to keep
        target_keep = max(1, int(n_unique * target_fraction))
        
        # If only a few unique sites, try all combinations
        if n_unique <= 12:
            best_score = -1.0
            best_indices = list(range(n_unique))
            
            for kept_indices in combinations(range(n_unique), target_keep):
                kept_indices = list(kept_indices)
                kept_frac = unique_frac[kept_indices]
                kept_cart = unique_cart[kept_indices]
                kept_mult = unique_mult[kept_indices]
                
                # Calculate mean regularity for this selection
                scores = []
                for metal_idx in unique_metal_indices:
                    offset_idx = metal_atoms.offset_idx[metal_idx]
                    metal_cart = metal_atoms.cartesian[metal_idx]
                    metal_cn = get_metal_cn(offset_idx)  # Use this metal's CN
                    
                    coord_sites = find_nearest_intersections_pbc(
                        metal_cart, kept_frac, kept_cart, kept_mult,
                        lat_vecs, metal_cn
                    )
                    
                    if coord_sites:
                        score = calculate_regularity_for_sites(metal_cart, coord_sites)
                        scores.append(score)
                
                mean_score = np.mean(scores) if scores else 0.0
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_indices = kept_indices
            
            # Calculate per-metal scores for best selection
            kept_frac = unique_frac[best_indices]
            kept_cart = unique_cart[best_indices]
            kept_mult = unique_mult[best_indices]
            
            per_metal = []
            for metal_idx in unique_metal_indices:
                offset_idx = metal_atoms.offset_idx[metal_idx]
                symbol = metals[offset_idx]['symbol'] if offset_idx < len(metals) else f'M{offset_idx+1}'
                metal_cart = metal_atoms.cartesian[metal_idx]
                metal_cn = get_metal_cn(offset_idx)  # Use this metal's CN
                
                coord_sites = find_nearest_intersections_pbc(
                    metal_cart, kept_frac, kept_cart, kept_mult,
                    lat_vecs, metal_cn
                )
                
                cn = len(coord_sites)
                score = calculate_regularity_for_sites(metal_cart, coord_sites) if coord_sites else 0.0
                
                per_metal.append({
                    'symbol': symbol,
                    'cn': cn,
                    'regularity': score
                })
            
            return HalfFillingResult(
                kept_site_indices=list(best_indices),
                kept_site_fractions=unique_frac[best_indices],
                original_count=n_unique,
                kept_count=len(best_indices),
                mean_regularity_before=mean_baseline,
                mean_regularity_after=best_score,
                per_metal_scores=per_metal,
                success=True,
                error=None
            )
        
        else:
            # For many sites, use greedy removal approach
            remaining_indices = list(range(n_unique))
            
            while len(remaining_indices) > target_keep:
                best_removal_score = -1.0
                best_removal_idx = 0
                
                for remove_pos in range(len(remaining_indices)):
                    # Try removing this site
                    test_indices = remaining_indices[:remove_pos] + remaining_indices[remove_pos+1:]
                    test_frac = unique_frac[test_indices]
                    test_cart = unique_cart[test_indices]
                    test_mult = unique_mult[test_indices]
                    
                    # Calculate mean regularity
                    scores = []
                    for metal_idx in unique_metal_indices:
                        offset_idx = metal_atoms.offset_idx[metal_idx]
                        metal_cart = metal_atoms.cartesian[metal_idx]
                        metal_cn = get_metal_cn(offset_idx)  # Use this metal's CN
                        
                        coord_sites = find_nearest_intersections_pbc(
                            metal_cart, test_frac, test_cart, test_mult,
                            lat_vecs, metal_cn
                        )
                        
                        if coord_sites:
                            score = calculate_regularity_for_sites(metal_cart, coord_sites)
                            scores.append(score)
                    
                    mean_score = np.mean(scores) if scores else 0.0
                    
                    if mean_score > best_removal_score:
                        best_removal_score = mean_score
                        best_removal_idx = remove_pos
                
                remaining_indices.pop(best_removal_idx)
            
            # Calculate final scores
            kept_frac = unique_frac[remaining_indices]
            kept_cart = unique_cart[remaining_indices]
            kept_mult = unique_mult[remaining_indices]
            
            per_metal = []
            final_scores = []
            for metal_idx in unique_metal_indices:
                offset_idx = metal_atoms.offset_idx[metal_idx]
                symbol = metals[offset_idx]['symbol'] if offset_idx < len(metals) else f'M{offset_idx+1}'
                metal_cart = metal_atoms.cartesian[metal_idx]
                metal_cn = get_metal_cn(offset_idx)  # Use this metal's CN
                
                coord_sites = find_nearest_intersections_pbc(
                    metal_cart, kept_frac, kept_cart, kept_mult,
                    lat_vecs, metal_cn
                )
                
                cn = len(coord_sites)
                score = calculate_regularity_for_sites(metal_cart, coord_sites) if coord_sites else 0.0
                final_scores.append(score)
                
                per_metal.append({
                    'symbol': symbol,
                    'cn': cn,
                    'regularity': score
                })
            
            return HalfFillingResult(
                kept_site_indices=remaining_indices,
                kept_site_fractions=unique_frac[remaining_indices],
                original_count=n_unique,
                kept_count=len(remaining_indices),
                mean_regularity_before=mean_baseline,
                mean_regularity_after=np.mean(final_scores) if final_scores else 0.0,
                per_metal_scores=per_metal,
                success=True,
                error=None
            )
    
    except Exception as e:
        return HalfFillingResult(
            kept_site_indices=[],
            kept_site_fractions=np.empty((0, 3)),
            original_count=0,
            kept_count=0,
            mean_regularity_before=0.0,
            mean_regularity_after=0.0,
            per_metal_scores=[],
            success=False,
            error=str(e)
        )


# -----------------
# Madelung Energy Calculation
# -----------------

@dataclass
class MadelungResult:
    """Result of Madelung energy calculation."""
    energy_per_formula: float      # eV per formula unit
    energy_per_atom: float         # eV per atom
    madelung_constant: float       # Dimensionless Madelung constant (approximate)
    nearest_neighbor_dist: float   # Å, shortest cation-anion distance
    n_cations: int                 # Number of cations in unit cell
    n_anions: int                  # Number of anions in unit cell
    formula_units: float           # Formula units per unit cell
    success: bool
    error: Optional[str] = None


def compute_physical_scale_factor(
    lattice_param_a: float,
    coord_radius: float,
    structure_type: str = 'rocksalt'
) -> float:
    """
    Compute the scale factor s that places anions at crystallographic positions.
    
    For ionic structures, anions should be at specific crystallographic sites
    (octahedral holes for rocksalt, tetrahedral holes for zinc blende, etc.)
    This function computes s such that sphere intersections occur at those sites.
    
    Args:
        lattice_param_a: Lattice parameter 'a' in Ångströms
        coord_radius: Coordination radius (r_cation + r_anion) in Ångströms
        structure_type: Type of structure ('rocksalt', 'fluorite', 'zincblende', etc.)
    
    Returns:
        Scale factor s (typically close to 1 for physical structures)
    """
    # For different structure types, anions are at different distances from cations
    if structure_type in ['rocksalt', 'nacl']:
        # Anions at octahedral sites: distance = a/2
        anion_dist = lattice_param_a / 2
    elif structure_type in ['fluorite', 'caf2']:
        # Anions at tetrahedral sites: distance = a*sqrt(3)/4
        anion_dist = lattice_param_a * np.sqrt(3) / 4
    elif structure_type in ['zincblende', 'sphalerite', 'zns']:
        # Anions at tetrahedral sites: distance = a*sqrt(3)/4
        anion_dist = lattice_param_a * np.sqrt(3) / 4
    elif structure_type in ['rutile', 'tio2']:
        # More complex, use approximate value
        anion_dist = lattice_param_a * 0.45
    elif structure_type in ['perovskite', 'abo3']:
        # Anions at face centers: distance = a/2
        anion_dist = lattice_param_a / 2
    else:
        # Default: assume octahedral-like
        anion_dist = lattice_param_a / 2
    
    # Scale factor such that spheres just reach the anion position
    # Add small factor (1.02) to ensure intersection occurs
    s = (anion_dist / coord_radius) * 1.02
    
    return s


def calculate_madelung_energy(
    structure: CompleteStructureData,
    metals: List[Dict],
    anion_charge: int,
    target_multiplicity: Optional[int] = None,
    supercell_size: int = 5,
    convergence_check: bool = True
) -> MadelungResult:
    """
    Calculate approximate Madelung energy using direct Coulomb summation.
    
    This is a simplified calculation using a finite supercell cutoff.
    For more accurate results, Ewald summation would be needed.
    
    Args:
        structure: Complete structure with metal atoms and anion positions
        metals: List of metal dicts with 'charge' keys
        anion_charge: Charge of anion (negative, e.g., -2 for O²⁻)
        target_multiplicity: Only use intersection sites with this multiplicity
                            (None = use all sites with multiplicity >= max/2)
        supercell_size: Size of supercell for summation (e.g., 5 means 5×5×5)
        convergence_check: If True, also compute with smaller supercell to estimate error
    
    Returns:
        MadelungResult with energy and Madelung constant
    """
    try:
        # Coulomb constant in convenient units: k = e²/(4πε₀) ≈ 14.3996 eV·Å
        K_COULOMB = 14.3996  # eV·Å
        
        lat_vecs = structure.lattice_vectors
        metal_atoms = structure.metal_atoms
        intersections = structure.intersections
        
        if len(metal_atoms.fractional) == 0:
            return MadelungResult(
                energy_per_formula=0, energy_per_atom=0, madelung_constant=0,
                nearest_neighbor_dist=0, n_cations=0, n_anions=0, formula_units=0,
                success=False, error="No metal atoms in structure"
            )
        
        if len(intersections.fractional) == 0:
            return MadelungResult(
                energy_per_formula=0, energy_per_atom=0, madelung_constant=0,
                nearest_neighbor_dist=0, n_cations=0, n_anions=0, formula_units=0,
                success=False, error="No anion sites in structure"
            )
        
        # Filter intersections by multiplicity
        mult = intersections.multiplicity
        if target_multiplicity is not None:
            mask = mult == target_multiplicity
        else:
            # Use sites with highest multiplicity (the "real" anion sites)
            max_mult = np.max(mult)
            mask = mult >= max_mult
        
        anion_frac_all = intersections.fractional[mask]
        anion_cart_all = intersections.cartesian[mask]
        
        if len(anion_frac_all) == 0:
            return MadelungResult(
                energy_per_formula=0, energy_per_atom=0, madelung_constant=0,
                nearest_neighbor_dist=0, n_cations=0, n_anions=0, formula_units=0,
                success=False, error="No anion sites with target multiplicity"
            )
        
        # Get unique sites only (not boundary equivalents)
        def get_unique_sites(frac_coords, cart_coords, tol=0.02):
            """Filter to unique sites within unit cell [0, 1)."""
            unique_frac = []
            unique_cart = []
            for i, f in enumerate(frac_coords):
                # Wrap to [0, 1)
                f_wrapped = f - np.floor(f)
                f_wrapped = np.where(np.abs(f_wrapped - 1.0) < tol, 0.0, f_wrapped)
                
                # Check not duplicate
                is_dup = False
                for uf in unique_frac:
                    if np.allclose(f_wrapped, uf, atol=tol):
                        is_dup = True
                        break
                if not is_dup:
                    unique_frac.append(f_wrapped)
                    unique_cart.append(frac_to_cart(f_wrapped, structure.lattice_vectors))
            return np.array(unique_frac), np.array(unique_cart)
        
        # Get unique cations
        cation_frac, cation_cart = get_unique_sites(
            metal_atoms.fractional, metal_atoms.cartesian
        )
        
        # Assign charges to cations
        cation_charges = []
        seen_offsets = set()
        for i, f in enumerate(metal_atoms.fractional):
            f_wrapped = f - np.floor(f)
            f_wrapped = np.where(np.abs(f_wrapped - 1.0) < 0.02, 0.0, f_wrapped)
            
            # Check if this is a new unique site
            is_new = True
            for sf in seen_offsets:
                if np.allclose(f_wrapped, np.array(sf), atol=0.02):
                    is_new = False
                    break
            
            if is_new:
                seen_offsets.add(tuple(f_wrapped))
                offset_idx = metal_atoms.offset_idx[i]
                if offset_idx < len(metals):
                    cation_charges.append(metals[offset_idx]['charge'])
                else:
                    cation_charges.append(2)
        
        cation_charges = np.array(cation_charges[:len(cation_frac)])
        
        # Get unique anions
        anion_frac, anion_cart = get_unique_sites(anion_frac_all, anion_cart_all)
        anion_charges_arr = np.full(len(anion_frac), anion_charge)
        
        n_cations = len(cation_frac)
        n_anions = len(anion_frac)
        
        if n_cations == 0 or n_anions == 0:
            return MadelungResult(
                energy_per_formula=0, energy_per_atom=0, madelung_constant=0,
                nearest_neighbor_dist=0, n_cations=n_cations, n_anions=n_anions, 
                formula_units=0, success=False, 
                error=f"No valid sites: {n_cations} cations, {n_anions} anions"
            )
        
        # Combine all ions
        all_frac = np.vstack([cation_frac, anion_frac])
        all_cart = np.vstack([cation_cart, anion_cart])
        all_charges = np.concatenate([cation_charges, anion_charges_arr])
        n_ions = len(all_frac)
        
        # Generate supercell translation vectors
        half = supercell_size // 2
        translations = []
        for i in range(-half, half + 1):
            for j in range(-half, half + 1):
                for k in range(-half, half + 1):
                    translations.append([i, j, k])
        translations = np.array(translations)
        trans_cart = translations @ lat_vecs
        
        # Calculate energy using direct summation
        total_energy = 0.0
        min_cation_anion_dist = float('inf')
        
        for i in range(n_ions):
            qi = all_charges[i]
            ri = all_cart[i]
            
            for j in range(n_ions):
                qj = all_charges[j]
                
                for t in trans_cart:
                    if i == j and np.allclose(t, 0):
                        continue
                    
                    rj = all_cart[j] + t
                    r = np.linalg.norm(rj - ri)
                    
                    if r < 0.1:
                        continue
                    
                    total_energy += qi * qj / r
                    
                    # Track minimum cation-anion distance (only in nearby cells)
                    if (i < n_cations) != (j < n_cations):  # One cation, one anion
                        if np.linalg.norm(t) < np.linalg.norm(lat_vecs[0]) * 1.5:
                            min_cation_anion_dist = min(min_cation_anion_dist, r)
        
        # Each pair counted twice
        total_energy = 0.5 * K_COULOMB * total_energy
        
        # Calculate formula units (based on charge balance)
        total_cation_charge = np.sum(cation_charges)
        total_anion_charge = abs(anion_charge) * n_anions
        
        # Formula units = how many times the formula repeats
        from math import gcd
        if total_cation_charge > 0 and total_anion_charge > 0:
            g = gcd(int(total_cation_charge), int(total_anion_charge))
            formula_units = g / abs(anion_charge) if abs(anion_charge) > 0 else 1
        else:
            formula_units = 1
        
        energy_per_formula = total_energy / max(formula_units, 1)
        energy_per_atom = total_energy / n_ions
        
        # Madelung constant: A = -E * r₀ / (k * z+ * |z-|)
        if min_cation_anion_dist < float('inf'):
            avg_cation_charge = np.mean(cation_charges)
            madelung_constant = -energy_per_formula * min_cation_anion_dist / (
                K_COULOMB * avg_cation_charge * abs(anion_charge)
            )
        else:
            madelung_constant = 0
        
        # Sanity checks for unrealistic results
        warning = None
        
        # Check for unrealistic Madelung constant (typical range: 1-5 for ionic crystals)
        if abs(madelung_constant) > 10:
            warning = f"Madelung constant {madelung_constant:.1f} is unrealistic (expected 1-5). Too many intersection sites?"
        
        # Check for anions too close together (indicates spurious intersections)
        if n_anions > 1:
            min_anion_anion_dist = float('inf')
            for i in range(n_anions):
                for j in range(i + 1, n_anions):
                    d = np.linalg.norm(anion_cart[i] - anion_cart[j])
                    if d > 0.1:
                        min_anion_anion_dist = min(min_anion_anion_dist, d)
            
            # Anions should typically be > 2 Å apart
            if min_anion_anion_dist < 1.5:
                warning = f"Anions only {min_anion_anion_dist:.2f} Å apart - likely spurious intersections"
        
        # Check M/X ratio is reasonable (expect integer or simple fraction)
        mx_ratio = n_cations / n_anions if n_anions > 0 else 0
        # Common ratios: 1:1, 1:2, 2:1, 2:3, 3:2, 1:3, 3:1
        common_ratios = [0.333, 0.5, 0.667, 1.0, 1.5, 2.0, 3.0]
        is_common = any(abs(mx_ratio - r) < 0.1 for r in common_ratios)
        if not is_common and warning is None:
            warning = f"Unusual M/X ratio {mx_ratio:.2f} - may indicate spurious sites"
        
        return MadelungResult(
            energy_per_formula=energy_per_formula,
            energy_per_atom=energy_per_atom,
            madelung_constant=madelung_constant,
            nearest_neighbor_dist=min_cation_anion_dist,
            n_cations=n_cations,
            n_anions=n_anions,
            formula_units=formula_units,
            success=True,
            error=warning  # Use error field for warnings too
        )
    
    except Exception as e:
        return MadelungResult(
            energy_per_formula=0, energy_per_atom=0, madelung_constant=0,
            nearest_neighbor_dist=0, n_cations=0, n_anions=0, formula_units=0,
            success=False, error=str(e)
        )
