"""
LATTICE SEARCH TOOL - Phase 1
==============================

Systematic enumeration of N interpenetrating Bravais lattices using
group theory and coset-of-subgroups methodology.

Key concepts:
- HNF (Hermite Normal Form) defines supercell: M = det(H) lattice points
- Quotient group G = Z³/HZ³ represents offsets modulo the supercell
- Orbits are cosets of subgroups of G
- Configuration = partition of G into orbits of specified sizes

Validation target: Spinel cation sublattice
- 6 interpenetrating FCC lattices
- Orbit sizes (2, 4): 2 tetrahedral + 4 octahedral sites
- M = 8 for conventional 2×2×2 supercell

Author: Rob (rebuilt with Claude)
"""

import numpy as np
from typing import List, Tuple, Set, Dict, Optional, Generator
from dataclasses import dataclass
from itertools import combinations, product
from functools import lru_cache
import math


# =============================================================================
# HNF ENUMERATION
# =============================================================================

def enumerate_hnf(M: int) -> List[np.ndarray]:
    """
    Enumerate all 3×3 Hermite Normal Form matrices with determinant M.
    
    HNF is upper triangular with:
    - Diagonal elements a, b, c where a*b*c = M
    - Off-diagonal: 0 ≤ H[0,1] < b, 0 ≤ H[0,2] < c, 0 ≤ H[1,2] < c
    
    Returns list of 3×3 integer numpy arrays.
    """
    hnfs = []
    
    # Find all factorizations M = a * b * c
    for a in range(1, M + 1):
        if M % a != 0:
            continue
        remaining = M // a
        
        for b in range(1, remaining + 1):
            if remaining % b != 0:
                continue
            c = remaining // b
            
            # Enumerate off-diagonal elements
            for h01 in range(b):
                for h02 in range(c):
                    for h12 in range(c):
                        H = np.array([
                            [a, h01, h02],
                            [0, b, h12],
                            [0, 0, c]
                        ], dtype=np.int64)
                        hnfs.append(H)
    
    return hnfs


def hnf_to_string(H: np.ndarray) -> str:
    """Human-readable HNF description."""
    diag = f"diag({H[0,0]},{H[1,1]},{H[2,2]})"
    off = f"off({H[0,1]},{H[0,2]},{H[1,2]})"
    return f"{diag} {off}"


def is_hnf(H: np.ndarray) -> bool:
    """
    Check if H is a valid Hermite Normal Form matrix.
    
    HNF requirements:
    1. Upper triangular (H[i,j] = 0 for i > j)
    2. Diagonal entries positive
    3. Off-diagonal entries in [0, diagonal) for their column
    
    Note: For upper triangular matrices, det(H) = product of diagonal entries.
    This is used throughout the codebase for computing group order.
    """
    # Check upper triangular
    if H[1, 0] != 0 or H[2, 0] != 0 or H[2, 1] != 0:
        return False
    
    # Check positive diagonal
    if H[0, 0] <= 0 or H[1, 1] <= 0 or H[2, 2] <= 0:
        return False
    
    # Check off-diagonal bounds
    if not (0 <= H[0, 1] < H[1, 1]):
        return False
    if not (0 <= H[0, 2] < H[2, 2]):
        return False
    if not (0 <= H[1, 2] < H[2, 2]):
        return False
    
    return True


def hnf_determinant(H: np.ndarray) -> int:
    """
    Compute determinant of an HNF matrix.
    
    For upper triangular matrices (which HNF always is),
    det(H) = product of diagonal entries.
    
    This is NOT valid for general matrices - only use on verified HNFs.
    """
    assert is_hnf(H), f"Matrix is not HNF: {H}"
    return int(H[0, 0]) * int(H[1, 1]) * int(H[2, 2])


def is_diagonal_hnf(H: np.ndarray) -> bool:
    """Check if HNF is diagonal (off-diagonal elements are zero)."""
    return H[0, 1] == 0 and H[0, 2] == 0 and H[1, 2] == 0


def enumerate_diagonal_hnf(M: int) -> List[np.ndarray]:
    """Enumerate only diagonal HNF matrices with determinant M."""
    return [H for H in enumerate_hnf(M) if is_diagonal_hnf(H)]


# =============================================================================
# SMITH NORMAL FORM
# =============================================================================

def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
    """
    Extended Euclidean algorithm.
    Returns (g, x, y) such that a*x + b*y = g = gcd(a, b).
    """
    if b == 0:
        return (a, 1, 0)
    else:
        g, x, y = extended_gcd(b, a % b)
        return (g, y, x - (a // b) * y)


def integer_matrix_inverse(M: np.ndarray) -> np.ndarray:
    """
    Compute exact integer inverse of a unimodular matrix (det = ±1).
    
    Uses adjugate formula: M^{-1} = adj(M) / det(M)
    For unimodular matrices, det = ±1, so M^{-1} = ±adj(M).
    
    Args:
        M: 3×3 integer matrix with det = ±1
    
    Returns:
        3×3 integer matrix M^{-1}
    
    Raises:
        ValueError: if matrix is not unimodular
    """
    M = np.asarray(M, dtype=np.int64)
    
    # Compute determinant using Sarrus' rule (exact for integers)
    det = (M[0,0] * (M[1,1]*M[2,2] - M[1,2]*M[2,1])
         - M[0,1] * (M[1,0]*M[2,2] - M[1,2]*M[2,0])
         + M[0,2] * (M[1,0]*M[2,1] - M[1,1]*M[2,0]))
    
    if abs(det) != 1:
        raise ValueError(f"Matrix is not unimodular: det = {det}")
    
    # Compute adjugate (transpose of cofactor matrix)
    adj = np.zeros((3, 3), dtype=np.int64)
    
    adj[0,0] = M[1,1]*M[2,2] - M[1,2]*M[2,1]
    adj[0,1] = -(M[0,1]*M[2,2] - M[0,2]*M[2,1])
    adj[0,2] = M[0,1]*M[1,2] - M[0,2]*M[1,1]
    
    adj[1,0] = -(M[1,0]*M[2,2] - M[1,2]*M[2,0])
    adj[1,1] = M[0,0]*M[2,2] - M[0,2]*M[2,0]
    adj[1,2] = -(M[0,0]*M[1,2] - M[0,2]*M[1,0])
    
    adj[2,0] = M[1,0]*M[2,1] - M[1,1]*M[2,0]
    adj[2,1] = -(M[0,0]*M[2,1] - M[0,1]*M[2,0])
    adj[2,2] = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    
    # M^{-1} = adj(M) / det(M) = ±adj(M)
    return adj * det  # det is ±1, so this is exact


def smith_normal_form(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Smith Normal Form of integer matrix H.
    
    Returns (D, U, V) such that:
    - U @ H @ V = D
    - D is diagonal with d1 | d2 | d3 (divisibility chain)
    - U, V are unimodular (det = ±1)
    
    For a 3×3 upper triangular HNF, D = diag(d1, d2, d3) where
    di are the invariant factors of the group Z³/HZ³.
    
    The quotient group is then isomorphic to Z_{d1} × Z_{d2} × Z_{d3}.
    """
    # Work with copies
    A = H.astype(np.int64).copy()
    n = A.shape[0]
    
    # Initialize transformation matrices
    U = np.eye(n, dtype=np.int64)
    V = np.eye(n, dtype=np.int64)
    
    for k in range(n):
        # Find pivot (smallest nonzero element in submatrix)
        pivot_found = False
        while not pivot_found:
            # Find minimum nonzero in A[k:, k:]
            min_val = None
            min_i, min_j = k, k
            for i in range(k, n):
                for j in range(k, n):
                    if A[i, j] != 0:
                        if min_val is None or abs(A[i, j]) < abs(min_val):
                            min_val = A[i, j]
                            min_i, min_j = i, j
            
            if min_val is None:
                # All zeros in submatrix, done with this diagonal
                break
            
            # Move pivot to (k, k)
            if min_i != k:
                A[[k, min_i]] = A[[min_i, k]]
                U[[k, min_i]] = U[[min_i, k]]
            if min_j != k:
                A[:, [k, min_j]] = A[:, [min_j, k]]
                V[:, [k, min_j]] = V[:, [min_j, k]]
            
            # Eliminate in row k and column k
            pivot_found = True
            
            # Column operations
            for j in range(k + 1, n):
                if A[k, j] != 0:
                    if A[k, j] % A[k, k] == 0:
                        q = A[k, j] // A[k, k]
                        A[:, j] -= q * A[:, k]
                        V[:, j] -= q * V[:, k]
                    else:
                        # Use extended GCD
                        g, x, y = extended_gcd(A[k, k], A[k, j])
                        a, b = A[k, k], A[k, j]
                        # New column k: x*col_k + y*col_j (has gcd in position k)
                        # New column j: -b/g*col_k + a/g*col_j
                        new_col_k = x * A[:, k] + y * A[:, j]
                        new_col_j = (-b // g) * A[:, k] + (a // g) * A[:, j]
                        A[:, k], A[:, j] = new_col_k, new_col_j
                        
                        new_V_k = x * V[:, k] + y * V[:, j]
                        new_V_j = (-b // g) * V[:, k] + (a // g) * V[:, j]
                        V[:, k], V[:, j] = new_V_k, new_V_j
                        
                        pivot_found = False
            
            # Row operations
            for i in range(k + 1, n):
                if A[i, k] != 0:
                    if A[i, k] % A[k, k] == 0:
                        q = A[i, k] // A[k, k]
                        A[i, :] -= q * A[k, :]
                        U[i, :] -= q * U[k, :]
                    else:
                        g, x, y = extended_gcd(A[k, k], A[i, k])
                        a, b = A[k, k], A[i, k]
                        new_row_k = x * A[k, :] + y * A[i, :]
                        new_row_i = (-b // g) * A[k, :] + (a // g) * A[i, :]
                        A[k, :], A[i, :] = new_row_k, new_row_i
                        
                        new_U_k = x * U[k, :] + y * U[i, :]
                        new_U_i = (-b // g) * U[k, :] + (a // g) * U[i, :]
                        U[k, :], U[i, :] = new_U_k, new_U_i
                        
                        pivot_found = False
    
    # Ensure divisibility chain and positive diagonal
    D = A.copy()
    for i in range(n):
        if D[i, i] < 0:
            D[i, :] *= -1
            U[i, :] *= -1
    
    # Enforce divisibility: d1 | d2 | d3
    for i in range(n - 1):
        if D[i, i] != 0 and D[i+1, i+1] != 0:
            if D[i+1, i+1] % D[i, i] != 0:
                g = math.gcd(D[i, i], D[i+1, i+1])
                # This shouldn't happen for proper SNF, but handle it
                pass
    
    return D, U, V


# =============================================================================
# QUOTIENT GROUP G = Z³ / HZ³
# =============================================================================

@dataclass(frozen=True)
class GroupElement:
    """Element of quotient group, represented as tuple (i, j, k) mod H."""
    coords: Tuple[int, int, int]
    
    def __repr__(self):
        return f"({self.coords[0]},{self.coords[1]},{self.coords[2]})"
    
    def __hash__(self):
        return hash(self.coords)
    
    def __eq__(self, other):
        return self.coords == other.coords


class QuotientGroup:
    """
    Quotient group G = Z³ / HZ³ where H is an HNF matrix.
    
    Uses Smith Normal Form for correct arithmetic on all HNFs (diagonal and non-diagonal).
    
    SNF gives: U @ H @ V = D = diag(d1, d2, d3)
    The group is isomorphic to Z_{d1} × Z_{d2} × Z_{d3}.
    
    Elements are stored in the "SNF basis" where arithmetic is simple
    component-wise modular arithmetic.
    """
    
    def __init__(self, H: np.ndarray):
        """
        Initialize quotient group from HNF matrix.
        
        Args:
            H: 3×3 upper triangular HNF matrix
        """
        self.H = H.astype(np.int64).copy()
        
        # Compute Smith Normal Form
        self.D, self.U, self.V = smith_normal_form(H)
        
        # Precompute U inverse (exact integer computation)
        self.U_inv = integer_matrix_inverse(self.U)
        
        # Invariant factors (diagonal of D)
        self.d1 = int(self.D[0, 0]) if self.D[0, 0] != 0 else 1
        self.d2 = int(self.D[1, 1]) if self.D[1, 1] != 0 else 1
        self.d3 = int(self.D[2, 2]) if self.D[2, 2] != 0 else 1
        self.invariant_factors = (self.d1, self.d2, self.d3)
        
        # Group order
        self.order = self.d1 * self.d2 * self.d3
        
        # For diagonal HNFs, also store the simple form (for compatibility)
        self.is_diagonal = is_diagonal_hnf(H)
        if self.is_diagonal:
            self.a = int(H[0, 0])
            self.b = int(H[1, 1])
            self.c = int(H[2, 2])
        
        # Enumerate all elements
        self._elements = self._enumerate_elements()
        self._element_to_idx = {e: i for i, e in enumerate(self._elements)}
    
    def _enumerate_elements(self) -> List[GroupElement]:
        """Generate all group elements in SNF basis."""
        elements = []
        for k in range(self.d3):
            for j in range(self.d2):
                for i in range(self.d1):
                    elements.append(GroupElement((i, j, k)))
        return elements
    
    def _to_snf_basis(self, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert vector from original basis to SNF basis."""
        v_arr = np.array(v, dtype=np.int64)
        u = self.U @ v_arr
        return (int(u[0]) % self.d1, int(u[1]) % self.d2, int(u[2]) % self.d3)
    
    def _from_snf_basis(self, u: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Convert vector from SNF basis back to original basis."""
        u_arr = np.array(u, dtype=np.int64)
        v = self.U_inv @ u_arr
        return (int(v[0]), int(v[1]), int(v[2]))
    
    def reduce(self, v: Tuple[int, int, int]) -> GroupElement:
        """
        Reduce a Z³ vector to its canonical representative in G.
        
        Uses SNF for correct reduction on all HNFs.
        """
        snf_coords = self._to_snf_basis(v)
        return GroupElement(snf_coords)
    
    def add(self, g1: GroupElement, g2: GroupElement) -> GroupElement:
        """Group operation: component-wise addition mod invariant factors."""
        i = (g1.coords[0] + g2.coords[0]) % self.d1
        j = (g1.coords[1] + g2.coords[1]) % self.d2
        k = (g1.coords[2] + g2.coords[2]) % self.d3
        return GroupElement((i, j, k))
    
    def neg(self, g: GroupElement) -> GroupElement:
        """Group inverse: component-wise negation mod invariant factors."""
        i = (-g.coords[0]) % self.d1
        j = (-g.coords[1]) % self.d2
        k = (-g.coords[2]) % self.d3
        return GroupElement((i, j, k))
    
    def identity(self) -> GroupElement:
        """Return identity element."""
        return GroupElement((0, 0, 0))
    
    def to_fractional(self, g: GroupElement) -> Tuple[float, float, float]:
        """
        DEPRECATED: Use build_fractional_rep_map(H, G) instead.
        
        This method is not guaranteed correct for non-diagonal HNFs.
        Kept for backward compatibility but should not be used for
        geometry/coordination calculations.
        
        For correct fractional coordinates, use:
            frac_map = build_fractional_rep_map(H, G)
            frac = frac_map[g]
        """
        if self.is_diagonal:
            return (g.coords[0] / self.d1, g.coords[1] / self.d2, g.coords[2] / self.d3)
        else:
            # WARNING: This may not give correct coset representatives
            orig = self._from_snf_basis(g.coords)
            H_inv = np.linalg.inv(self.H.astype(np.float64))
            v = np.array(orig, dtype=np.float64)
            frac = H_inv @ v
            frac = frac - np.floor(frac)
            return (float(frac[0]), float(frac[1]), float(frac[2]))
    
    @property
    def elements(self) -> List[GroupElement]:
        """All group elements."""
        return self._elements
    
    def __len__(self) -> int:
        return self.order
    
    def __repr__(self):
        if self.is_diagonal:
            return f"QuotientGroup(order={self.order}, H=diag({self.a},{self.b},{self.c}))"
        else:
            return f"QuotientGroup(order={self.order}, factors={self.invariant_factors})"


# =============================================================================
# SUBGROUP ENUMERATION
# =============================================================================

def is_subgroup(G: QuotientGroup, elements: Set[GroupElement]) -> bool:
    """
    Check if a set of elements forms a subgroup.
    
    Criteria:
    1. Contains identity
    2. Closed under group operation
    3. Closed under inverses
    """
    if G.identity() not in elements:
        return False
    
    for g1 in elements:
        # Check inverse
        if G.neg(g1) not in elements:
            return False
        
        # Check closure
        for g2 in elements:
            if G.add(g1, g2) not in elements:
                return False
    
    return True


def get_divisors(n: int) -> List[int]:
    """Get all positive divisors of n, sorted."""
    if n <= 0:
        return []
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def subgroup_closure(G: QuotientGroup, generators: Set[GroupElement]) -> Set[GroupElement]:
    """
    Compute the subgroup generated by a set of generators.
    
    Uses BFS-style closure: only compute products involving new elements.
    """
    subgroup = {G.identity()}
    frontier = set(generators) - subgroup
    subgroup.update(generators)
    
    # Process frontier: compute products with frontier elements
    while frontier:
        new_frontier = set()
        for g in frontier:
            for h in subgroup:
                prod = G.add(g, h)
                if prod not in subgroup:
                    new_frontier.add(prod)
        subgroup.update(new_frontier)
        frontier = new_frontier
    
    return subgroup


def enumerate_subgroups_generator(G: QuotientGroup, size: int) -> List[Set[GroupElement]]:
    """
    Enumerate all subgroups of G with given size using generator approach.
    
    Strategy: Try all possible generator sets and compute their closure.
    A subgroup of order k in an abelian group needs at most rank(G) generators.
    
    Optimizations:
    - Early termination when closure exceeds target size
    - Skip generators that individually generate larger subgroups
    - Use efficient closure algorithm
    """
    if len(G) % size != 0:
        return []  # Lagrange's theorem
    
    if size == 1:
        return [{G.identity()}]
    
    if size == len(G):
        return [set(G.elements)]
    
    found_subgroups: Set[frozenset] = set()
    elements = list(G.elements)
    identity = G.identity()
    non_identity = [e for e in elements if e != identity]
    
    # Precompute cyclic subgroup sizes for each element
    cyclic_sizes = {}
    for g in non_identity:
        sub = subgroup_closure(G, {g})
        cyclic_sizes[g] = len(sub)
        if len(sub) == size:
            found_subgroups.add(frozenset(sub))
    
    # For 2-generator subgroups, only try pairs where both 
    # generators have cyclic order dividing the target size
    d1, d2, d3 = G.invariant_factors
    rank = sum(1 for d in [d1, d2, d3] if d > 1)
    
    if rank >= 2:
        # Filter to elements whose cyclic order divides target size
        valid_gens = [g for g in non_identity if size % cyclic_sizes[g] == 0]
        
        for i, g1 in enumerate(valid_gens):
            for g2 in valid_gens[i+1:]:
                # Skip if g2 is in the cyclic group of g1
                sub1 = subgroup_closure(G, {g1})
                if g2 in sub1:
                    continue
                    
                sub = subgroup_closure(G, {g1, g2})
                if len(sub) == size:
                    found_subgroups.add(frozenset(sub))
    
    if rank >= 3:
        valid_gens = [g for g in non_identity if size % cyclic_sizes[g] == 0]
        
        for i, g1 in enumerate(valid_gens):
            sub1 = subgroup_closure(G, {g1})
            for j, g2 in enumerate(valid_gens[i+1:], i+1):
                if g2 in sub1:
                    continue
                sub2 = subgroup_closure(G, {g1, g2})
                if len(sub2) == size:
                    continue  # Already found
                if len(sub2) > size:
                    continue  # Can't shrink
                    
                for g3 in valid_gens[j+1:]:
                    if g3 in sub2:
                        continue
                    sub = subgroup_closure(G, {g1, g2, g3})
                    if len(sub) == size:
                        found_subgroups.add(frozenset(sub))
    
    return [set(fs) for fs in found_subgroups]
    
    return [set(fs) for fs in found_subgroups]


def enumerate_subgroups(G: QuotientGroup, size: int) -> List[Set[GroupElement]]:
    """
    Enumerate all subgroups of G with given size.
    
    Uses a hybrid approach:
    - For |G| ≤ 12: brute force (faster for small groups)
    - For |G| > 12: generator-based (avoids C(n,k) explosion)
    
    For abelian groups, every subgroup can be generated by at most
    rank(G) generators, which is at most 3 for our 3D lattice groups.
    """
    if len(G) <= 12:
        return enumerate_subgroups_brute_force(G, size)
    else:
        return enumerate_subgroups_generator(G, size)


def enumerate_subgroups_brute_force(G: QuotientGroup, size: int) -> List[Set[GroupElement]]:
    """
    Enumerate all subgroups of G with given size (brute force method).
    
    DEPRECATED: Use enumerate_subgroups() which uses generator-based enumeration.
    Kept for testing/validation purposes.
    
    This is O(C(|G|, size)) which becomes expensive for larger groups.
    """
    if len(G) % size != 0:
        return []  # Lagrange's theorem
    
    subgroups = []
    identity = G.identity()
    other_elements = [e for e in G.elements if e != identity]
    
    # Subgroup must contain identity, so choose (size-1) from the rest
    for subset in combinations(other_elements, size - 1):
        candidate = {identity} | set(subset)
        if is_subgroup(G, candidate):
            subgroups.append(candidate)
    
    return subgroups


def enumerate_subgroups_cached(G: QuotientGroup) -> Dict[int, List[Set[GroupElement]]]:
    """
    Enumerate all subgroups of G, organized by size.
    
    Returns dict mapping size -> list of subgroups of that size.
    """
    result = {}
    
    # Only check sizes that divide |G| (Lagrange's theorem)
    for size in range(1, len(G) + 1):
        if len(G) % size == 0:
            subs = enumerate_subgroups(G, size)
            if subs:
                result[size] = subs
    
    return result


# =============================================================================
# COSET GENERATION
# =============================================================================

def generate_coset(G: QuotientGroup, subgroup: Set[GroupElement], 
                   shift: GroupElement) -> Set[GroupElement]:
    """
    Generate coset: shift + subgroup = {shift + h : h ∈ subgroup}.
    """
    return {G.add(shift, h) for h in subgroup}


def generate_all_cosets(G: QuotientGroup, 
                        subgroup: Set[GroupElement]) -> List[Set[GroupElement]]:
    """
    Generate all distinct cosets of a subgroup.
    
    Number of cosets = |G| / |subgroup| (index of subgroup).
    """
    cosets = []
    covered = set()
    
    for g in G.elements:
        if g in covered:
            continue
        
        coset = generate_coset(G, subgroup, g)
        cosets.append(coset)
        covered.update(coset)
    
    return cosets


# =============================================================================
# ORBIT CONFIGURATION SEARCH
# =============================================================================

@dataclass
class OrbitConfiguration:
    """
    A valid configuration of orbits partitioning the group G.
    
    Attributes:
        orbits: List of orbit sets, one per sublattice
        orbit_sizes: Tuple of orbit sizes (for validation)
        subgroups: The subgroups used to generate each orbit
        shifts: The shifts applied to each subgroup
    """
    orbits: List[Set[GroupElement]]
    orbit_sizes: Tuple[int, ...]
    subgroups: Optional[List[Set[GroupElement]]] = None
    shifts: Optional[List[GroupElement]] = None
    
    def __repr__(self):
        sizes = tuple(len(o) for o in self.orbits)
        return f"OrbitConfig(sizes={sizes}, n_orbits={len(self.orbits)})"
    
    def to_offset_list(self) -> List[List[Tuple[int, int, int]]]:
        """Convert to list of offset coordinate lists."""
        return [[g.coords for g in orbit] for orbit in self.orbits]


def search_configurations(
    G: QuotientGroup,
    orbit_sizes: Tuple[int, ...],
    gauge_fix: bool = True,
    max_results: Optional[int] = None
) -> Generator[OrbitConfiguration, None, None]:
    """
    Search for valid orbit configurations in G.
    
    An orbit configuration partitions G into orbits of specified sizes.
    Each orbit is a coset of some subgroup.
    
    Args:
        G: The quotient group to partition
        orbit_sizes: Tuple specifying size of each orbit (e.g., (2, 4) for spinel)
        gauge_fix: If True, fix first orbit to contain identity (reduces redundancy)
        max_results: Stop after finding this many configurations
    
    Yields:
        OrbitConfiguration objects representing valid partitions
    """
    # Validate total size
    if sum(orbit_sizes) != len(G):
        raise ValueError(f"Orbit sizes {orbit_sizes} don't sum to |G|={len(G)}")
    
    # Precompute subgroups by size
    subgroups_by_size = enumerate_subgroups_cached(G)
    
    # Check we have subgroups for all needed sizes
    for size in orbit_sizes:
        if size not in subgroups_by_size:
            return  # No valid configurations possible
    
    n_found = 0
    
    def search_recursive(
        remaining_elements: Set[GroupElement],
        current_orbits: List[Set[GroupElement]],
        current_subgroups: List[Set[GroupElement]],
        current_shifts: List[GroupElement],
        orbit_idx: int
    ) -> Generator[OrbitConfiguration, None, None]:
        """Recursive backtracking search."""
        nonlocal n_found
        
        if max_results and n_found >= max_results:
            return
        
        # Base case: all orbits assigned
        if orbit_idx == len(orbit_sizes):
            if len(remaining_elements) == 0:
                n_found += 1
                yield OrbitConfiguration(
                    orbits=list(current_orbits),
                    orbit_sizes=orbit_sizes,
                    subgroups=list(current_subgroups),
                    shifts=list(current_shifts)
                )
            return
        
        target_size = orbit_sizes[orbit_idx]
        
        # Get candidate subgroups for this orbit
        candidate_subgroups = subgroups_by_size.get(target_size, [])
        
        for subgroup in candidate_subgroups:
            # Try different shifts
            for shift in remaining_elements:
                # Gauge fix: first orbit must contain identity
                if gauge_fix and orbit_idx == 0:
                    if G.identity() not in generate_coset(G, subgroup, shift):
                        continue
                
                coset = generate_coset(G, subgroup, shift)
                
                # Check coset is subset of remaining elements
                if not coset.issubset(remaining_elements):
                    continue
                
                # Recurse
                new_remaining = remaining_elements - coset
                current_orbits.append(coset)
                current_subgroups.append(subgroup)
                current_shifts.append(shift)
                
                yield from search_recursive(
                    new_remaining,
                    current_orbits,
                    current_subgroups,
                    current_shifts,
                    orbit_idx + 1
                )
                
                current_orbits.pop()
                current_subgroups.pop()
                current_shifts.pop()
    
    # Start search
    yield from search_recursive(
        remaining_elements=set(G.elements),
        current_orbits=[],
        current_subgroups=[],
        current_shifts=[],
        orbit_idx=0
    )


def search_configurations_list(
    G: QuotientGroup,
    orbit_sizes: Tuple[int, ...],
    gauge_fix: bool = True,
    max_results: Optional[int] = None
) -> List[OrbitConfiguration]:
    """Convenience wrapper returning list instead of generator."""
    return list(search_configurations(G, orbit_sizes, gauge_fix, max_results))


def search_subset_configurations(
    G: QuotientGroup,
    orbit_sizes: Tuple[int, ...],
    gauge_fix: bool = True,
    max_results: Optional[int] = None
) -> Generator[OrbitConfiguration, None, None]:
    """
    Search for disjoint cosets of specified sizes, NOT requiring full partition.
    
    This is the correct formulation for problems like spinel:
    - G has M elements (e.g., M=8 for Z_2^3)
    - We want N = sum(orbit_sizes) offsets (e.g., N=6 for spinel)
    - Select disjoint cosets matching orbit_sizes
    - Leave M-N elements unused
    
    Args:
        G: The quotient group (offset pool)
        orbit_sizes: Tuple specifying size of each orbit (e.g., (2, 4))
        gauge_fix: If True, fix first orbit to contain identity
        max_results: Stop after finding this many configurations
    
    Yields:
        OrbitConfiguration objects representing valid selections
    """
    N = sum(orbit_sizes)
    
    if N > len(G):
        raise ValueError(f"sum(orbit_sizes)={N} exceeds |G|={len(G)}")
    
    # Check Lagrange: each orbit size must divide |G|
    for size in orbit_sizes:
        if len(G) % size != 0:
            return  # No subgroups of this size exist
    
    # Precompute subgroups by size
    subgroups_by_size = enumerate_subgroups_cached(G)
    
    # Check we have subgroups for all needed sizes
    for size in orbit_sizes:
        if size not in subgroups_by_size:
            return
    
    n_found = 0
    
    def search_recursive(
        available_elements: Set[GroupElement],
        current_orbits: List[Set[GroupElement]],
        current_subgroups: List[Set[GroupElement]],
        current_shifts: List[GroupElement],
        orbit_idx: int
    ) -> Generator[OrbitConfiguration, None, None]:
        """Recursive backtracking search for disjoint cosets."""
        nonlocal n_found
        
        if max_results and n_found >= max_results:
            return
        
        # Base case: all orbits assigned
        if orbit_idx == len(orbit_sizes):
            n_found += 1
            yield OrbitConfiguration(
                orbits=list(current_orbits),
                orbit_sizes=orbit_sizes,
                subgroups=list(current_subgroups),
                shifts=list(current_shifts)
            )
            return
        
        target_size = orbit_sizes[orbit_idx]
        candidate_subgroups = subgroups_by_size.get(target_size, [])
        
        # To avoid redundant exploration, iterate shifts in canonical order
        sorted_available = sorted(available_elements, key=lambda g: g.coords)
        
        for subgroup in candidate_subgroups:
            for shift in sorted_available:
                # Gauge fix: first orbit must contain identity
                if gauge_fix and orbit_idx == 0:
                    coset = generate_coset(G, subgroup, shift)
                    if G.identity() not in coset:
                        continue
                else:
                    coset = generate_coset(G, subgroup, shift)
                
                # Check coset is subset of available elements
                if not coset.issubset(available_elements):
                    continue
                
                # Recurse with this coset removed from available
                new_available = available_elements - coset
                current_orbits.append(coset)
                current_subgroups.append(subgroup)
                current_shifts.append(shift)
                
                yield from search_recursive(
                    new_available,
                    current_orbits,
                    current_subgroups,
                    current_shifts,
                    orbit_idx + 1
                )
                
                current_orbits.pop()
                current_subgroups.pop()
                current_shifts.pop()
    
    # Start search from full group
    yield from search_recursive(
        available_elements=set(G.elements),
        current_orbits=[],
        current_subgroups=[],
        current_shifts=[],
        orbit_idx=0
    )


def search_subset_configurations_list(
    G: QuotientGroup,
    orbit_sizes: Tuple[int, ...],
    gauge_fix: bool = True,
    max_results: Optional[int] = None
) -> List[OrbitConfiguration]:
    """Convenience wrapper returning list instead of generator."""
    return list(search_subset_configurations(G, orbit_sizes, gauge_fix, max_results))


# =============================================================================
# EQUIVALENCE AND DEDUPLICATION
# =============================================================================

def configuration_signature(config: OrbitConfiguration) -> Tuple:
    """
    Compute a canonical signature for a configuration.
    
    This helps identify equivalent configurations under:
    - Permutation of orbits of the same size
    - Translation of all orbits by the same offset
    
    For now, we use a simple sorted tuple of sorted orbit tuples.
    """
    # Sort elements within each orbit, then sort orbits
    sorted_orbits = tuple(
        tuple(sorted(g.coords for g in orbit))
        for orbit in sorted(config.orbits, key=lambda o: tuple(sorted(g.coords for g in o)))
    )
    return sorted_orbits


def deduplicate_configurations(
    configs: List[OrbitConfiguration]
) -> List[OrbitConfiguration]:
    """
    Remove equivalent configurations based on signature.
    
    Note: This is basic deduplication that handles orbit ordering.
    For full canonicalization including translation equivalence,
    use deduplicate_with_canonicalization() instead.
    """
    seen = set()
    unique = []
    
    for config in configs:
        sig = configuration_signature(config)
        if sig not in seen:
            seen.add(sig)
            unique.append(config)
    
    return unique


def canonicalize_configuration(
    config: OrbitConfiguration,
    G: QuotientGroup,
    frac_map: Dict[GroupElement, np.ndarray]
) -> OrbitConfiguration:
    """
    Canonicalize a configuration by translation.
    
    Shifts all group elements so that the lexicographically smallest
    fractional coordinate among all sites becomes the origin.
    This removes translation-equivalent duplicates.
    
    Args:
        config: Original configuration
        G: Quotient group
        frac_map: Fractional coordinate map
    
    Returns:
        New OrbitConfiguration with canonical translation
    """
    # Find the lexicographically smallest fractional coordinate
    all_elements = []
    for orbit in config.orbits:
        all_elements.extend(orbit)
    
    if not all_elements:
        return config
    
    # Get fractional coords and find lex-smallest
    fracs_with_elements = [(tuple(frac_map[g]), g) for g in all_elements]
    
    # Sort by fractional coordinates (lex order)
    fracs_with_elements.sort(key=lambda x: x[0])
    
    # The element with lex-smallest frac becomes the "origin"
    _, origin_element = fracs_with_elements[0]
    
    # Compute shift: we want origin_element to map to identity
    # So we subtract origin_element from all elements
    neg_origin = G.neg(origin_element)
    
    # Shift all orbits
    new_orbits = []
    for orbit in config.orbits:
        new_orbit = frozenset(G.add(g, neg_origin) for g in orbit)
        new_orbits.append(new_orbit)
    
    return OrbitConfiguration(
        orbits=new_orbits,
        orbit_sizes=config.orbit_sizes,
        subgroups=config.subgroups,
        shifts=None  # Shifts are no longer valid after canonicalization
    )


def deduplicate_with_canonicalization(
    configs: List[OrbitConfiguration],
    G: QuotientGroup,
    frac_map: Dict[GroupElement, np.ndarray]
) -> List[OrbitConfiguration]:
    """
    Deduplicate configurations with translation canonicalization.
    
    This is stronger than basic deduplication - it identifies
    configurations that are equivalent under translation.
    
    Args:
        configs: List of configurations to deduplicate
        G: Quotient group
        frac_map: Fractional coordinate map
    
    Returns:
        List of unique configurations (canonicalized)
    """
    seen = set()
    unique = []
    
    for config in configs:
        # Canonicalize first
        canonical = canonicalize_configuration(config, G, frac_map)
        sig = configuration_signature(canonical)
        
        if sig not in seen:
            seen.add(sig)
            unique.append(canonical)
    
    return unique


def analyzed_configuration_signature(analyzed: 'AnalyzedConfiguration') -> Tuple:
    """
    Generate a signature for an analyzed configuration based on geometry.
    
    Two configs with the same signature have identical CN patterns
    and shell structures (up to numerical tolerance).
    """
    # Build signature from orbit sizes and CN patterns
    orbit_sigs = []
    for orbit_idx in sorted(analyzed.signatures.keys()):
        sig = analyzed.signatures[orbit_idx]
        orbit_sig = (
            len(analyzed.config.orbits[orbit_idx]),  # Size
            sig.cn1,
            tuple(sorted(sig.cn1_by_orbit.items())),
            sig.cn2,
            tuple(sorted(sig.cn2_by_orbit.items())),
            # First shell distance (rounded to avoid float issues)
            round(sig.shells[0].distance, 4) if sig.shells else 0.0
        )
        orbit_sigs.append(orbit_sig)
    
    # Sort by orbit signature for canonical ordering
    return tuple(sorted(orbit_sigs))


def deduplicate_analyzed_configurations(
    configs: List['AnalyzedConfiguration']
) -> List['AnalyzedConfiguration']:
    """
    Deduplicate analyzed configurations based on geometric signature.
    
    Removes configurations that have identical CN patterns and shell
    structures, even if they came from different HNF matrices.
    """
    seen = set()
    unique = []
    
    for config in configs:
        sig = analyzed_configuration_signature(config)
        if sig not in seen:
            seen.add(sig)
            unique.append(config)
    
    return unique


# =============================================================================
# CONVENIENCE: SEARCH ACROSS HNFs
# =============================================================================

def search_subset_across_hnfs(
    M_values: List[int],
    orbit_sizes: Tuple[int, ...],
    diagonal_only: bool = True,
    max_per_hnf: int = 50,
    deduplicate: bool = True
) -> Dict[str, List[OrbitConfiguration]]:
    """
    Search for subset configurations across multiple HNFs.
    
    This is the primary interface for spinel-style searches:
    - orbit_sizes defines the partition (e.g., (2, 4) for spinel)
    - N = sum(orbit_sizes) offsets are selected from M-element groups
    - M - N elements are left unused
    
    Args:
        M_values: List of supercell sizes to try (must be >= N)
        orbit_sizes: Target orbit sizes (e.g., (2, 4))
        diagonal_only: If True, only use diagonal HNFs (safe until SNF implemented)
        max_per_hnf: Max configurations per HNF
        deduplicate: Remove equivalent configurations per HNF
    
    Returns:
        Dict mapping HNF description to list of configurations
    """
    N = sum(orbit_sizes)
    results = {}
    
    for M in M_values:
        if M < N:
            print(f"Warning: M={M} < N={N}, skipping")
            continue
        
        # Check Lagrange compatibility
        lagrange_ok = all(M % size == 0 for size in orbit_sizes)
        if not lagrange_ok:
            continue
        
        if diagonal_only:
            hnfs = enumerate_diagonal_hnf(M)
        else:
            hnfs = enumerate_hnf(M)
        
        for H in hnfs:
            G = QuotientGroup(H)
            
            configs = search_subset_configurations_list(
                G, orbit_sizes, max_results=max_per_hnf
            )
            
            if deduplicate:
                configs = deduplicate_configurations(configs)
            
            if configs:
                key = hnf_to_string(H)
                results[key] = configs
    
    return results


def search_all_hnf(
    M_values: List[int],
    orbit_sizes: Tuple[int, ...],
    max_per_hnf: int = 10,
    deduplicate: bool = True
) -> Dict[str, List[OrbitConfiguration]]:
    """
    Search for configurations across all HNFs for given M values.
    
    Args:
        M_values: List of supercell sizes to try
        orbit_sizes: Target orbit sizes (e.g., (2, 4) for spinel)
        max_per_hnf: Max configurations to find per HNF
        deduplicate: Whether to remove equivalent configurations
    
    Returns:
        Dict mapping HNF description to list of configurations
    """
    # Validate orbit sizes sum
    total_sites = sum(orbit_sizes)
    
    results = {}
    
    for M in M_values:
        if M != total_sites:
            print(f"Warning: M={M} != sum(orbit_sizes)={total_sites}, skipping")
            continue
            
        hnfs = enumerate_hnf(M)
        
        for H in hnfs:
            G = QuotientGroup(H)
            
            configs = search_configurations_list(G, orbit_sizes, max_results=max_per_hnf)
            
            if deduplicate:
                configs = deduplicate_configurations(configs)
            
            if configs:
                key = hnf_to_string(H)
                results[key] = configs
    
    return results


# =============================================================================
# QUOTIENT GROUP CORRECTNESS TESTS
# =============================================================================

def test_quotient_group_correctness(
    H: np.ndarray, 
    G: QuotientGroup, 
    n_samples: int = 100,
    verbose: bool = False
) -> bool:
    """
    Test that QuotientGroup.reduce() correctly implements Z³/HZ³.
    
    Test A: Invariance under adding lattice vectors
        reduce(x) == reduce(x + H @ m) for any m ∈ Z³
    
    Test B: Group homomorphism
        reduce(x + y) == add(reduce(x), reduce(y))
    
    Args:
        H: HNF matrix
        G: QuotientGroup built from H
        n_samples: Number of random tests
        verbose: Print failures
    
    Returns:
        True if all tests pass
    """
    rng = np.random.default_rng(42)
    all_passed = True
    
    # Test A: Invariance under lattice translation
    for _ in range(n_samples):
        x = rng.integers(-10, 10, size=3)
        m = rng.integers(-5, 5, size=3)
        
        x_shifted = x + H @ m
        
        g1 = G.reduce(tuple(x))
        g2 = G.reduce(tuple(x_shifted))
        
        if g1 != g2:
            if verbose:
                print(f"FAIL Test A: reduce({x}) = {g1} != reduce({x} + H@{m}) = {g2}")
            all_passed = False
    
    # Test B: Homomorphism
    for _ in range(n_samples):
        x = rng.integers(-10, 10, size=3)
        y = rng.integers(-10, 10, size=3)
        
        # reduce(x + y)
        g_sum = G.reduce(tuple(x + y))
        
        # add(reduce(x), reduce(y))
        gx = G.reduce(tuple(x))
        gy = G.reduce(tuple(y))
        g_add = G.add(gx, gy)
        
        if g_sum != g_add:
            if verbose:
                print(f"FAIL Test B: reduce({x}+{y}) = {g_sum} != add(reduce({x}), reduce({y})) = {g_add}")
            all_passed = False
    
    return all_passed


def test_fractional_map_consistency(
    H: np.ndarray,
    G: QuotientGroup,
    frac_map: Dict[GroupElement, np.ndarray],
    n_samples: int = 50,
    verbose: bool = False
) -> bool:
    """
    Test that fractional rep map is consistent with group structure.
    
    Test 1: Map size equals group order
        len(frac_map) == |G| == det(H)
    
    Test 2: Consistency with group addition
        wrap(frac(g) + frac(h)) ≡ frac(add(g, h)) mod 1
    
    Args:
        H: HNF matrix
        G: QuotientGroup
        frac_map: Fractional rep map from build_fractional_rep_map
        n_samples: Number of random pair tests
        verbose: Print failures
    
    Returns:
        True if all tests pass
    """
    all_passed = True
    
    # Verify H is actually HNF before using diagonal product for determinant
    if not is_hnf(H):
        if verbose:
            print(f"FAIL: H is not a valid HNF matrix")
        return False
    
    # For HNF (upper triangular), det = product of diagonal entries
    # This would NOT be valid for general matrices
    expected_size = hnf_determinant(H)
    
    # Test 1: Size check
    if len(frac_map) != expected_size:
        if verbose:
            print(f"FAIL: len(frac_map) = {len(frac_map)} != det(H) = {expected_size}")
        all_passed = False
    
    if len(frac_map) != len(G):
        if verbose:
            print(f"FAIL: len(frac_map) = {len(frac_map)} != |G| = {len(G)}")
        all_passed = False
    
    # Test 2: Consistency with group addition
    elements = list(frac_map.keys())
    rng = np.random.default_rng(42)
    
    def periodic_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
        """Check if two fractional coordinates are equivalent mod 1."""
        diff = a - b
        # Wrap difference to [-0.5, 0.5)
        diff = diff - np.round(diff)
        return np.allclose(diff, 0, atol=tol)
    
    for _ in range(min(n_samples, len(elements) * len(elements))):
        g = elements[rng.integers(len(elements))]
        h = elements[rng.integers(len(elements))]
        
        # frac(g) + frac(h) 
        frac_g = frac_map[g]
        frac_h = frac_map[h]
        frac_sum = frac_g + frac_h
        
        # frac(add(g, h))
        g_plus_h = G.add(g, h)
        frac_gh = frac_map[g_plus_h]
        
        # Check equivalence modulo 1 (handles 0.999... ≈ 0.0 case)
        if not periodic_close(frac_sum, frac_gh):
            if verbose:
                print(f"FAIL: frac({g}) + frac({h}) = {frac_sum}")
                print(f"       != frac(add({g}, {h})) = {frac_gh} (mod 1)")
            all_passed = False
    
    return all_passed


def run_all_correctness_tests(M_values: List[int] = [4, 6, 8], verbose: bool = True) -> bool:
    """
    Run correctness tests for quotient groups and fractional maps.
    
    Tests both diagonal and non-diagonal HNFs for each M value.
    For small groups (M ≤ 16), tests ALL pairs for fractional map consistency.
    Also validates SNF-based subgroup enumeration against brute force.
    """
    print("=" * 70)
    print("CORRECTNESS TESTS")
    print("=" * 70)
    
    all_passed = True
    
    for M in M_values:
        print(f"\nM = {M}:")
        hnfs = enumerate_hnf(M)
        
        # Test a sample of HNFs (all diagonal + some non-diagonal)
        test_hnfs = [H for H in hnfs if is_diagonal_hnf(H)]
        non_diag = [H for H in hnfs if not is_diagonal_hnf(H)]
        if non_diag:
            test_hnfs.extend(non_diag[:min(5, len(non_diag))])  # Add up to 5 non-diagonal
        
        n_passed = 0
        n_tested = 0
        
        for H in test_hnfs:
            G = QuotientGroup(H)
            
            # Test quotient group (100 random samples)
            qg_ok = test_quotient_group_correctness(H, G, n_samples=100, verbose=False)
            
            # Test fractional map (all pairs for small groups)
            frac_map = build_fractional_rep_map(H, G)
            n_pairs = len(G) * len(G) if M <= 16 else 100
            fm_ok = test_fractional_map_consistency(H, G, frac_map, n_samples=n_pairs, verbose=False)
            
            # Test SNF subgroup enumeration (for small M only)
            sg_ok = True
            if M <= 8:
                sg_ok = test_snf_subgroup_enumeration(G, verbose=False)
            
            n_tested += 1
            if qg_ok and fm_ok and sg_ok:
                n_passed += 1
            else:
                all_passed = False
                if verbose:
                    hnf_str = hnf_to_string(H)
                    print(f"  ✗ {hnf_str}: QG={qg_ok}, FM={fm_ok}, SG={sg_ok}")
        
        # Summary for this M
        diag_count = len([H for H in test_hnfs if is_diagonal_hnf(H)])
        nondiag_count = len(test_hnfs) - diag_count
        status = "ALL PASSED ✓" if n_passed == n_tested else f"{n_passed}/{n_tested} PASSED"
        print(f"  Tested {diag_count} diagonal + {nondiag_count} non-diagonal HNFs: {status}")
    
    return all_passed


def test_snf_subgroup_enumeration(G: QuotientGroup, verbose: bool = False) -> bool:
    """
    Validate generator-based subgroup enumeration against brute force.
    
    For each valid subgroup size, checks that:
    1. Generator method finds the same number of subgroups
    2. All subgroups found are valid subgroups
    3. Both methods find the same subgroups (as sets)
    """
    all_passed = True
    
    for size in range(1, len(G) + 1):
        if len(G) % size != 0:
            continue
        
        gen_subs = enumerate_subgroups_generator(G, size)
        bf_subs = enumerate_subgroups_brute_force(G, size)
        
        # Check count matches
        if len(gen_subs) != len(bf_subs):
            if verbose:
                print(f"FAIL: size={size}, Generator found {len(gen_subs)}, BF found {len(bf_subs)}")
            all_passed = False
            continue
        
        # Check all generator subgroups are valid
        for sub in gen_subs:
            if not is_subgroup(G, sub):
                if verbose:
                    print(f"FAIL: Generator subgroup {sub} is not a valid subgroup")
                all_passed = False
        
        # Check same subgroups found (as frozen sets for comparison)
        gen_frozen = {frozenset(s) for s in gen_subs}
        bf_frozen = {frozenset(s) for s in bf_subs}
        
        if gen_frozen != bf_frozen:
            if verbose:
                print(f"FAIL: size={size}, Generator and BF found different subgroups")
                print(f"  Generator only: {gen_frozen - bf_frozen}")
                print(f"  BF only: {bf_frozen - gen_frozen}")
            all_passed = False
    
    return all_passed
    
    return all_passed


# =============================================================================
# PHASE 2: GEOMETRY AND COORDINATION ENVIRONMENTS
# =============================================================================

def build_fractional_rep_map(
    H: np.ndarray, 
    G: QuotientGroup,
    validate: bool = True
) -> Dict[GroupElement, np.ndarray]:
    """
    Build a map from group elements to fractional offsets in [0,1)³.
    
    This is the CORRECT general method for all HNFs (diagonal and non-diagonal).
    
    For each integer vector k in the HNF fundamental domain:
    1. k ∈ [0, h11) × [0, h22) × [0, h33)
    2. Compute fractional offset: t = H^{-1} k (mod 1)
    3. Reduce k to group element g using SNF machinery
    4. Store map[g] = t
    
    This guarantees every group element maps to a true coset representative
    in real space, independent of SNF coordinate choices.
    
    Args:
        H: 3×3 HNF matrix (upper triangular)
        G: QuotientGroup built from H
        validate: If True, assert consistency checks
    
    Returns:
        Dict mapping GroupElement → fractional offset as (3,) array
    """
    # Verify H is HNF before proceeding
    assert is_hnf(H), f"Matrix must be HNF (upper triangular with positive diagonal): {H}"
    
    H_inv = np.linalg.inv(H.astype(np.float64))
    frac_map = {}
    
    # For HNF (upper triangular), det = product of diagonal entries
    # We use diagonals directly for enumeration bounds
    h11, h22, h33 = int(H[0, 0]), int(H[1, 1]), int(H[2, 2])
    expected_size = h11 * h22 * h33  # = det(H) for upper triangular
    
    for k3 in range(h33):
        for k2 in range(h22):
            for k1 in range(h11):
                k = np.array([k1, k2, k3], dtype=np.float64)
                
                # Fractional offset: t = H^{-1} k (mod 1)
                t = H_inv @ k
                t = t - np.floor(t)  # Reduce to [0, 1)
                
                # Reduce k to group element using SNF
                g = G.reduce((k1, k2, k3))
                
                # Store (first one wins if there are duplicates, 
                # but there shouldn't be for proper HNF)
                if g not in frac_map:
                    frac_map[g] = t
    
    if validate:
        # Assert 1: Map size equals group order
        assert len(frac_map) == expected_size, \
            f"frac_map size {len(frac_map)} != det(H) = {expected_size}"
        assert len(frac_map) == len(G), \
            f"frac_map size {len(frac_map)} != |G| = {len(G)}"
    
    return frac_map


# Standard Bravais lattice matrices (columns are primitive vectors)
# FCC primitive: a1 = (0, 1/2, 1/2), a2 = (1/2, 0, 1/2), a3 = (1/2, 1/2, 0)
FCC_PRIMITIVE = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0]
], dtype=np.float64).T  # Columns are lattice vectors

# BCC primitive: a1 = (-1/2, 1/2, 1/2), a2 = (1/2, -1/2, 1/2), a3 = (1/2, 1/2, -1/2)
BCC_PRIMITIVE = np.array([
    [-0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5]
], dtype=np.float64).T

# Simple cubic
SC_PRIMITIVE = np.eye(3, dtype=np.float64)


def lattice_matrix_from_params(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """
    Build lattice matrix from cell parameters (general triclinic).
    
    Args:
        a, b, c: Lattice vector lengths
        alpha, beta, gamma: Angles in degrees (α=bc, β=ac, γ=ab)
    
    Returns:
        3×3 matrix with lattice vectors as columns
    """
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)
    
    # Standard crystallographic convention:
    # a along x, b in xy-plane, c general
    ax = a
    bx = b * cos_gamma
    by = b * sin_gamma
    cx = c * cos_beta
    cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    
    return np.array([
        [ax, bx, cx],
        [0.0, by, cy],
        [0.0, 0.0, cz]
    ], dtype=np.float64)


def fractional_to_cartesian(frac: Tuple[float, float, float], 
                            lattice: np.ndarray = FCC_PRIMITIVE,
                            a: float = 1.0) -> np.ndarray:
    """
    Convert fractional coordinates to Cartesian.
    
    Args:
        frac: Fractional coordinates (f1, f2, f3)
        lattice: 3×3 matrix with lattice vectors as columns
        a: Lattice parameter (conventional cell edge length)
    
    Returns:
        Cartesian coordinates as (3,) array
    """
    frac_arr = np.array(frac, dtype=np.float64)
    return a * (lattice @ frac_arr)


@dataclass
class SublatticePoint:
    """A point in a sublattice with its orbit label."""
    cartesian: np.ndarray  # (3,) Cartesian coordinates
    fractional: Tuple[float, float, float]  # Fractional in primitive basis
    orbit_index: int  # Which orbit this belongs to
    group_element: GroupElement  # Original group element
    
    def __repr__(self):
        return f"SublatticePoint(orbit={self.orbit_index}, frac={self.fractional})"


def build_point_set(
    config: OrbitConfiguration,
    G: QuotientGroup,
    frac_map: Dict[GroupElement, np.ndarray],
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0,
    supercell_range: int = 2
) -> List[SublatticePoint]:
    """
    Build a set of points from an orbit configuration.
    
    Creates points for each group element in each orbit, replicated
    over a supercell range for periodic boundary handling.
    
    Args:
        config: Orbit configuration with disjoint orbits
        G: Quotient group (for reference)
        frac_map: Dict mapping GroupElement → fractional offset (from build_fractional_rep_map)
        lattice: Primitive lattice vectors (3×3, columns)
        a: Lattice parameter
        supercell_range: Replicate over [-range, range]^3
    
    Returns:
        List of SublatticePoint objects
    """
    points = []
    
    for orbit_idx, orbit in enumerate(config.orbits):
        for g in orbit:
            # Get fractional coordinates from the correct coset rep map
            frac = frac_map[g]
            
            # Replicate over supercell
            for di in range(-supercell_range, supercell_range + 1):
                for dj in range(-supercell_range, supercell_range + 1):
                    for dk in range(-supercell_range, supercell_range + 1):
                        shifted_frac = (frac[0] + di, frac[1] + dj, frac[2] + dk)
                        cart = fractional_to_cartesian(shifted_frac, lattice, a)
                        
                        points.append(SublatticePoint(
                            cartesian=cart,
                            fractional=shifted_frac,
                            orbit_index=orbit_idx,
                            group_element=g
                        ))
    
    return points


@dataclass
class RadialShell:
    """A radial shell at a specific distance."""
    distance: float  # Shell radius
    count: int  # Total number of neighbors in this shell
    by_orbit: Dict[int, int]  # Count per orbit index
    
    def __repr__(self):
        orbit_str = ", ".join(f"O{k}:{v}" for k, v in sorted(self.by_orbit.items()))
        return f"Shell(r={self.distance:.4f}, n={self.count}, {orbit_str})"


def compute_radial_shells(
    center: SublatticePoint,
    points: List[SublatticePoint],
    r_max: float = 2.0,
    eps: float = 0.01
) -> List[RadialShell]:
    """
    Compute radial shells around a center point.
    
    Args:
        center: The center point
        points: All points in the structure
        r_max: Maximum radius to consider
        eps: Tolerance for binning distances into shells
    
    Returns:
        List of RadialShell objects, sorted by distance
    """
    # Compute distances and orbit labels
    distances_and_orbits = []
    
    for p in points:
        if np.allclose(p.cartesian, center.cartesian):
            continue  # Skip self
        
        d = np.linalg.norm(p.cartesian - center.cartesian)
        if d <= r_max:
            distances_and_orbits.append((d, p.orbit_index))
    
    if not distances_and_orbits:
        return []
    
    # Sort by distance
    distances_and_orbits.sort(key=lambda x: x[0])
    
    # Bin into shells
    shells = []
    current_shell_distances = [distances_and_orbits[0][0]]
    current_shell_orbits = [distances_and_orbits[0][1]]
    
    for d, orbit in distances_and_orbits[1:]:
        if d - current_shell_distances[0] < eps:
            # Same shell
            current_shell_distances.append(d)
            current_shell_orbits.append(orbit)
        else:
            # New shell - finalize current
            avg_distance = np.mean(current_shell_distances)
            orbit_counts = {}
            for o in current_shell_orbits:
                orbit_counts[o] = orbit_counts.get(o, 0) + 1
            
            shells.append(RadialShell(
                distance=avg_distance,
                count=len(current_shell_orbits),
                by_orbit=orbit_counts
            ))
            
            # Start new shell
            current_shell_distances = [d]
            current_shell_orbits = [orbit]
    
    # Finalize last shell
    if current_shell_distances:
        avg_distance = np.mean(current_shell_distances)
        orbit_counts = {}
        for o in current_shell_orbits:
            orbit_counts[o] = orbit_counts.get(o, 0) + 1
        
        shells.append(RadialShell(
            distance=avg_distance,
            count=len(current_shell_orbits),
            by_orbit=orbit_counts
        ))
    
    return shells


@dataclass 
class CoordinationSignature:
    """
    Coordination signature for a site.
    
    CN1: First shell coordination number
    CN2: Second shell coordination number
    shells: Full shell structure
    """
    orbit_index: int
    cn1: int
    cn2: int
    cn1_by_orbit: Dict[int, int]
    cn2_by_orbit: Dict[int, int]
    shells: List[RadialShell]
    
    def __repr__(self):
        return f"Signature(orbit={self.orbit_index}, CN1={self.cn1}, CN2={self.cn2})"
    
    def to_tuple(self) -> Tuple:
        """Convert to hashable tuple for comparison."""
        cn1_tuple = tuple(sorted(self.cn1_by_orbit.items()))
        cn2_tuple = tuple(sorted(self.cn2_by_orbit.items()))
        return (self.cn1, self.cn2, cn1_tuple, cn2_tuple)


def compute_coordination_signature(
    center: SublatticePoint,
    points: List[SublatticePoint],
    r_max: float = 2.0,
    eps: float = 0.01
) -> CoordinationSignature:
    """
    Compute coordination signature for a center point.
    
    Args:
        center: The center point
        points: All points in the structure
        r_max: Maximum radius for shell computation
        eps: Tolerance for shell binning
    
    Returns:
        CoordinationSignature with CN1, CN2, and full shell data
    """
    shells = compute_radial_shells(center, points, r_max, eps)
    
    if len(shells) >= 1:
        cn1 = shells[0].count
        cn1_by_orbit = shells[0].by_orbit
    else:
        cn1 = 0
        cn1_by_orbit = {}
    
    if len(shells) >= 2:
        cn2 = shells[1].count
        cn2_by_orbit = shells[1].by_orbit
    else:
        cn2 = 0
        cn2_by_orbit = {}
    
    return CoordinationSignature(
        orbit_index=center.orbit_index,
        cn1=cn1,
        cn2=cn2,
        cn1_by_orbit=cn1_by_orbit,
        cn2_by_orbit=cn2_by_orbit,
        shells=shells
    )


def auto_supercell_range(
    config: OrbitConfiguration,
    G: QuotientGroup,
    H: np.ndarray,
    lattice: np.ndarray,
    a: float = 1.0,
    r_max: float = 2.0,
    eps: float = 0.01,
    max_range: int = 5
) -> int:
    """
    Automatically determine sufficient supercell_range for converged CN.
    
    Increases supercell_range until CN signatures for ALL orbits stop changing.
    Uses full CoordinationSignature.to_tuple() for comparison (including cn*_by_orbit).
    
    Args:
        config: Orbit configuration
        G: Quotient group
        H: HNF matrix
        lattice: Primitive lattice
        a: Lattice parameter
        r_max: Max radius for shells
        eps: Shell binning tolerance
        max_range: Maximum range to try
    
    Returns:
        Converged supercell_range value
    """
    frac_map = build_fractional_rep_map(H, G)
    
    prev_all_sigs = None
    
    for supercell_range in range(1, max_range + 1):
        points = build_point_set(config, G, frac_map, lattice, a, supercell_range)
        
        # Get signatures for one representative per orbit
        all_sigs = []
        for orbit_idx, orbit in enumerate(config.orbits):
            # Find a point from this orbit in the central cell
            for p in points:
                if p.orbit_index == orbit_idx:
                    frac = p.fractional
                    if all(0 <= f < 1 for f in frac):
                        sig = compute_coordination_signature(p, points, r_max, eps)
                        all_sigs.append(sig.to_tuple())
                        break
        
        all_sigs_tuple = tuple(all_sigs)
        
        if prev_all_sigs is not None and all_sigs_tuple == prev_all_sigs:
            return supercell_range
        
        prev_all_sigs = all_sigs_tuple
    
    return max_range  # Didn't converge, use max


def analyze_configuration_with_convergence(
    config: OrbitConfiguration,
    G: QuotientGroup,
    H: np.ndarray,
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0,
    r_max: float = 2.0,
    eps: float = 0.01,
    max_range: int = 5
) -> Tuple[Dict[int, List[CoordinationSignature]], int]:
    """
    Analyze configuration with automatic supercell convergence.
    
    Returns:
        Tuple of (signatures_by_orbit, converged_supercell_range)
    """
    # Find converged range
    converged_range = auto_supercell_range(config, G, H, lattice, a, r_max, eps, max_range)
    
    # Build final analysis with converged range
    frac_map = build_fractional_rep_map(H, G)
    points = build_point_set(config, G, frac_map, lattice, a, converged_range)
    
    signatures_by_orbit: Dict[int, List[CoordinationSignature]] = {}
    
    for orbit_idx, orbit in enumerate(config.orbits):
        signatures_by_orbit[orbit_idx] = []
        
        for p in points:
            if p.orbit_index == orbit_idx:
                frac = p.fractional
                if all(0 <= f < 1 for f in frac):
                    sig = compute_coordination_signature(p, points, r_max, eps)
                    signatures_by_orbit[orbit_idx].append(sig)
                    break
    
    return signatures_by_orbit, converged_range


def analyze_configuration(
    config: OrbitConfiguration,
    G: QuotientGroup,
    H: np.ndarray,
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0,
    r_max: float = 2.0,
    eps: float = 0.01,
    supercell_range: int = 3
) -> Dict[int, List[CoordinationSignature]]:
    """
    Analyze coordination environments for all sites in a configuration.
    
    Args:
        config: Orbit configuration
        G: Quotient group
        H: HNF matrix (needed for correct fractional mapping)
        lattice: Primitive lattice (3×3)
        a: Lattice parameter
        r_max: Max radius for shells
        eps: Shell binning tolerance
        supercell_range: Supercell replication range
    
    Returns:
        Dict mapping orbit index to list of signatures for sites in that orbit
    """
    # Build correct fractional rep map from HNF
    frac_map = build_fractional_rep_map(H, G)
    
    # Build point set using correct map
    points = build_point_set(config, G, frac_map, lattice, a, supercell_range)
    
    # Get signatures for each orbit
    signatures_by_orbit: Dict[int, List[CoordinationSignature]] = {}
    
    # We only need to analyze one representative per orbit (in the central cell)
    # since all sites in an orbit are equivalent by translation
    for orbit_idx, orbit in enumerate(config.orbits):
        signatures_by_orbit[orbit_idx] = []
        
        # Find a point from this orbit in the central cell
        for p in points:
            if p.orbit_index == orbit_idx:
                frac = p.fractional
                # Check if in central cell [0, 1)^3
                if all(0 <= f < 1 for f in frac):
                    sig = compute_coordination_signature(p, points, r_max, eps)
                    signatures_by_orbit[orbit_idx].append(sig)
                    break  # One representative is enough
    
    return signatures_by_orbit


def configuration_cn_signature(
    config: OrbitConfiguration,
    G: QuotientGroup,
    H: np.ndarray,
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0
) -> Tuple[Tuple[int, int], ...]:
    """
    Get a simple (CN1, CN2) signature tuple for each orbit.
    
    Returns tuple of (CN1, CN2) pairs, one per orbit, sorted by orbit index.
    Useful for quick comparison/filtering of configurations.
    """
    sigs = analyze_configuration(config, G, H, lattice, a, r_max=1.5, eps=0.02)
    
    result = []
    for orbit_idx in sorted(sigs.keys()):
        if sigs[orbit_idx]:
            s = sigs[orbit_idx][0]
            result.append((s.cn1, s.cn2))
        else:
            result.append((0, 0))
    
    return tuple(result)


def find_configurations_by_cn(
    configs: List[OrbitConfiguration],
    G: QuotientGroup,
    H: np.ndarray,
    target_cn: List[Tuple[int, int]],
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0
) -> List[OrbitConfiguration]:
    """
    Find configurations matching target CN1 values for each orbit.
    
    Args:
        configs: List of configurations to search
        G: Quotient group
        H: HNF matrix
        target_cn: List of (CN1, CN2) tuples, one per orbit
        lattice: Primitive lattice
        a: Lattice parameter
    
    Returns:
        Configurations where orbit i has CN1 matching target_cn[i][0]
    """
    matches = []
    
    for config in configs:
        sig = configuration_cn_signature(config, G, H, lattice, a)
        
        # Check CN1 matches (CN2 can vary more)
        cn1_match = all(s[0] == t[0] for s, t in zip(sig, target_cn))
        if cn1_match:
            matches.append(config)
    
    return matches


def summarize_cn_distribution(
    configs: List[OrbitConfiguration],
    G: QuotientGroup,
    H: np.ndarray,
    lattice: np.ndarray = FCC_PRIMITIVE,
    a: float = 1.0
) -> Dict[Tuple, int]:
    """
    Summarize the distribution of CN signatures across configurations.
    
    Returns dict mapping CN signature → count.
    """
    distribution: Dict[Tuple, int] = {}
    
    for config in configs:
        sig = configuration_cn_signature(config, G, H, lattice, a)
        distribution[sig] = distribution.get(sig, 0) + 1
    
    return distribution


def validate_spinel_geometry():
    """
    Validate Phase 2 geometry against known spinel structure.
    
    Tests both diagonal and non-diagonal HNFs to verify the
    fractional rep map is working correctly.
    """
    print("=" * 70)
    print("PHASE 2: GEOMETRY VALIDATION (GENERAL)")
    print("=" * 70)
    
    # Test 1: Diagonal HNF
    print("\n--- Test 1: Diagonal HNF diag(2,2,2) ---")
    H_diag = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    G_diag = QuotientGroup(H_diag)
    
    print(f"Quotient group: {G_diag}")
    
    # Build fractional rep map
    frac_map = build_fractional_rep_map(H_diag, G_diag)
    print(f"\nFractional rep map (H^{{-1}} k method):")
    for g in sorted(frac_map.keys(), key=lambda x: x.coords):
        t = frac_map[g]
        print(f"  {g.coords} → ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
    
    # Find (2, 4) configurations
    configs = search_subset_configurations_list(G_diag, (2, 4), max_results=20)
    configs = deduplicate_configurations(configs)
    print(f"\nFound {len(configs)} unique (2,4) configurations")
    
    # CN distribution
    cn_dist = summarize_cn_distribution(configs, G_diag, H_diag)
    print(f"\nCN Distribution:")
    for sig, count in sorted(cn_dist.items(), key=lambda x: -x[1]):
        print(f"  O0 CN1={sig[0][0]}, O1 CN1={sig[1][0]}: {count} config(s)")
    
    # Test 2: Non-diagonal HNF
    print("\n--- Test 2: Non-diagonal HNF ---")
    H_nondiag = np.array([[2, 1, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    G_nondiag = QuotientGroup(H_nondiag)
    
    print(f"H = [[2,1,0],[0,2,0],[0,0,2]]")
    print(f"Quotient group: {G_nondiag}")
    
    # Build fractional rep map for non-diagonal
    frac_map_nd = build_fractional_rep_map(H_nondiag, G_nondiag)
    print(f"\nFractional rep map (H^{{-1}} k method):")
    for g in sorted(frac_map_nd.keys(), key=lambda x: x.coords):
        t = frac_map_nd[g]
        print(f"  {g.coords} → ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
    
    # Find configurations for non-diagonal
    configs_nd = search_subset_configurations_list(G_nondiag, (2, 4), max_results=20)
    configs_nd = deduplicate_configurations(configs_nd)
    print(f"\nFound {len(configs_nd)} unique (2,4) configurations")
    
    if configs_nd:
        cn_dist_nd = summarize_cn_distribution(configs_nd, G_nondiag, H_nondiag)
        print(f"\nCN Distribution:")
        for sig, count in sorted(cn_dist_nd.items(), key=lambda x: -x[1]):
            print(f"  O0 CN1={sig[0][0]}, O1 CN1={sig[1][0]}: {count} config(s)")
    
    # Test 3: Verify consistency - different Bravais types
    print("\n--- Test 3: Different Bravais Lattice (BCC) ---")
    
    H_bcc = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    G_bcc = QuotientGroup(H_bcc)
    
    configs_bcc = search_subset_configurations_list(G_bcc, (2, 4), max_results=10)
    configs_bcc = deduplicate_configurations(configs_bcc)
    
    if configs_bcc:
        # Compare FCC vs BCC metrics for same abstract configuration
        config = configs_bcc[0]
        
        frac_map_bcc = build_fractional_rep_map(H_bcc, G_bcc)
        
        sig_fcc = configuration_cn_signature(config, G_bcc, H_bcc, FCC_PRIMITIVE, a=1.0)
        sig_bcc = configuration_cn_signature(config, G_bcc, H_bcc, BCC_PRIMITIVE, a=1.0)
        sig_sc = configuration_cn_signature(config, G_bcc, H_bcc, SC_PRIMITIVE, a=1.0)
        
        print(f"\nSame abstract config, different lattices:")
        print(f"  FCC: O0 CN={sig_fcc[0]}, O1 CN={sig_fcc[1]}")
        print(f"  BCC: O0 CN={sig_bcc[0]}, O1 CN={sig_bcc[1]}")
        print(f"  SC:  O0 CN={sig_sc[0]}, O1 CN={sig_sc[1]}")
    
    # Reference distances
    print("\n" + "-" * 70)
    print("REFERENCE: LATTICE NEAREST NEIGHBORS")
    print("-" * 70)
    
    fcc_nn = np.linalg.norm(fractional_to_cartesian((1, 0, 0), FCC_PRIMITIVE, a=1.0))
    bcc_nn = np.linalg.norm(fractional_to_cartesian((1, 0, 0), BCC_PRIMITIVE, a=1.0))
    sc_nn = np.linalg.norm(fractional_to_cartesian((1, 0, 0), SC_PRIMITIVE, a=1.0))
    
    print(f"  FCC primitive nearest neighbor: {fcc_nn:.4f}a")
    print(f"  BCC primitive nearest neighbor: {bcc_nn:.4f}a")
    print(f"  SC primitive nearest neighbor:  {sc_nn:.4f}a")
    
    return configs


# =============================================================================
# SPINEL VALIDATION
# =============================================================================

def validate_spinel_sublattice():
    """
    Validate the algorithm against known spinel cation sublattice.
    
    Spinel AB₂O₄ has 6 interpenetrating FCC lattices:
    - 2 for tetrahedral (A) sites  
    - 4 for octahedral (B) sites
    
    We want orbit_sizes = (2, 4), N = 6 total offsets.
    
    With M=8 (e.g., H=diag(2,2,2) giving G = Z_2^3):
    - Size-2 and size-4 subgroups exist (2 and 4 both divide 8)
    - We select one coset of size 2 and one of size 4
    - 6 elements used, 2 elements left unused
    
    IMPORTANT: Only use diagonal HNFs until SNF is implemented,
    since group arithmetic is incorrect for non-diagonal cases.
    """
    print("=" * 70)
    print("SPINEL CATION SUBLATTICE VALIDATION")
    print("=" * 70)
    
    # The correct test: M=8, orbit_sizes=(2,4), partial partition
    M = 8
    orbit_sizes = (2, 4)
    N = sum(orbit_sizes)
    
    print(f"\nSearching for {orbit_sizes} configurations (N={N} of M={M})")
    print("Using DIAGONAL HNFs only (group arithmetic correct)")
    print("-" * 70)
    
    # Only diagonal HNFs for now
    hnfs = enumerate_diagonal_hnf(M)
    print(f"Found {len(hnfs)} diagonal HNF matrices with det = {M}")
    
    total_configs = 0
    
    for H in hnfs:
        G = QuotientGroup(H)
        
        # Use the subset search (doesn't require full partition)
        configs = search_subset_configurations_list(G, orbit_sizes, max_results=50)
        configs = deduplicate_configurations(configs)
        
        if configs:
            diag_str = f"diag({H[0,0]},{H[1,1]},{H[2,2]})"
            print(f"\n{diag_str}: {len(configs)} unique configuration(s)")
            total_configs += len(configs)
            
            # Show first configuration
            cfg = configs[0]
            used = cfg.orbits[0] | cfg.orbits[1]
            unused = set(G.elements) - used
            
            print(f"  Example configuration:")
            print(f"    Orbit 1 (tet, size {len(cfg.orbits[0])}): {sorted([g.coords for g in cfg.orbits[0]])}")
            print(f"    Orbit 2 (oct, size {len(cfg.orbits[1])}): {sorted([g.coords for g in cfg.orbits[1]])}")
            print(f"    Unused ({len(unused)}): {sorted([g.coords for g in unused])}")
    
    print(f"\n{'='*70}")
    print(f"Total unique (2,4) configurations in M=8 diagonal HNFs: {total_configs}")
    print("="*70)
    
    # Also test the full partition case for comparison
    print("\n\nCOMPARISON: Full partition (2,2,4) covering all 8 elements")
    print("-" * 70)
    
    for H in hnfs[:3]:  # Just first 3 for comparison
        G = QuotientGroup(H)
        configs = search_configurations_list(G, (2, 2, 4), max_results=20)
        configs = deduplicate_configurations(configs)
        if configs:
            diag_str = f"diag({H[0,0]},{H[1,1]},{H[2,2]})"
            print(f"{diag_str}: {len(configs)} config(s) for (2,2,4)")
    
    return total_configs


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    print("=" * 70)
    print("LATTICE SEARCH TOOL - Phase 1 & 2")
    print("=" * 70)
    
    # Step 0: Run correctness tests first
    print("\n0. CORRECTNESS TESTS (must pass)")
    print("-" * 70)
    tests_passed = run_all_correctness_tests(M_values=[4, 6, 8], verbose=True)
    if not tests_passed:
        print("\n*** CORRECTNESS TESTS FAILED - DO NOT TRUST SEARCH RESULTS ***")
        return
    print("\nAll correctness tests passed ✓")
    
    # Demo 1: Basic HNF enumeration
    print("\n1. HNF ENUMERATION")
    print("-" * 70)
    for M in [2, 4, 6, 8]:
        all_hnfs = enumerate_hnf(M)
        diag_hnfs = enumerate_diagonal_hnf(M)
        print(f"M = {M}: {len(all_hnfs)} total HNFs, {len(diag_hnfs)} diagonal")
    
    # Demo 2: SNF verification
    print("\n2. SMITH NORMAL FORM VERIFICATION")
    print("-" * 70)
    
    # Test diagonal HNF
    H_diag = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    D, U, V = smith_normal_form(H_diag)
    print(f"Diagonal H = diag(2,2,2)")
    print(f"  SNF D = diag({D[0,0]},{D[1,1]},{D[2,2]})")
    print(f"  U @ H @ V = D: {np.allclose(U @ H_diag @ V, D)}")
    
    # Test integer inverse
    U_inv = integer_matrix_inverse(U)
    print(f"  Integer U_inv @ U = I: {np.allclose(U_inv @ U, np.eye(3))}")
    
    # Test non-diagonal HNF  
    H_nondiag = np.array([[2, 1, 0], [0, 2, 1], [0, 0, 2]], dtype=np.int64)
    D, U, V = smith_normal_form(H_nondiag)
    det_H = int(H_nondiag[0,0]) * int(H_nondiag[1,1]) * int(H_nondiag[2,2])  # Upper triangular
    print(f"\nNon-diagonal H = [[2,1,0],[0,2,1],[0,0,2]]")
    print(f"  det(H) = {det_H}")
    print(f"  SNF D = diag({D[0,0]},{D[1,1]},{D[2,2]})")
    print(f"  Invariant factors: {D[0,0]} | {D[1,1]} | {D[2,2]}")
    print(f"  U @ H @ V = D: {np.allclose(U @ H_nondiag @ V, D)}")
    
    # Verify integer inverse for non-diagonal case
    U_inv = integer_matrix_inverse(U)
    print(f"  Integer U_inv @ U = I: {np.allclose(U_inv @ U, np.eye(3))}")
    
    # Verify group structure for non-diagonal
    G = QuotientGroup(H_nondiag)
    print(f"  QuotientGroup: {G}")
    print(f"  |G| = {len(G)}, factors = {G.invariant_factors}")
    
    # Verify group axioms
    e = G.identity()
    g = G.elements[3]
    h = G.elements[5]
    print(f"  Identity check: g + e = g: {G.add(g, e) == g}")
    print(f"  Inverse check: g + (-g) = e: {G.add(g, G.neg(g)) == e}")
    print(f"  Associativity: (g+h)+k = g+(h+k): {G.add(G.add(g, h), G.elements[2]) == G.add(g, G.add(h, G.elements[2]))}")
    
    # Demo 3: Quotient group structure (diagonal HNF)
    print("\n3. QUOTIENT GROUP EXAMPLE (diagonal HNF)")
    print("-" * 70)
    H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    G = QuotientGroup(H)
    print(f"H = diag(2,2,2) → G = Z_2 × Z_2 × Z_2")
    print(f"|G| = {len(G)}")
    print(f"Elements: {[g.coords for g in G.elements]}")
    
    # Demo 4: Subgroup enumeration
    print("\n4. SUBGROUP ENUMERATION for Z_2^3")
    print("-" * 70)
    subgroups = enumerate_subgroups_cached(G)
    for size, subs in sorted(subgroups.items()):
        print(f"Size {size}: {len(subs)} subgroup(s)")
        if size == 2:
            print(f"  Subgroups: {[[g.coords for g in s] for s in subs[:3]]}...")
    
    # Demo 5: THE KEY TEST - Spinel (2,4) partial partition
    print("\n5. SPINEL VALIDATION: orbit_sizes=(2,4) from M=8")
    print("-" * 70)
    validate_spinel_sublattice()
    
    # Demo 6: Non-diagonal HNF search (now safe with SNF!)
    print("\n6. NON-DIAGONAL HNF SEARCH (using SNF)")
    print("-" * 70)
    
    # Try M=8 with non-diagonal HNFs
    all_hnfs = enumerate_hnf(8)
    nondiag_hnfs = [H for H in all_hnfs if not is_diagonal_hnf(H)]
    print(f"Testing {len(nondiag_hnfs)} non-diagonal HNFs with M=8")
    
    total_configs = 0
    examples_shown = 0
    
    for H in nondiag_hnfs[:20]:  # First 20 for demo
        G = QuotientGroup(H)
        configs = search_subset_configurations_list(G, (2, 4), max_results=10)
        configs = deduplicate_configurations(configs)
        
        if configs and examples_shown < 3:
            print(f"\n{hnf_to_string(H)} → factors {G.invariant_factors}: {len(configs)} config(s)")
            cfg = configs[0]
            print(f"  Orbit 1: {sorted([g.coords for g in cfg.orbits[0]])}")
            print(f"  Orbit 2: {sorted([g.coords for g in cfg.orbits[1]])}")
            examples_shown += 1
        
        total_configs += len(configs)
    
    print(f"\nTotal (2,4) configs in first 20 non-diagonal HNFs: {total_configs}")
    
    # Demo 7: Full vs partial comparison
    print("\n7. FULL vs PARTIAL PARTITION COMPARISON")
    print("-" * 70)
    H = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.int64)
    G = QuotientGroup(H)
    
    partial = search_subset_configurations_list(G, (2, 4), max_results=100)
    partial = deduplicate_configurations(partial)
    print(f"Partial partition (2,4): {len(partial)} unique configs")
    
    full = search_configurations_list(G, (2, 2, 4), max_results=100)
    full = deduplicate_configurations(full)
    print(f"Full partition (2,2,4):  {len(full)} unique configs")
    
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE - SNF NOW ENABLES ALL HNFs")
    print("=" * 70)
    
    # Demo 8: Convenience function
    print("\n8. CONVENIENCE FUNCTION: search_subset_across_hnfs")
    print("-" * 70)
    
    results = search_subset_across_hnfs(
        M_values=[8],
        orbit_sizes=(2, 4),
        diagonal_only=False,  # Now safe with SNF!
        max_per_hnf=10
    )
    
    total = sum(len(configs) for configs in results.values())
    print(f"M=8, orbit_sizes=(2,4), all HNFs:")
    print(f"  {len(results)} HNFs with valid configurations")
    print(f"  {total} total unique configurations")
    
    # Phase 2: Geometry validation
    print("\n")
    validate_spinel_geometry()
    
    # Phase 3: Search + Filter + Rank + Export
    print("\n")
    demo_phase3()
    
    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE")
    print("=" * 70)
    print("\nPhase 1: Group theory (HNF, SNF, coset enumeration)")
    print("Phase 2: Geometry (fractional map, CN signatures, shells)")
    print("Phase 3: Search pipeline (constraints, scoring, export)")
    print("\nMain entry point: search_and_analyze(lattice, a, orbit_sizes, M_list)")


# =============================================================================
# PHASE 3A: GENERAL SEARCH RUNNER
# =============================================================================

@dataclass
class AnalyzedConfiguration:
    """
    A configuration with full geometric analysis.
    
    Contains the abstract group-theoretic configuration plus
    all computed geometric properties.
    
    Important: The same abstract config can give different CN/distances
    when applied to different parent lattices. The parent_basis_name
    field tracks which lattice was used for the geometric analysis.
    
    Uniformity: By default, we check that all sites within each orbit
    have identical coordination signatures. If not uniform, the config
    is flagged (is_uniform=False) and may be filtered out.
    """
    config: OrbitConfiguration
    H: np.ndarray
    G: QuotientGroup
    frac_map: Dict[GroupElement, np.ndarray]
    signatures: Dict[int, CoordinationSignature]
    lattice: np.ndarray
    a: float
    converged_range: int
    
    # Parent lattice identification (critical for interpretation)
    parent_basis_name: str = "unknown"  # e.g., "FCC-primitive", "BCC-primitive", "triclinic"
    cell_params: Optional[Tuple[float, float, float, float, float, float]] = None  # (a,b,c,α,β,γ)
    
    # Uniformity tracking
    is_uniform: bool = True  # All sites in each orbit have identical signatures
    uniformity_details: Optional[Dict[int, bool]] = None  # Per-orbit uniformity
    
    # Derived properties (computed lazily)
    _min_distance: Optional[float] = None
    _shell_gap: Optional[float] = None
    
    @property
    def orbit_sizes(self) -> Tuple[int, ...]:
        return self.config.orbit_sizes
    
    @property
    def min_distance(self) -> float:
        """Minimum distance between any two sites."""
        if self._min_distance is None:
            self._compute_distance_stats()
        return self._min_distance
    
    @property
    def shell_gap(self) -> float:
        """Gap between first and second shell (d2 - d1)."""
        if self._shell_gap is None:
            self._compute_distance_stats()
        return self._shell_gap
    
    def _compute_distance_stats(self):
        """Compute min distance and shell gap from signatures."""
        all_d1 = []
        all_d2 = []
        
        for sig in self.signatures.values():
            if sig.shells:
                all_d1.append(sig.shells[0].distance)
                if len(sig.shells) >= 2:
                    all_d2.append(sig.shells[1].distance)
        
        self._min_distance = min(all_d1) if all_d1 else 0.0
        
        if all_d1 and all_d2:
            self._shell_gap = min(all_d2) - max(all_d1)
        else:
            self._shell_gap = 0.0
    
    def get_fractional_coords(self, orbit_idx: int) -> List[np.ndarray]:
        """Get fractional coordinates for all sites in an orbit."""
        return [self.frac_map[g] for g in self.config.orbits[orbit_idx]]
    
    def get_all_fractional_coords(self) -> List[List[np.ndarray]]:
        """Get fractional coordinates grouped by orbit."""
        return [self.get_fractional_coords(i) for i in range(len(self.config.orbits))]
    
    def __repr__(self):
        sizes = tuple(len(o) for o in self.config.orbits)
        cn1s = [self.signatures[i].cn1 for i in range(len(self.config.orbits))]
        uniform_str = "" if self.is_uniform else ", NON-UNIFORM"
        return f"AnalyzedConfig(basis={self.parent_basis_name}, sizes={sizes}, CN1={cn1s}, d_min={self.min_distance:.3f}{uniform_str})"


def signature_key(sig: CoordinationSignature) -> Tuple:
    """
    Generate a hashable key from a CoordinationSignature for comparison.
    
    Two sites with the same key have identical coordination environments.
    """
    return (
        sig.cn1,
        sig.cn2,
        tuple(sorted(sig.cn1_by_orbit.items())),
        tuple(sorted(sig.cn2_by_orbit.items())),
        # Include first shell distance (rounded to avoid float issues)
        round(sig.shells[0].distance, 4) if sig.shells else 0.0
    )


def compute_orbit_signatures_full(
    orbit_idx: int,
    config: OrbitConfiguration,
    G: QuotientGroup,
    frac_map: Dict[GroupElement, np.ndarray],
    lattice: np.ndarray,
    a: float,
    supercell_range: int,
    r_max: float = 2.0,
    eps: float = 0.01
) -> Tuple[List[CoordinationSignature], bool]:
    """
    Compute coordination signatures for ALL sites in an orbit.
    
    Returns:
        (signatures_list, is_uniform)
        - signatures_list: List of signatures, one per site in the orbit
        - is_uniform: True if all sites have identical signatures
    """
    points = build_point_set(config, G, frac_map, lattice, a, supercell_range)
    
    # Find all sites in this orbit within the primitive cell
    orbit_sites = []
    for p in points:
        if p.orbit_index == orbit_idx:
            frac = p.fractional
            if all(0 <= f < 1 - 1e-9 for f in frac):  # In primitive cell
                orbit_sites.append(p)
    
    if not orbit_sites:
        return [], True
    
    # Compute signature for each site
    signatures = []
    for site in orbit_sites:
        sig = compute_coordination_signature(site, points, r_max, eps)
        signatures.append(sig)
    
    # Check uniformity
    if len(signatures) <= 1:
        is_uniform = True
    else:
        first_key = signature_key(signatures[0])
        is_uniform = all(signature_key(s) == first_key for s in signatures[1:])
    
    return signatures, is_uniform


def analyze_single_configuration(
    config: OrbitConfiguration,
    H: np.ndarray,
    G: QuotientGroup,
    lattice: np.ndarray,
    a: float = 1.0,
    r_max: float = 2.0,
    eps: float = 0.01,
    parent_basis_name: str = "unknown",
    cell_params: Optional[Tuple[float, float, float, float, float, float]] = None,
    check_uniformity: bool = True
) -> AnalyzedConfiguration:
    """
    Fully analyze a single configuration.
    
    Args:
        config: The orbit configuration to analyze
        H: HNF matrix
        G: Quotient group
        lattice: Bravais lattice matrix
        a: Lattice parameter
        r_max: Max radius for shell computation
        eps: Shell binning tolerance
        parent_basis_name: Name of parent Bravais lattice
        cell_params: Optional cell parameters (a,b,c,α,β,γ)
        check_uniformity: If True, verify all sites in each orbit have
                         identical coordination. If False, only sample one site.
    
    Returns:
        AnalyzedConfiguration with all geometric properties computed.
        If check_uniformity=True and orbits are non-uniform, is_uniform=False.
    """
    frac_map = build_fractional_rep_map(H, G)
    
    # Find converged supercell range
    converged_range = auto_supercell_range(
        config, G, H, lattice, a, r_max, eps, max_range=5
    )
    
    # Build points for analysis
    points = build_point_set(config, G, frac_map, lattice, a, converged_range)
    
    signatures = {}
    is_uniform = True
    uniformity_details = {}
    
    for orbit_idx in range(len(config.orbits)):
        if check_uniformity:
            # Check ALL sites in this orbit
            orbit_sigs, orbit_uniform = compute_orbit_signatures_full(
                orbit_idx, config, G, frac_map, lattice, a,
                converged_range, r_max, eps
            )
            
            uniformity_details[orbit_idx] = orbit_uniform
            if not orbit_uniform:
                is_uniform = False
            
            # Use first signature as representative
            if orbit_sigs:
                signatures[orbit_idx] = orbit_sigs[0]
        else:
            # Just sample one site (faster but doesn't verify uniformity)
            for p in points:
                if p.orbit_index == orbit_idx:
                    frac = p.fractional
                    if all(0 <= f < 1 for f in frac):
                        sig = compute_coordination_signature(p, points, r_max, eps)
                        signatures[orbit_idx] = sig
                        uniformity_details[orbit_idx] = True  # Assumed
                        break
    
    return AnalyzedConfiguration(
        config=config,
        H=H,
        G=G,
        frac_map=frac_map,
        signatures=signatures,
        lattice=lattice,
        a=a,
        converged_range=converged_range,
        parent_basis_name=parent_basis_name,
        cell_params=cell_params,
        is_uniform=is_uniform,
        uniformity_details=uniformity_details
    )


def search_and_analyze(
    lattice: np.ndarray,
    a: float,
    orbit_sizes: Tuple[int, ...],
    M_list: List[int],
    diagonal_only: bool = False,
    max_per_hnf: int = 20,
    r_max: float = 2.0,
    eps: float = 0.01,
    verbose: bool = True,
    parent_basis_name: str = "unknown",
    cell_params: Optional[Tuple[float, float, float, float, float, float]] = None
) -> List[AnalyzedConfiguration]:
    """
    Main Phase 3 entry point: search and analyze configurations.
    
    Pipeline:
    1. Generate candidate configs for each HNF in M_list
    2. Embed to real space, compute per-orbit signatures
    3. Return all analyzed configurations
    
    Args:
        lattice: Bravais lattice matrix (3×3, columns are vectors)
        a: Lattice parameter
        orbit_sizes: Target orbit sizes, e.g., (2, 4)
        M_list: List of supercell sizes to search
        diagonal_only: If True, only use diagonal HNFs
        max_per_hnf: Max configurations per HNF
        r_max: Max radius for shell computation
        eps: Shell binning tolerance
        verbose: Print progress
        parent_basis_name: Name of the parent Bravais lattice (e.g., "FCC-primitive")
        cell_params: Optional (a,b,c,α,β,γ) if lattice came from parameters
    
    Returns:
        List of AnalyzedConfiguration objects
    """
    N = sum(orbit_sizes)
    results = []
    
    for M in M_list:
        if M < N:
            continue
        
        # Check Lagrange compatibility
        if not all(M % size == 0 for size in orbit_sizes):
            continue
        
        if diagonal_only:
            hnfs = enumerate_diagonal_hnf(M)
        else:
            hnfs = enumerate_hnf(M)
        
        if verbose:
            print(f"M={M}: searching {len(hnfs)} HNFs...")
        
        for H in hnfs:
            G = QuotientGroup(H)
            frac_map = build_fractional_rep_map(H, G)
            
            configs = search_subset_configurations_list(
                G, orbit_sizes, max_results=max_per_hnf
            )
            # Use stronger deduplication with translation canonicalization
            configs = deduplicate_with_canonicalization(configs, G, frac_map)
            
            for config in configs:
                analyzed = analyze_single_configuration(
                    config, H, G, lattice, a, r_max, eps,
                    parent_basis_name=parent_basis_name,
                    cell_params=cell_params
                )
                results.append(analyzed)
    
    # Global deduplication across HNFs based on CN signature
    # (same geometry can arise from different HNF matrices)
    results = deduplicate_analyzed_configurations(results)
    
    if verbose:
        print(f"Total: {len(results)} analyzed configurations")
    
    return results


# =============================================================================
# PHASE 3B: CONSTRAINT LANGUAGE
# =============================================================================

@dataclass
class Constraint:
    """Base class for constraints."""
    pass


@dataclass
class UniformityConstraint(Constraint):
    """
    Require that the configuration has uniform orbits.
    
    A uniform orbit is one where ALL sites have identical coordination
    signatures (same CN, same neighbor breakdown by orbit, same distances).
    
    This filters out "broken" configurations where subset selection
    accidentally created non-equivalent sites within an orbit.
    """
    pass


@dataclass
class CN1TotalConstraint(Constraint):
    """
    Require total CN1 == target for a specific orbit.
    
    WARNING: This checks TOTAL first-shell neighbors regardless of source.
    If you want "orbit A has exactly 4 neighbors from orbit B",
    use CN1ByOrbitConstraint instead.
    
    Example: CN1TotalConstraint(orbit_idx=0, target=4)
             Orbit 0 must have exactly 4 first-shell neighbors (from any orbit)
    """
    orbit_idx: int
    target: int


@dataclass
class CN1ByOrbitConstraint(Constraint):
    """
    Require a specific number of CN1 neighbors from a specific source orbit.
    
    This is the constraint you usually want for chemistry queries like
    "tetrahedral sites coordinated by 4 oxygen atoms".
    
    Example: CN1ByOrbitConstraint(orbit_idx=0, source_orbit=2, target=4)
             Every site in orbit 0 must have exactly 4 neighbors from orbit 2
             
    Example: For spinel tetrahedral sites with 4 O neighbors:
             CN1ByOrbitConstraint(orbit_idx=0, source_orbit=2, target=4)
             where orbit 2 is the anion sublattice
    """
    orbit_idx: int  # The orbit whose CN we're checking
    source_orbit: int  # The orbit contributing neighbors
    target: int  # Required count from that source


@dataclass
class FirstShellDistanceConstraint(Constraint):
    """Require first shell distance in a range (scaled by a)."""
    orbit_idx: int
    min_d: float
    max_d: float


@dataclass
class MinSeparationConstraint(Constraint):
    """Require minimum separation between any two sites."""
    min_distance: float  # In units of a


@dataclass
class ShellGapConstraint(Constraint):
    """Require minimum gap between first and second shells."""
    min_gap: float


@dataclass
class MajorityNeighborConstraint(Constraint):
    """Require that majority of orbit A's CN1 neighbors are from orbit B."""
    orbit_idx: int  # Orbit A
    neighbor_orbit: int  # Orbit B
    min_fraction: float = 0.5  # At least this fraction


@dataclass
class CN1RangeConstraint(Constraint):
    """Require CN1 in a range for a specific orbit."""
    orbit_idx: int
    min_cn: int
    max_cn: int


@dataclass
class DistinctCNConstraint(Constraint):
    """Require that different orbits have distinct CN1 values."""
    pass


@dataclass
class FirstShellCompositionConstraint(Constraint):
    """
    Require exact first-shell composition for an orbit.
    
    Specifies exactly how many neighbors should come from each source orbit.
    
    Example: For orbit 0 to have CN1 = 4 all from orbit 1:
        FirstShellCompositionConstraint(orbit_idx=0, composition={1: 4})
    
    Example: For spinel tetrahedral site with 4 neighbors from oxygen:
        FirstShellCompositionConstraint(orbit_idx=0, composition={2: 4})  # orbit 2 = O
    """
    orbit_idx: int
    composition: Dict[int, int]  # source_orbit -> required_count


@dataclass
class CN2TotalConstraint(Constraint):
    """Require CN2 == target for a specific orbit."""
    orbit_idx: int
    target: int


@dataclass
class CN2RangeConstraint(Constraint):
    """Require CN2 in a range for a specific orbit."""
    orbit_idx: int
    min_cn: int
    max_cn: int


@dataclass
class CN2ByOrbitConstraint(Constraint):
    """Require CN2 contribution from a specific source orbit."""
    orbit_idx: int  # The orbit whose CN2 we're checking
    source_orbit: int  # The orbit contributing to CN2
    target: int  # Required count


@dataclass
class SecondShellCompositionConstraint(Constraint):
    """
    Require exact second-shell composition for an orbit.
    
    Like FirstShellCompositionConstraint but for the second shell.
    Useful for distinguishing nets that differ only at shell 2.
    """
    orbit_idx: int
    composition: Dict[int, int]  # source_orbit -> required_count


@dataclass
class ShellRatioConstraint(Constraint):
    """
    Require a specific ratio between CN1 and CN2.
    
    Useful for identifying specific coordination geometries.
    """
    orbit_idx: int
    min_ratio: float  # CN2 / CN1 >= min_ratio
    max_ratio: float  # CN2 / CN1 <= max_ratio


def check_constraint(
    analyzed: AnalyzedConfiguration,
    constraint: Constraint
) -> bool:
    """
    Check if an analyzed configuration satisfies a constraint.
    
    Returns True if constraint is satisfied.
    """
    if isinstance(constraint, UniformityConstraint):
        return analyzed.is_uniform
    
    elif isinstance(constraint, CN1TotalConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        return sig.cn1 == constraint.target
    
    elif isinstance(constraint, CN1ByOrbitConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        count = sig.cn1_by_orbit.get(constraint.source_orbit, 0)
        return count == constraint.target
    
    elif isinstance(constraint, CN1RangeConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        return constraint.min_cn <= sig.cn1 <= constraint.max_cn
    
    elif isinstance(constraint, FirstShellDistanceConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None or not sig.shells:
            return False
        d1 = sig.shells[0].distance
        return constraint.min_d <= d1 <= constraint.max_d
    
    elif isinstance(constraint, MinSeparationConstraint):
        return analyzed.min_distance >= constraint.min_distance
    
    elif isinstance(constraint, ShellGapConstraint):
        return analyzed.shell_gap >= constraint.min_gap
    
    elif isinstance(constraint, MajorityNeighborConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None or sig.cn1 == 0:
            return False
        neighbor_count = sig.cn1_by_orbit.get(constraint.neighbor_orbit, 0)
        fraction = neighbor_count / sig.cn1
        return fraction >= constraint.min_fraction
    
    elif isinstance(constraint, DistinctCNConstraint):
        cn1_values = [sig.cn1 for sig in analyzed.signatures.values()]
        return len(cn1_values) == len(set(cn1_values))
    
    elif isinstance(constraint, FirstShellCompositionConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        # Check each required source orbit count
        for source_orbit, required_count in constraint.composition.items():
            actual_count = sig.cn1_by_orbit.get(source_orbit, 0)
            if actual_count != required_count:
                return False
        # Also verify no unexpected contributions
        for source_orbit, actual_count in sig.cn1_by_orbit.items():
            if source_orbit not in constraint.composition and actual_count > 0:
                return False
        return True
    
    elif isinstance(constraint, CN2TotalConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        return sig.cn2 == constraint.target
    
    elif isinstance(constraint, CN2RangeConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        return constraint.min_cn <= sig.cn2 <= constraint.max_cn
    
    elif isinstance(constraint, CN2ByOrbitConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        count = sig.cn2_by_orbit.get(constraint.source_orbit, 0)
        return count == constraint.target
    
    elif isinstance(constraint, SecondShellCompositionConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None:
            return False
        # Check each required source orbit count
        for source_orbit, required_count in constraint.composition.items():
            actual_count = sig.cn2_by_orbit.get(source_orbit, 0)
            if actual_count != required_count:
                return False
        # Also verify no unexpected contributions
        for source_orbit, actual_count in sig.cn2_by_orbit.items():
            if source_orbit not in constraint.composition and actual_count > 0:
                return False
        return True
    
    elif isinstance(constraint, ShellRatioConstraint):
        sig = analyzed.signatures.get(constraint.orbit_idx)
        if sig is None or sig.cn1 == 0:
            return False
        ratio = sig.cn2 / sig.cn1
        return constraint.min_ratio <= ratio <= constraint.max_ratio
    
    else:
        raise ValueError(f"Unknown constraint type: {type(constraint)}")


def filter_by_constraints(
    configs: List[AnalyzedConfiguration],
    constraints: List[Constraint]
) -> List[AnalyzedConfiguration]:
    """
    Filter configurations by a list of constraints.
    
    Returns configurations that satisfy ALL constraints.
    """
    return [
        c for c in configs
        if all(check_constraint(c, constraint) for constraint in constraints)
    ]


# =============================================================================
# PHASE 3C: RANKING AND SCORING
# =============================================================================

@dataclass
class ScoringWeights:
    """Weights for scoring function."""
    shell_gap_weight: float = 1.0  # Maximize d2 - d1
    min_distance_penalty: float = -2.0  # Penalize small separations
    min_distance_threshold: float = 0.3  # Below this, apply penalty
    cn_target_bonus: float = 0.5  # Bonus for matching target CN pattern
    cn_targets: Optional[Dict[int, int]] = None  # orbit_idx -> target CN1


def score_configuration(
    analyzed: AnalyzedConfiguration,
    weights: ScoringWeights
) -> float:
    """
    Compute a score for ranking configurations.
    
    Higher score = better configuration.
    """
    score = 0.0
    
    # Shell gap bonus (larger gap = cleaner structure)
    score += weights.shell_gap_weight * analyzed.shell_gap
    
    # Minimum distance penalty
    if analyzed.min_distance < weights.min_distance_threshold:
        penalty = weights.min_distance_threshold - analyzed.min_distance
        score += weights.min_distance_penalty * penalty
    
    # CN target bonus
    if weights.cn_targets:
        for orbit_idx, target_cn in weights.cn_targets.items():
            if orbit_idx in analyzed.signatures:
                actual_cn = analyzed.signatures[orbit_idx].cn1
                if actual_cn == target_cn:
                    score += weights.cn_target_bonus
    
    return score


def rank_configurations(
    configs: List[AnalyzedConfiguration],
    weights: Optional[ScoringWeights] = None,
    top_n: Optional[int] = None
) -> List[Tuple[AnalyzedConfiguration, float]]:
    """
    Rank configurations by score.
    
    Args:
        configs: Configurations to rank
        weights: Scoring weights (uses defaults if None)
        top_n: Return only top N (all if None)
    
    Returns:
        List of (config, score) tuples, sorted by score descending
    """
    if weights is None:
        weights = ScoringWeights()
    
    scored = [(c, score_configuration(c, weights)) for c in configs]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    if top_n is not None:
        scored = scored[:top_n]
    
    return scored


# =============================================================================
# PHASE 3D: EXPORT
# =============================================================================

def export_to_json(
    analyzed: AnalyzedConfiguration,
    include_shells: bool = False
) -> dict:
    """
    Export configuration to JSON-serializable dict.
    
    Suitable for web apps and data exchange.
    """
    orbits_data = []
    
    for orbit_idx, orbit in enumerate(analyzed.config.orbits):
        fracs = analyzed.get_fractional_coords(orbit_idx)
        sig = analyzed.signatures.get(orbit_idx)
        
        orbit_data = {
            "orbit_index": orbit_idx,
            "size": len(orbit),
            "fractional_coords": [f.tolist() for f in fracs],
            "cn1": sig.cn1 if sig else None,
            "cn2": sig.cn2 if sig else None,
            "cn1_by_orbit": dict(sig.cn1_by_orbit) if sig else {},
        }
        
        if include_shells and sig:
            orbit_data["shells"] = [
                {
                    "distance": shell.distance,
                    "count": shell.count,
                    "by_orbit": dict(shell.by_orbit)
                }
                for shell in sig.shells[:5]  # First 5 shells
            ]
        
        orbits_data.append(orbit_data)
    
    return {
        "parent_basis": analyzed.parent_basis_name,
        "cell_params": list(analyzed.cell_params) if analyzed.cell_params else None,
        "is_uniform": analyzed.is_uniform,
        "uniformity_details": analyzed.uniformity_details,
        "orbit_sizes": list(analyzed.orbit_sizes),
        "M": len(analyzed.G),
        "hnf_diagonal": [int(analyzed.H[i, i]) for i in range(3)],
        "hnf_off_diagonal": [
            int(analyzed.H[0, 1]),
            int(analyzed.H[0, 2]),
            int(analyzed.H[1, 2])
        ],
        "lattice_parameter": analyzed.a,
        "min_distance": analyzed.min_distance,
        "shell_gap": analyzed.shell_gap,
        "orbits": orbits_data
    }


def export_to_cif(
    analyzed: AnalyzedConfiguration,
    species_per_orbit: Optional[List[str]] = None,
    title: str = "LatticeSearch_Structure"
) -> str:
    """
    Export configuration to CIF format.
    
    Args:
        analyzed: Analyzed configuration
        species_per_orbit: Element symbol per orbit (e.g., ["Fe", "Ni"])
        title: Structure title
    
    Returns:
        CIF file content as string
    """
    if species_per_orbit is None:
        # Default: A, B, C, ... for each orbit
        species_per_orbit = [chr(65 + i) for i in range(len(analyzed.config.orbits))]
    
    # Get lattice parameters from lattice matrix
    # For general triclinic, extract a, b, c, alpha, beta, gamma
    L = analyzed.lattice * analyzed.a
    a_vec = L[:, 0]
    b_vec = L[:, 1]
    c_vec = L[:, 2]
    
    a_len = np.linalg.norm(a_vec)
    b_len = np.linalg.norm(b_vec)
    c_len = np.linalg.norm(c_vec)
    
    alpha = np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b_len * c_len)))
    beta = np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a_len * c_len)))
    gamma = np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a_len * b_len)))
    
    lines = [
        f"data_{title}",
        f"_cell_length_a {a_len:.6f}",
        f"_cell_length_b {b_len:.6f}",
        f"_cell_length_c {c_len:.6f}",
        f"_cell_angle_alpha {alpha:.4f}",
        f"_cell_angle_beta {beta:.4f}",
        f"_cell_angle_gamma {gamma:.4f}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "_symmetry_Int_Tables_number 1",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_occupancy",
    ]
    
    atom_idx = 1
    for orbit_idx, orbit in enumerate(analyzed.config.orbits):
        species = species_per_orbit[orbit_idx]
        fracs = analyzed.get_fractional_coords(orbit_idx)
        
        for frac in fracs:
            label = f"{species}{atom_idx}"
            lines.append(
                f"{label} {species} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} 1.0"
            )
            atom_idx += 1
    
    return "\n".join(lines)


def export_to_poscar(
    analyzed: AnalyzedConfiguration,
    species_per_orbit: Optional[List[str]] = None,
    title: str = "LatticeSearch Structure"
) -> str:
    """
    Export configuration to VASP POSCAR format.
    
    Args:
        analyzed: Analyzed configuration
        species_per_orbit: Element symbol per orbit
        title: Structure title
    
    Returns:
        POSCAR file content as string
    """
    if species_per_orbit is None:
        species_per_orbit = [chr(65 + i) for i in range(len(analyzed.config.orbits))]
    
    L = analyzed.lattice * analyzed.a
    
    lines = [
        title,
        "1.0",  # Scale factor
        f"  {L[0, 0]:.10f}  {L[1, 0]:.10f}  {L[2, 0]:.10f}",
        f"  {L[0, 1]:.10f}  {L[1, 1]:.10f}  {L[2, 1]:.10f}",
        f"  {L[0, 2]:.10f}  {L[1, 2]:.10f}  {L[2, 2]:.10f}",
        "  ".join(species_per_orbit),
        "  ".join(str(len(orbit)) for orbit in analyzed.config.orbits),
        "Direct",
    ]
    
    for orbit_idx in range(len(analyzed.config.orbits)):
        fracs = analyzed.get_fractional_coords(orbit_idx)
        for frac in fracs:
            lines.append(f"  {frac[0]:.10f}  {frac[1]:.10f}  {frac[2]:.10f}")
    
    return "\n".join(lines)


def export_summary(
    ranked: List[Tuple[AnalyzedConfiguration, float]],
    top_n: int = 10
) -> str:
    """
    Generate a human-readable summary of top configurations.
    """
    lines = [
        "=" * 70,
        "TOP CONFIGURATIONS",
        "=" * 70,
        ""
    ]
    
    for rank, (config, score) in enumerate(ranked[:top_n], 1):
        lines.append(f"Rank {rank}: Score = {score:.4f}")
        lines.append(f"  Parent basis: {config.parent_basis_name}")
        if config.cell_params:
            a, b, c, alpha, beta, gamma = config.cell_params
            lines.append(f"  Cell params: a={a:.3f}, b={b:.3f}, c={c:.3f}, α={alpha:.1f}°, β={beta:.1f}°, γ={gamma:.1f}°")
        lines.append(f"  HNF: diag({config.H[0,0]},{config.H[1,1]},{config.H[2,2]})")
        lines.append(f"  Min distance: {config.min_distance:.4f}a")
        lines.append(f"  Shell gap: {config.shell_gap:.4f}a")
        
        for orbit_idx, sig in config.signatures.items():
            cn1_str = ", ".join(f"O{k}:{v}" for k, v in sig.cn1_by_orbit.items())
            lines.append(f"  Orbit {orbit_idx}: CN1={sig.cn1} ({cn1_str})")
        
        lines.append("")
    
    return "\n".join(lines)


def full_pipeline(
    lattice: np.ndarray,
    a: float,
    orbit_sizes: Tuple[int, ...],
    M_list: List[int],
    constraints: Optional[List[Constraint]] = None,
    weights: Optional[ScoringWeights] = None,
    diagonal_only: bool = False,
    max_per_hnf: int = 20,
    top_n: int = 10,
    verbose: bool = True,
    parent_basis_name: str = "unknown",
    cell_params: Optional[Tuple[float, float, float, float, float, float]] = None
) -> List[Tuple[AnalyzedConfiguration, float]]:
    """
    Complete pipeline: search → filter → rank → return top candidates.
    
    This is the main high-level API for the lattice search tool.
    
    Args:
        lattice: Bravais lattice matrix
        a: Lattice parameter
        orbit_sizes: Target orbit sizes
        M_list: Supercell sizes to search
        constraints: List of constraints (optional)
        weights: Scoring weights (optional)
        diagonal_only: Restrict to diagonal HNFs
        max_per_hnf: Max configs per HNF
        top_n: Return top N ranked configs
        verbose: Print progress
        parent_basis_name: Name of the parent Bravais lattice
        cell_params: Optional (a,b,c,α,β,γ) if lattice from parameters
    
    Returns:
        List of (AnalyzedConfiguration, score) tuples, sorted by score
    
    Example:
        >>> results = full_pipeline(
        ...     lattice=FCC_PRIMITIVE,
        ...     a=4.0,  # Angstroms
        ...     orbit_sizes=(2, 4),
        ...     M_list=[8, 16],
        ...     constraints=[
        ...         MinSeparationConstraint(min_distance=0.3),
        ...         CN1TotalConstraint(orbit_idx=0, target=4)
        ...     ],
        ...     parent_basis_name="FCC-primitive",
        ...     top_n=5
        ... )
    """
    # Step 1: Search
    if verbose:
        print(f"Searching orbit_sizes={orbit_sizes} in M={M_list}...")
    
    configs = search_and_analyze(
        lattice=lattice,
        a=a,
        orbit_sizes=orbit_sizes,
        M_list=M_list,
        diagonal_only=diagonal_only,
        max_per_hnf=max_per_hnf,
        verbose=verbose,
        parent_basis_name=parent_basis_name,
        cell_params=cell_params
    )
    
    if not configs:
        if verbose:
            print("No configurations found.")
        return []
    
    # Step 2: Filter
    if constraints:
        filtered = filter_by_constraints(configs, constraints)
        if verbose:
            print(f"After constraints: {len(filtered)}/{len(configs)} configs")
    else:
        filtered = configs
    
    if not filtered:
        if verbose:
            print("No configurations pass constraints.")
        return []
    
    # Step 3: Rank
    ranked = rank_configurations(filtered, weights, top_n=top_n)
    
    if verbose:
        print(f"Returning top {len(ranked)} configurations")
    
    return ranked


def save_results(
    ranked: List[Tuple[AnalyzedConfiguration, float]],
    output_dir: str = ".",
    prefix: str = "lattice_search",
    formats: List[str] = ["json", "cif", "poscar", "summary"],
    species_per_orbit: Optional[List[str]] = None
):
    """
    Save ranked configurations to files.
    
    Args:
        ranked: List of (config, score) tuples
        output_dir: Output directory
        prefix: Filename prefix
        formats: List of formats to export ("json", "cif", "poscar", "summary")
        species_per_orbit: Element symbols per orbit
    """
    import os
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    
    if "summary" in formats:
        summary = export_summary(ranked)
        with open(os.path.join(output_dir, f"{prefix}_summary.txt"), "w") as f:
            f.write(summary)
    
    for rank, (config, score) in enumerate(ranked, 1):
        if "json" in formats:
            json_data = export_to_json(config, include_shells=True)
            json_data["rank"] = rank
            json_data["score"] = score
            with open(os.path.join(output_dir, f"{prefix}_rank{rank}.json"), "w") as f:
                json.dump(json_data, f, indent=2)
        
        if "cif" in formats:
            cif = export_to_cif(config, species_per_orbit, title=f"Rank{rank}")
            with open(os.path.join(output_dir, f"{prefix}_rank{rank}.cif"), "w") as f:
                f.write(cif)
        
        if "poscar" in formats:
            poscar = export_to_poscar(config, species_per_orbit, title=f"Rank {rank}")
            with open(os.path.join(output_dir, f"{prefix}_rank{rank}.vasp"), "w") as f:
                f.write(poscar)


# =============================================================================
# PHASE 3 DEMO
# =============================================================================

def demo_phase3():
    """Demonstrate Phase 3 capabilities."""
    print("=" * 70)
    print("PHASE 3 DEMO: SEARCH + FILTER + RANK + EXPORT")
    print("=" * 70)
    
    # Search for (2, 4) configurations on FCC
    print("\n1. SEARCH (with uniformity checking)")
    print("-" * 70)
    
    configs = search_and_analyze(
        lattice=FCC_PRIMITIVE,
        a=1.0,
        orbit_sizes=(2, 4),
        M_list=[8],
        diagonal_only=True,  # Fast demo
        max_per_hnf=20,
        verbose=True,
        parent_basis_name="FCC-primitive"
    )
    
    # Report uniformity
    uniform_count = sum(1 for c in configs if c.is_uniform)
    print(f"Uniform configurations: {uniform_count} / {len(configs)}")
    
    # Apply constraints
    print("\n2. CONSTRAINTS")
    print("-" * 70)
    
    # Demonstrate different constraint types
    constraints = [
        UniformityConstraint(),  # Only uniform orbits
        MinSeparationConstraint(min_distance=0.3),
        # Use orbit-specific CN constraint:
        # "Orbit 0 has 8 neighbors from orbit 1"
        CN1ByOrbitConstraint(orbit_idx=0, source_orbit=1, target=8),
    ]
    
    filtered = filter_by_constraints(configs, constraints)
    print(f"After constraints: {len(filtered)} / {len(configs)} configurations")
    
    for c in constraints:
        print(f"  - {c}")
    
    # Show orbit-specific CN breakdown for filtered configs
    if filtered:
        print("\nOrbit-specific CN breakdown:")
        for i, config in enumerate(filtered[:3]):
            print(f"  Config {i+1}:")
            for orbit_idx, sig in config.signatures.items():
                cn_breakdown = ", ".join(f"from O{k}:{v}" for k, v in sig.cn1_by_orbit.items())
                print(f"    Orbit {orbit_idx}: CN1={sig.cn1} ({cn_breakdown})")
    
    # Rank
    print("\n3. RANKING")
    print("-" * 70)
    
    weights = ScoringWeights(
        shell_gap_weight=1.0,
        min_distance_penalty=-2.0,
        cn_targets={0: 8, 1: 10}
    )
    
    ranked = rank_configurations(filtered, weights, top_n=5)
    
    for rank, (config, score) in enumerate(ranked, 1):
        uniform_str = "✓" if config.is_uniform else "✗"
        print(f"  {rank}. Score={score:.3f}, uniform={uniform_str}, CN1=[{config.signatures[0].cn1}, {config.signatures[1].cn1}], d_min={config.min_distance:.3f}")
    
    # Export
    print("\n4. EXPORT")
    print("-" * 70)
    
    if ranked:
        best = ranked[0][0]
        
        # JSON
        json_data = export_to_json(best, include_shells=True)
        print(f"JSON export: {len(json_data['orbits'])} orbits, "
              f"M={json_data['M']}, d_min={json_data['min_distance']:.3f}")
        
        # CIF
        cif_content = export_to_cif(best, species_per_orbit=["Fe", "Ni"])
        print(f"CIF export: {len(cif_content)} characters")
        
        # POSCAR
        poscar_content = export_to_poscar(best, species_per_orbit=["Fe", "Ni"])
        print(f"POSCAR export: {len(poscar_content)} characters")
        
        # Summary
        print("\n" + export_summary(ranked, top_n=3))
    
    return ranked


if __name__ == "__main__":
    main()
