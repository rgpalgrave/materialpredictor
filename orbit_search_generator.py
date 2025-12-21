"""
orbit_search_generator.py

Generates SearchSpec objects for lattice_search.py based on chemistry input.
This module takes user chemistry (cations + anion) and outputs ranked search
specifications for enumerating candidate cation configurations.

The output feeds into lattice_search.py, which produces cation offset configurations
that are then passed to the anion position calculator (lattice predictor).
"""

from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from math import gcd
from functools import reduce
import itertools

# ─────────────────────────────────────────────────────────────────────────────
# Type Definitions
# ─────────────────────────────────────────────────────────────────────────────

SearchSpec = Dict[str, Any]


@dataclass
class CNHypothesis:
    """A hypothesis for cation-cation CN based on cation-anion CN target."""
    template: str       # A, B, C, or D (allocation model)
    CN1: int           # First coordination shell target
    CN2: int           # Second coordination shell target
    priority: int      # Lower = better (for ranking)
    rationale: str     # Human-readable explanation


@dataclass
class OrbitPlan:
    """A plan for how to partition cations into orbits."""
    orbit_sizes: List[int]
    orbit_species: List[str]
    multiplier: int  # Cell multiplier m


# ─────────────────────────────────────────────────────────────────────────────
# Math Utilities
# ─────────────────────────────────────────────────────────────────────────────

def lcm(a: int, b: int) -> int:
    """Least common multiple of two integers."""
    return abs(a * b) // gcd(a, b) if a and b else 0


def lcm_list(numbers: List[int]) -> int:
    """LCM of a list of integers."""
    return reduce(lcm, numbers, 1)


def gcd_list(numbers: List[int]) -> int:
    """GCD of a list of integers."""
    return reduce(gcd, numbers)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Stoichiometry from Charge Balance
# ─────────────────────────────────────────────────────────────────────────────

def derive_stoichiometry(
    cations: List[Dict], 
    anion: Dict
) -> Tuple[List[int], int]:
    """
    Balance charges to get smallest integer stoichiometric coefficients.
    
    Args:
        cations: List of {"element": str, "charge": int, "count": int (optional)}
        anion: {"element": str, "charge": int}
    
    Returns:
        (cation_coefficients, anion_coefficient) as smallest integers
        
    Examples:
        Sr(+2), Ti(+4), O(-2) → [1, 1], 3  (SrTiO3)
        Cs(+1), Sn(+4), Br(-1) → [2, 1], 6  (Cs2SnBr6)
    """
    anion_charge_abs = abs(anion["charge"])
    
    # Case 1: Counts explicitly provided
    if all("count" in c for c in cations):
        cation_counts = [c["count"] for c in cations]
        total_positive = sum(c["count"] * c["charge"] for c in cations)
        
        if total_positive <= 0:
            raise ValueError("Total cation charge must be positive")
        if total_positive % anion_charge_abs != 0:
            raise ValueError(
                f"Cannot balance charges: total cation charge {total_positive} "
                f"not divisible by anion charge magnitude {anion_charge_abs}"
            )
        
        anion_count = total_positive // anion_charge_abs
        
        # Reduce to smallest integers
        g = gcd_list(cation_counts + [anion_count])
        return [c // g for c in cation_counts], anion_count // g
    
    # Case 2: Derive from charges assuming given ratios (default 1:1:...)
    # If some have counts and others don't, use the counts for those that have them
    cation_ratios = []
    for c in cations:
        cation_ratios.append(c.get("count", 1))
    
    # Calculate total positive charge with these ratios
    total_positive = sum(r * c["charge"] for r, c in zip(cation_ratios, cations))
    
    if total_positive <= 0:
        raise ValueError("Total cation charge must be positive")
    
    # Find minimum scale factor to make anion count integral
    # We need: total_positive * scale / anion_charge_abs = integer
    # So: scale must make (total_positive * scale) divisible by anion_charge_abs
    
    scale = anion_charge_abs // gcd(total_positive, anion_charge_abs)
    
    cation_coeffs = [r * scale for r in cation_ratios]
    anion_coeff = (total_positive * scale) // anion_charge_abs
    
    # Reduce to smallest integers
    g = gcd_list(cation_coeffs + [anion_coeff])
    return [c // g for c in cation_coeffs], anion_coeff // g


def format_formula(cations: List[Dict], cation_coeffs: List[int], 
                   anion: Dict, anion_coeff: int) -> str:
    """Format stoichiometry as chemical formula string."""
    parts = []
    for c, coeff in zip(cations, cation_coeffs):
        parts.append(f"{c['element']}{coeff if coeff > 1 else ''}")
    parts.append(f"{anion['element']}{anion_coeff if anion_coeff > 1 else ''}")
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Orbit Plans
# ─────────────────────────────────────────────────────────────────────────────

def generate_orbit_plans(
    cations: List[Dict],
    cation_coeffs: List[int],
    m_max: int,
    M_max: int = 64
) -> List[OrbitPlan]:
    """
    Generate orbit plans: one orbit per cation species, scaled by multiplier m.
    
    For stoichiometry A:B = 1:1, generates plans with orbit_sizes:
    (1,1), (2,2), (3,3), (4,4) for m in 1..m_max
    
    Skips plans where sum(orbit_sizes) > M_max (no valid M possible).
    """
    plans = []
    species = [c["element"] for c in cations]
    
    for m in range(1, m_max + 1):
        sizes = [coeff * m for coeff in cation_coeffs]
        N = sum(sizes)
        
        # Skip if N > M_max: no valid M can exist
        if N > M_max:
            continue
        
        plans.append(OrbitPlan(
            orbit_sizes=sizes,
            orbit_species=species,
            multiplier=m
        ))
    
    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: M_list Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_m_list(
    orbit_sizes: List[int], 
    M_max: int,
    max_entries: int = 4,
    prefer_powers_of_2: bool = True
) -> List[int]:
    """
    Generate valid M values (refinement indices) for lattice_search.
    
    Requirements:
    - M >= N = sum(orbit_sizes)
    - Each orbit_size divides M
    - M <= M_max
    
    Strategy:
    - For simple cases (LCM=1 or 2), prefer powers of 2: [2,4,8,16,...]
      This avoids unhelpful values like 3, 5, 7 that expand search without benefit.
    - For other cases, use LCM multiples as before.
    
    Returns up to max_entries values.
    """
    N = sum(orbit_sizes)
    L = lcm_list(orbit_sizes)
    
    if L == 0:
        return []
    
    m_list = []
    
    # For LCM <= 2 (e.g., orbit_sizes=[1,1] or [2,2]), prefer powers of 2
    if prefer_powers_of_2 and L <= 2:
        # Start from smallest power of 2 >= N that's divisible by L
        power = 1
        while power < N or power % L != 0:
            power *= 2
        
        while len(m_list) < max_entries and power <= M_max:
            m_list.append(power)
            power *= 2
    else:
        # Standard LCM-multiple approach
        t = 1
        while len(m_list) < max_entries:
            M = L * t
            if M > M_max:
                break
            if M >= N:
                m_list.append(M)
            t += 1
    
    return m_list


# ─────────────────────────────────────────────────────────────────────────────
# Shannon Radii Database (Å)
# ─────────────────────────────────────────────────────────────────────────────

# Format: {element: {charge: {CN: radius}}}
# Source: Shannon, R.D. (1976) Acta Cryst. A32, 751-767
# Using "IR" (ionic radii) values where available

SHANNON_RADII = {
    # Alkali metals
    "Li": {1: {4: 0.59, 6: 0.76, 8: 0.92}},
    "Na": {1: {4: 0.99, 6: 1.02, 8: 1.18, 12: 1.39}},
    "K":  {1: {6: 1.38, 8: 1.51, 12: 1.64}},
    "Rb": {1: {6: 1.52, 8: 1.61, 12: 1.72}},
    "Cs": {1: {6: 1.67, 8: 1.74, 12: 1.88}},
    
    # Alkaline earth
    "Mg": {2: {4: 0.57, 6: 0.72, 8: 0.89}},
    "Ca": {2: {6: 1.00, 8: 1.12, 12: 1.34}},
    "Sr": {2: {6: 1.18, 8: 1.26, 12: 1.44}},
    "Ba": {2: {6: 1.35, 8: 1.42, 12: 1.61}},
    
    # Transition metals (common oxidation states)
    "Ti": {3: {6: 0.67}, 4: {4: 0.42, 6: 0.605}},
    "V":  {3: {6: 0.64}, 4: {6: 0.58}, 5: {4: 0.355, 6: 0.54}},
    "Cr": {3: {6: 0.615}, 6: {4: 0.26, 6: 0.44}},
    "Mn": {2: {4: 0.66, 6: 0.83}, 3: {6: 0.645}, 4: {4: 0.39, 6: 0.53}},
    "Fe": {2: {4: 0.63, 6: 0.78}, 3: {4: 0.49, 6: 0.645}},
    "Co": {2: {4: 0.58, 6: 0.745}, 3: {6: 0.61}},
    "Ni": {2: {4: 0.55, 6: 0.69}},
    "Cu": {1: {4: 0.60, 6: 0.77}, 2: {4: 0.57, 6: 0.73}},
    "Zn": {2: {4: 0.60, 6: 0.74}},
    
    # Post-transition metals
    "Al": {3: {4: 0.39, 6: 0.535}},
    "Ga": {3: {4: 0.47, 6: 0.62}},
    "In": {3: {6: 0.80}},
    "Sn": {2: {6: 0.93}, 4: {4: 0.55, 6: 0.69}},
    "Pb": {2: {6: 1.19, 8: 1.29}, 4: {4: 0.65, 6: 0.775}},
    "Bi": {3: {6: 1.03}, 5: {6: 0.76}},
    
    # Lanthanides (3+)
    "La": {3: {6: 1.032, 8: 1.16, 12: 1.36}},
    "Ce": {3: {6: 1.01, 8: 1.143}, 4: {6: 0.87, 8: 0.97}},
    "Nd": {3: {6: 0.983, 8: 1.109}},
    "Eu": {2: {6: 1.17, 8: 1.25}, 3: {6: 0.947}},
    "Gd": {3: {6: 0.938, 8: 1.053}},
    "Y":  {3: {6: 0.90, 8: 1.019}},
    
    # Other common cations
    "Zr": {4: {4: 0.59, 6: 0.72, 8: 0.84}},
    "Hf": {4: {4: 0.58, 6: 0.71, 8: 0.83}},
    "Nb": {5: {4: 0.48, 6: 0.64}},
    "Ta": {5: {6: 0.64}},
    "Mo": {4: {6: 0.65}, 6: {4: 0.41, 6: 0.59}},
    "W":  {4: {6: 0.66}, 6: {4: 0.42, 6: 0.60}},
    
    # Anions (for radius-ratio calculations)
    "O":  {-2: {2: 1.35, 3: 1.36, 4: 1.38, 6: 1.40}},
    "S":  {-2: {6: 1.84}},
    "F":  {-1: {2: 1.285, 4: 1.31, 6: 1.33}},
    "Cl": {-1: {6: 1.81}},
    "Br": {-1: {6: 1.96}},
    "I":  {-1: {6: 2.20}},
}

# Pauling radius-ratio thresholds for CN prediction
# (rho = r_cation / r_anion) -> predicted CN
RADIUS_RATIO_THRESHOLDS = [
    (0.155, 2),   # linear
    (0.225, 3),   # trigonal planar
    (0.414, 4),   # tetrahedral
    (0.732, 6),   # octahedral
    (1.000, 8),   # cubic
    (float('inf'), 12),  # cuboctahedral
]

# Common coordination numbers to prefer
COMMON_CN = frozenset({4, 6, 8, 12})

# Reasonable anion CN values (for consistency check)
REASONABLE_ANION_CN = frozenset({2, 3, 4, 6})


# ─────────────────────────────────────────────────────────────────────────────
# Step 4a: Radius-Ratio CN Prediction
# ─────────────────────────────────────────────────────────────────────────────

def get_shannon_radius(
    element: str, 
    charge: int, 
    cn: int = 6
) -> Optional[float]:
    """
    Get Shannon ionic radius for element/charge/CN.
    Returns None if not in database.
    """
    if element not in SHANNON_RADII:
        return None
    charge_dict = SHANNON_RADII[element].get(charge)
    if charge_dict is None:
        return None
    
    # Try exact CN, then nearest available
    if cn in charge_dict:
        return charge_dict[cn]
    
    # Find nearest CN
    available_cns = sorted(charge_dict.keys())
    if not available_cns:
        return None
    
    nearest = min(available_cns, key=lambda x: abs(x - cn))
    return charge_dict[nearest]


def predict_cn_from_radius_ratio(
    r_cation: float, 
    r_anion: float,
    max_candidates: int = 3
) -> List[int]:
    """
    Predict likely CN values from radius ratio using Pauling rules.
    
    Returns up to max_candidates CN values, ranked by likelihood.
    Includes adjacent CNs when near threshold boundaries.
    """
    rho = r_cation / r_anion
    
    candidates = []
    
    # Find primary CN from threshold
    primary_cn = 6  # default
    primary_idx = 0
    for i, (threshold, cn) in enumerate(RADIUS_RATIO_THRESHOLDS):
        if rho < threshold:
            primary_cn = cn
            primary_idx = i
            break
    
    candidates.append(primary_cn)
    
    # For the 4-6 borderline (rho ~0.35-0.55), both are geometrically viable
    # Include both CNs when in this ambiguous region
    if 0.35 < rho < 0.55:
        for cn in [4, 6]:
            if cn not in candidates:
                candidates.append(cn)
    
    # For the 6-8 borderline (rho ~0.65-0.85), include both
    if 0.65 < rho < 0.85:
        for cn in [6, 8]:
            if cn not in candidates:
                candidates.append(cn)
    
    # For the 8-12 borderline (rho ~0.90-1.10), include both
    if 0.90 < rho < 1.10:
        for cn in [8, 12]:
            if cn not in candidates:
                candidates.append(cn)
    
    # Add adjacent CNs if near boundaries (within 25% of threshold)
    if primary_idx > 0:
        lower_threshold = RADIUS_RATIO_THRESHOLDS[primary_idx - 1][0]
        if rho < lower_threshold * 1.30:
            lower_cn = RADIUS_RATIO_THRESHOLDS[primary_idx - 1][1]
            if lower_cn not in candidates:
                candidates.append(lower_cn)
    
    if primary_idx < len(RADIUS_RATIO_THRESHOLDS) - 1:
        current_threshold = RADIUS_RATIO_THRESHOLDS[primary_idx][0]
        if rho > current_threshold * 0.75:
            higher_cn = RADIUS_RATIO_THRESHOLDS[primary_idx + 1][1]
            if higher_cn not in candidates:
                candidates.append(higher_cn)
    
    # Sort: prefer common CNs, with slight preference for 6 (most common)
    def sort_key(cn):
        is_common = cn in COMMON_CN
        # Octahedral (6) is very common in ionic crystals
        return (not is_common, 0 if cn == 6 else 1, cn)
    
    candidates.sort(key=sort_key)
    
    return candidates[:max_candidates]


def predict_cn_from_field_strength(
    charge: int,
    r_cation: Optional[float] = None,
    max_candidates: int = 3
) -> List[int]:
    """
    Fallback CN prediction using field strength heuristic.
    
    Used when Shannon radii are unavailable.
    Based on charge and estimated size.
    """
    if r_cation is not None:
        # z/r² proxy
        field_strength = charge / (r_cation ** 2)
        
        if field_strength > 8:      # Very high (e.g. Si4+, P5+)
            return [4, 6][:max_candidates]
        elif field_strength > 4:    # High (e.g. Ti4+, Al3+)
            return [6, 4][:max_candidates]
        elif field_strength > 2:    # Medium (e.g. Mg2+, Fe2+)
            return [6, 4, 8][:max_candidates]
        else:                       # Low (e.g. K+, Ba2+)
            return [6, 8, 12][:max_candidates]
    else:
        # Pure charge-based fallback (minimal assumptions)
        if charge >= 5:
            return [4, 6][:max_candidates]
        elif charge >= 4:
            return [6, 4][:max_candidates]
        elif charge >= 3:
            return [6, 4][:max_candidates]
        elif charge >= 2:
            return [6, 4, 8][:max_candidates]
        else:  # charge = 1
            return [6, 8, 4][:max_candidates]


# Charge-class bias rules for CN candidates
# Only apply to LARGE cations where high CN is geometrically favored
# Small cations (Li, Na, Mg) should rely on radius-ratio
CHARGE_CLASS_CN_BIAS = {
    # Large alkaline earth (2+): typically 8-12 in oxides/halides
    # Note: Mg is EXCLUDED - it's small and typically 4-6 coordinate
    ("large_alkaline_earth", 2): [12, 8],
    # Large alkali metals (1+): often high-CN
    # Note: Li, Na are EXCLUDED - they're small/medium
    ("large_alkali", 1): [12, 8],
}

# Element to class mapping - only truly LARGE cations
ELEMENT_CLASS = {
    # Large alkali - exclude Li, Na (they're small/medium)
    "K": "large_alkali", "Rb": "large_alkali", "Cs": "large_alkali",
    # Large alkaline earth - exclude Mg (it's small)
    "Ca": "large_alkaline_earth", "Sr": "large_alkaline_earth", "Ba": "large_alkaline_earth",
}


def get_charge_class_bias(element: str, charge: int) -> Optional[List[int]]:
    """
    Get CN bias based on element class and charge.
    
    Only applies to LARGE cations (K, Rb, Cs, Ca, Sr, Ba).
    Small/medium cations (Li, Na, Mg, transition metals) rely on radius-ratio.
    
    Returns None if no specific bias applies.
    """
    elem_class = ELEMENT_CLASS.get(element)
    if elem_class:
        return CHARGE_CLASS_CN_BIAS.get((elem_class, charge))
    return None


def get_cation_cn_candidates(
    cation: Dict,
    anion: Dict,
    max_candidates: int = 3
) -> List[int]:
    """
    Get CN candidates for a cation using:
    1. Radius-ratio (Pauling) as PRIMARY predictor
    2. Charge-class bias to ADD high-CN options for large cations
    3. Field-strength fallback if radii unavailable
    
    Returns up to max_candidates CN values, ranked by likelihood.
    """
    element = cation["element"]
    charge = cation["charge"]
    anion_element = anion["element"]
    anion_charge = anion["charge"]
    
    # Try to get radii
    r_cation = get_shannon_radius(element, charge, cn=6)
    r_anion = get_shannon_radius(anion_element, anion_charge, cn=6)
    
    if r_cation is not None and r_anion is not None:
        # Use radius-ratio method (primary)
        candidates = predict_cn_from_radius_ratio(r_cation, r_anion, max_candidates)
    else:
        # Fallback to field-strength heuristic
        candidates = predict_cn_from_field_strength(charge, r_cation, max_candidates)
    
    # For LARGE cations only: merge in high-CN options
    # This ensures Sr, Ba, Cs get 12-coordinate options even if radius-ratio is borderline
    class_bias = get_charge_class_bias(element, charge)
    if class_bias:
        seen = set(candidates)
        for cn in class_bias:
            if cn not in seen:
                candidates.append(cn)
                seen.add(cn)
    
    return candidates[:max_candidates]


def get_default_cation_anion_cn(cation: Dict, anion: Dict) -> List[int]:
    """
    Get default cation-anion coordination number candidates.
    
    Uses radius-ratio (Pauling) rules as primary method,
    falls back to field-strength heuristic if radii unavailable.
    
    Returns a list of up to 3 CN candidates, ranked by likelihood.
    """
    return get_cation_cn_candidates(cation, anion, max_candidates=3)


def normalize_cn_targets(
    raw_target: Union[int, List[int], None],
    default: List[int]
) -> List[int]:
    """
    Normalize CN target input to a list.
    
    Accepts:
      - int: single CN target -> [int]
      - List[int]: multiple CN targets -> as-is
      - None: use default
    """
    if raw_target is None:
        return default
    if isinstance(raw_target, int):
        return [raw_target]
    return list(raw_target)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4c: Anion CN Consistency Check
# ─────────────────────────────────────────────────────────────────────────────

def compute_implied_anion_cn(
    cation_coeffs: List[int],
    cation_cns: List[int],
    anion_coeff: int
) -> float:
    """
    Compute implied average anion CN from cation CNs and stoichiometry.
    
    For formula: sum_i(n_i * M_i) + n_X * X
    
    CN_X_avg = sum_i(n_i * CN_M_i_X) / n_X
    
    This is the average number of cations around each anion.
    """
    if anion_coeff == 0:
        return 0.0
    
    total_coordination = sum(n * cn for n, cn in zip(cation_coeffs, cation_cns))
    return total_coordination / anion_coeff


def is_anion_cn_reasonable(
    implied_cn: float,
    target_anion_cn: Optional[int] = None,
    tolerance: float = 1.0
) -> bool:
    """
    Check if implied anion CN is reasonable.
    
    Args:
        implied_cn: Computed average anion CN
        target_anion_cn: User-specified target (if any)
        tolerance: Allowed deviation from target
    
    Returns:
        True if CN is acceptable
    """
    if target_anion_cn is not None:
        # Check against user-specified target
        return abs(implied_cn - target_anion_cn) <= tolerance
    else:
        # Check if it's a "reasonable" anion CN
        # Common values: 2 (linear), 3 (trigonal), 4 (tetrahedral), 6 (octahedral)
        # Allow some tolerance for non-integer values
        for reasonable_cn in REASONABLE_ANION_CN:
            if abs(implied_cn - reasonable_cn) <= tolerance:
                return True
        return False


def filter_cn_combinations_by_anion_cn(
    combinations: List[List[CNHypothesis]],
    cation_coeffs: List[int],
    anion_coeff: int,
    target_anion_cn: Optional[int] = None,
    tolerance: float = 1.0
) -> List[List[CNHypothesis]]:
    """
    Filter CN hypothesis combinations by anion CN consistency.
    
    Removes combinations where implied anion CN is unreasonable.
    """
    filtered = []
    
    for combo in combinations:
        # Extract cation-anion CN from each hypothesis (CN1 = direct cation-anion CN)
        cation_cns = [h.CN1 for h in combo]
        
        implied_cn = compute_implied_anion_cn(cation_coeffs, cation_cns, anion_coeff)
        
        if is_anion_cn_reasonable(implied_cn, target_anion_cn, tolerance):
            filtered.append(combo)
    
    return filtered


def generate_cn_hypotheses(
    cn_mx_target: int, 
    max_hypotheses: int = 6
) -> List[CNHypothesis]:
    """
    Generate candidate (CN1, CN2) pairs for cation-cation network
    based on target cation-anion CN.
    
    These are bond-allocation models describing how cation-anion bonds
    distribute across cation-cation coordination shells:
    
    - Model A: CN_MX ≈ CN1 (bonds via shell 1 only)
    - Model B: CN_MX ≈ CN1 + CN2/2 (shell 1 + half shell 2, edge-sharing)
    - Model C: CN_MX ≈ CN1 + CN2 (shell 1 + shell 2, full sharing)
    - Model D: CN_MX ≈ 2*CN1 (dense sharing, face-sharing)
    
    For large CN targets (≥8), Model B and D alternatives are boosted
    since achieving very high CN1 directly is geometrically challenging.
    
    Returns list sorted by priority (lower = better).
    """
    hypotheses = []
    is_high_cn = cn_mx_target >= 8  # High CN benefits from alternative models
    
    # Model A: CN1 = CN_MX_target, CN2 = 0
    # Bonds allocated only to first coordination shell
    if 2 <= cn_mx_target <= 16:
        # For high CN, demote Model A slightly - it's hard to achieve CN1=12 directly
        priority = 1 if cn_mx_target in COMMON_CN else 2
        if is_high_cn and cn_mx_target > 8:
            priority = 2  # Demote for very high CN
        
        hypotheses.append(CNHypothesis(
            template="A",
            CN1=cn_mx_target,
            CN2=0,
            priority=priority,
            rationale=f"Model A: CN1={cn_mx_target} (shell 1 only)"
        ))
    
    # Model B: CN1 + CN2/2 = target
    # Bonds split: shell 1 gets full bonds, shell 2 shares (edge-sharing geometry)
    # For high CN, this is often more achievable than pure Model A
    for cn1 in range(2, min(17, cn_mx_target + 1)):
        cn2_needed = 2 * (cn_mx_target - cn1)
        
        if cn2_needed < 0 or cn2_needed > 24:
            continue
        if cn1 == cn_mx_target and cn2_needed == 0:
            continue  # Already covered by Model A
        
        # Base priority
        priority = 3
        
        # Boost when CN1 is common and CN2 is reasonable
        if cn1 in COMMON_CN:
            priority = 2
        
        # For high CN targets, boost Model B with moderate CN1 (4, 6, 8)
        if is_high_cn and cn1 in {4, 6, 8} and cn2_needed <= 16:
            priority = 1  # Make competitive with Model A
        
        priority += cn2_needed // 10  # Smaller penalty for large CN2
        
        hypotheses.append(CNHypothesis(
            template="B",
            CN1=cn1,
            CN2=cn2_needed,
            priority=priority,
            rationale=f"Model B: CN1={cn1} + CN2/2={cn2_needed//2} = {cn_mx_target}"
        ))
    
    # Model C: CN1 + CN2 = target
    # Bonds split evenly across both shells
    for cn1 in range(2, min(17, cn_mx_target)):
        cn2_needed = cn_mx_target - cn1
        
        if cn2_needed < 1 or cn2_needed > 24:
            continue
        
        priority = 4
        if cn1 in COMMON_CN and cn2_needed in COMMON_CN:
            priority = 3
        
        # For high CN, boost when both shells are common
        if is_high_cn and cn1 in {4, 6} and cn2_needed in {4, 6, 8}:
            priority = 2
        
        priority += cn2_needed // 8  # Penalty for large CN2
        
        hypotheses.append(CNHypothesis(
            template="C",
            CN1=cn1,
            CN2=cn2_needed,
            priority=priority,
            rationale=f"Model C: CN1={cn1} + CN2={cn2_needed} = {cn_mx_target}"
        ))
    
    # Model D: CN1 = target/2 (dense packing with face-sharing)
    # Each cation-anion bond shared by 2 cation-cation contacts
    # For high CN (like 12), CN1=6 is very achievable
    if cn_mx_target % 2 == 0:
        cn1 = cn_mx_target // 2
        if 2 <= cn1 <= 16:
            priority = 5 if cn1 in COMMON_CN else 6
            
            # For high CN targets, boost Model D significantly
            if is_high_cn and cn1 in COMMON_CN:
                priority = 2  # Make competitive
            
            hypotheses.append(CNHypothesis(
                template="D",
                CN1=cn1,
                CN2=0,
                priority=priority,
                rationale=f"Model D: 2×CN1 = 2×{cn1} = {cn_mx_target} (dense sharing)"
            ))
    
    # Sort by priority, then by smaller CN2, then by CN1 in common set
    hypotheses.sort(key=lambda h: (
        h.priority, 
        h.CN2, 
        0 if h.CN1 in COMMON_CN else 1,
        -h.CN1
    ))
    
    return hypotheses[:max_hypotheses]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Build SearchSpecs
# ─────────────────────────────────────────────────────────────────────────────

def build_parent_lattices(
    has_edge_sharing_model: bool,
    allowed_parents: Optional[List[str]] = None
) -> List[Dict]:
    """
    Build list of parent lattice specifications.
    
    Always includes SC, BCC, FCC with a=1.
    If edge-sharing model (Model B) is used, adds tetragonal scan.
    
    Each entry includes:
    - metric_family: "bravais" or "params"
      - "bravais": use predefined primitive matrices (SC/BCC/FCC)
      - "params": call lattice_matrix_from_params(params), then apply scan
    """
    default_parents = ["SC", "BCC", "FCC"]
    if allowed_parents:
        default_parents = [p for p in default_parents if p in allowed_parents]
    
    parents = []
    
    for name in default_parents:
        parents.append({
            "name": name,
            "metric_family": "bravais",  # Use predefined primitive matrices
            "params": None,
            "scan": None
        })
    
    # Add tetragonal scan for edge-sharing models (B, C)
    # These often produce tetragonal distortions
    if has_edge_sharing_model:
        parents.append({
            "name": "TRICLINIC",
            "metric_family": "params",  # Build via lattice_matrix_from_params()
            "params": {
                "a": 1.0, "b": 1.0, "c": 1.0,
                "alpha": 90.0, "beta": 90.0, "gamma": 90.0
            },
            "scan": {
                "param": "c/a",
                "values": [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
            }
        })
    
    return parents


def build_constraints(
    orbit_hypotheses: List[CNHypothesis],
    orbit_species: List[str] = None,
    min_separation: float = 0.3,
    shell_gap: float = 0.08,
    majority_neighbor_fraction: float = 0.7  # Raised from 0.5 - stricter A-B preference
) -> List[Dict]:
    """
    Build constraint list from orbit CN hypotheses.
    
    Constraints are on the cation-only network (CN shells), not cation-anion CN.
    
    For two-orbit systems, adds MajorityNeighbor constraints to prefer
    A-B neighbors over A-A or B-B (common in ionic crystals).
    
    For "big cation" species (Sr, Ba, Ca, K, Rb, Cs), uses CN1Range instead of
    CN1Total to allow flexibility in achieving high coordination.
    """
    # Big cations that benefit from CN range constraints
    BIG_CATIONS = {"Sr", "Ba", "Ca", "K", "Rb", "Cs", "La", "Ce", "Nd", "Y"}
    
    constraints = []
    n_orbits = len(orbit_hypotheses)
    
    # CN constraints per orbit
    for i, hyp in enumerate(orbit_hypotheses):
        species = orbit_species[i] if orbit_species and i < len(orbit_species) else None
        is_big_cation = species in BIG_CATIONS if species else False
        
        if hyp.CN1 > 0:
            if is_big_cation and hyp.CN1 >= 8:
                # Use range for big cations with high CN targets
                # Allow CN1 from (target-4) to target
                min_cn = max(4, hyp.CN1 - 4)
                constraints.append({
                    "type": "CN1Range",
                    "orbit": i,
                    "min_cn": min_cn,
                    "max_cn": hyp.CN1
                })
            else:
                # Exact constraint for small cations or low CN
                constraints.append({
                    "type": "CN1Total",
                    "orbit": i,
                    "target": hyp.CN1
                })
        
        if hyp.CN2 > 0:
            if is_big_cation:
                # Also use range for CN2 on big cations
                min_cn2 = max(0, hyp.CN2 - 4)
                constraints.append({
                    "type": "CN2Range",
                    "orbit": i,
                    "min_cn": min_cn2,
                    "max_cn": hyp.CN2 + 4
                })
            else:
                constraints.append({
                    "type": "CN2Total",
                    "orbit": i,
                    "target": hyp.CN2
                })
    
    # For two-orbit systems: add cross-orbit neighbor preference
    # 0.7 is stricter than 0.5, reduces mixed A-A / B-B adjacency
    if n_orbits == 2 and majority_neighbor_fraction > 0:
        constraints.append({
            "type": "MajorityNeighbor",
            "orbit": 0,
            "neighbor_orbit": 1,
            "min_fraction": majority_neighbor_fraction
        })
        constraints.append({
            "type": "MajorityNeighbor",
            "orbit": 1,
            "neighbor_orbit": 0,
            "min_fraction": majority_neighbor_fraction
        })
    
    # Generic structural constraints
    constraints.append({
        "type": "MinSeparation",
        "min_distance": min_separation
    })
    constraints.append({
        "type": "ShellGap",
        "min_gap": shell_gap
    })
    constraints.append({
        "type": "OrbitUniform"
    })
    
    return constraints


def build_generator_priors() -> Dict[str, Any]:
    """
    Build generator priors/metadata for this SearchSpec.
    
    These document the first-principles approach used.
    No structure-specific overrides are needed.
    """
    return {
        "model_preference": "A > B > C > D",
        "cn_method": "Pauling radius-ratio with Shannon radii (first principles)",
        "cn_bias": "Large cations only (K,Rb,Cs,Ca,Sr,Ba)",
        "orbit_model": "one_per_species",
        "neighbor_model": "majority_cross_orbit_for_2_species",
        "anion_cn_filter": "stoichiometry_consistency_check",
        "overrides_needed": False
    }


def compute_template_priority(templates: List[str]) -> int:
    """
    Compute overall allocation model priority score.
    Lower = simpler = better.
    
    Model A (shell 1 only) is simplest.
    Model D (dense sharing) is most complex.
    """
    priority_map = {"A": 1, "B": 2, "C": 3, "D": 4}
    return sum(priority_map.get(t, 5) for t in templates)


def create_search_spec(
    label: str,
    orbit_sizes: List[int],
    orbit_species: List[str],
    m_list: List[int],
    parent_lattices: List[Dict],
    constraints: List[Dict],
    generator_priors: Dict[str, Any],
    chemistry: Dict,
    templates: List[str],
    template_priority: int
) -> SearchSpec:
    """Create a SearchSpec dictionary with internal ranking metadata."""
    return {
        "label": label,
        "orbit_sizes": orbit_sizes,
        "orbit_species": orbit_species,
        "M_list": m_list,
        "parent_lattices": parent_lattices,
        "constraints": constraints,
        "generator_priors": generator_priors,
        "chemistry": chemistry,
        # Internal metadata for ranking (removed before output)
        "_templates": templates,
        "_template_priority": template_priority,
        "_min_M": min(m_list) if m_list else 999,
        "_num_constraints": len(constraints)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Ranking
# ─────────────────────────────────────────────────────────────────────────────

def rank_specs(specs: List[SearchSpec]) -> List[SearchSpec]:
    """
    Rank SearchSpecs by preference:
    1. Smaller M first (simpler cell)
    2. Simpler allocation model first (A > B > C > D)
    3. Fewer constraints (simpler search)
    """
    return sorted(specs, key=lambda s: (
        s["_min_M"],
        s["_template_priority"],
        s["_num_constraints"]
    ))


def clean_spec(spec: SearchSpec) -> SearchSpec:
    """Remove internal ranking metadata from SearchSpec."""
    return {k: v for k, v in spec.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────────────
# Main Generator Function
# ─────────────────────────────────────────────────────────────────────────────

def generate_search_specs(
    chemistry: Dict,
    top_X: int = 20,
    m_max: int = 4,
    M_max: int = 64,
    max_specs_per_plan: int = 50
) -> List[SearchSpec]:
    """
    Generate ranked SearchSpec objects for lattice_search.
    
    Args:
        chemistry: {
            "cations": [
                {"element": str, "charge": int, "count": int (optional)},
                ...
            ],
            "anion": {"element": str, "charge": int},
            "overrides": {
                "cation_cn": {"Ti": 6, "Sr": [8, 12], ...},  # int or list
                "anion_cn": int,               # anion CN target (for echo)
                "parent_lattices": ["SC", "BCC", "FCC"],  # allowed parents
            } (optional)
        }
        top_X: Maximum number of SearchSpecs to return
        m_max: Maximum cell multiplier for orbit sizes
        M_max: Maximum refinement index
        max_specs_per_plan: Maximum specs to generate per orbit plan
                           (caps combinatorics for multi-cation systems)
    
    Returns:
        List of SearchSpec dictionaries, ranked by preference.
        
    Example:
        >>> chemistry = {
        ...     "cations": [
        ...         {"element": "Sr", "charge": 2},
        ...         {"element": "Ti", "charge": 4}
        ...     ],
        ...     "anion": {"element": "O", "charge": -2},
        ...     "overrides": {"cation_cn": {"Ti": 6, "Sr": [8, 12]}}
        ... }
        >>> specs = generate_search_specs(chemistry, top_X=10)
    """
    cations = chemistry["cations"]
    anion = chemistry["anion"]
    overrides = chemistry.get("overrides", {})
    allowed_parents = overrides.get("parent_lattices")
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Derive stoichiometry
    # ─────────────────────────────────────────────────────────────────────
    cation_coeffs, anion_coeff = derive_stoichiometry(cations, anion)
    formula = format_formula(cations, cation_coeffs, anion, anion_coeff)
    
    stoichiometry = {
        "cations": {c["element"]: coeff for c, coeff in zip(cations, cation_coeffs)},
        "anion": {anion["element"]: anion_coeff}
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Get CN targets for each cation species (now supports lists)
    # ─────────────────────────────────────────────────────────────────────
    cation_cn_overrides = overrides.get("cation_cn", {})
    cn_targets = {}  # element -> List[int]
    
    for c in cations:
        elem = c["element"]
        raw_override = cation_cn_overrides.get(elem)
        default_cn = get_default_cation_anion_cn(c, anion)
        cn_targets[elem] = normalize_cn_targets(raw_override, default_cn)
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Generate orbit plans (with M_max guardrail)
    # ─────────────────────────────────────────────────────────────────────
    orbit_plans = generate_orbit_plans(cations, cation_coeffs, m_max, M_max)
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Generate CN hypotheses for each (species, target) pair
    # ─────────────────────────────────────────────────────────────────────
    # For each species, collect hypotheses for ALL its CN targets,
    # then sort by priority so best combos come first in product
    cn_hypotheses_by_species = {}
    for c in cations:
        elem = c["element"]
        all_hyps = []
        for target_cn in cn_targets[elem]:
            hyps = generate_cn_hypotheses(target_cn)
            all_hyps.extend(hyps)
        
        # Sort by priority (lower = better), dedupe by (CN1, CN2)
        seen = set()
        sorted_hyps = []
        for h in sorted(all_hyps, key=lambda x: (x.priority, x.CN2, -x.CN1)):
            key = (h.CN1, h.CN2)
            if key not in seen:
                seen.add(key)
                sorted_hyps.append(h)
        
        # Cap at 5 hypotheses per species as specified
        cn_hypotheses_by_species[elem] = sorted_hyps[:5]
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Build SearchSpecs (with combinatorics cap)
    # ─────────────────────────────────────────────────────────────────────
    all_specs = []
    
    for plan in orbit_plans:
        orbit_sizes = plan.orbit_sizes
        orbit_species = plan.orbit_species
        
        # Generate M_list for this orbit plan
        m_list = generate_m_list(orbit_sizes, M_max)
        if not m_list:
            continue  # Skip if no valid M values
        
        # Get hypothesis lists (already sorted by priority)
        hypothesis_lists = [cn_hypotheses_by_species[sp] for sp in orbit_species]
        
        # Generate all combinations
        all_combos = list(itertools.product(*hypothesis_lists))
        
        # Filter by anion CN consistency (Step 2 from suggestions)
        # This uses stoichiometry to prune combinations where implied anion CN
        # doesn't match target or isn't reasonable
        target_anion_cn = overrides.get("anion_cn")
        
        # Scale cation_coeffs by multiplier for this plan
        scaled_cation_coeffs = [c * plan.multiplier for c in cation_coeffs]
        scaled_anion_coeff = anion_coeff * plan.multiplier
        
        filtered_combos = filter_cn_combinations_by_anion_cn(
            [list(c) for c in all_combos],
            scaled_cation_coeffs,
            scaled_anion_coeff,
            target_anion_cn,
            tolerance=1.0
        )
        
        # Generate specs from filtered combinations, capped at max_specs_per_plan
        specs_this_plan = 0
        for combo_list in filtered_combos:
            if specs_this_plan >= max_specs_per_plan:
                break
            
            templates = [h.template for h in combo_list]
            
            # Check if any edge-sharing model (B or C) - these benefit from tetragonal scan
            has_edge_sharing = any(t in ("B", "C") for t in templates)
            
            # Build constraints (pass orbit_species for CN1Range on big cations)
            constraints = build_constraints(combo_list, orbit_species=orbit_species)
            
            # Build parent lattices
            parents = build_parent_lattices(has_edge_sharing, allowed_parents)
            
            # Compute implied anion CN for label
            cation_cns = [h.CN1 for h in combo_list]
            implied_anion_cn = compute_implied_anion_cn(
                scaled_cation_coeffs, cation_cns, scaled_anion_coeff
            )
            
            # Build descriptive label
            template_str = "+".join(templates)
            cn_parts = [f"{sp}:CN1={h.CN1}" + (f",CN2={h.CN2}" if h.CN2 else "")
                        for sp, h in zip(orbit_species, combo_list)]
            anion_cn_str = f" X_CN≈{implied_anion_cn:.1f}"
            label = f"{formula} m={plan.multiplier} [{template_str}] {' '.join(cn_parts)}{anion_cn_str}"
            
            # Chemistry echo (now includes all CN targets and implied anion CN)
            chem_echo = {
                "formula": formula,
                "stoichiometry": stoichiometry,
                "cn_targets": cn_targets,
                "anion_cn_target": target_anion_cn,
                "anion_cn_implied": round(implied_anion_cn, 2)
            }
            
            # Create spec
            spec = create_search_spec(
                label=label,
                orbit_sizes=orbit_sizes,
                orbit_species=orbit_species,
                m_list=m_list,
                parent_lattices=parents,
                constraints=constraints,
                generator_priors=build_generator_priors(),
                chemistry=chem_echo,
                templates=templates,
                template_priority=compute_template_priority(templates)
            )
            
            all_specs.append(spec)
            specs_this_plan += 1
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Rank and return top_X
    # ─────────────────────────────────────────────────────────────────────
    ranked = rank_specs(all_specs)
    cleaned = [clean_spec(s) for s in ranked[:top_X]]
    
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Validation Utilities
# ─────────────────────────────────────────────────────────────────────────────

def validate_search_spec(spec: SearchSpec) -> List[str]:
    """
    Validate a SearchSpec and return list of issues.
    
    Checks:
    - orbit_sizes sum > 0
    - M_list is non-empty
    - Each M is divisible by each orbit_size
    - orbit_sizes and orbit_species have same length
    """
    issues = []
    
    if sum(spec["orbit_sizes"]) == 0:
        issues.append("orbit_sizes sum to 0")
    
    if not spec["M_list"]:
        issues.append("M_list is empty")
    else:
        for M in spec["M_list"]:
            for i, size in enumerate(spec["orbit_sizes"]):
                if M % size != 0:
                    issues.append(
                        f"M={M} not divisible by orbit_size[{i}]={size}"
                    )
    
    if len(spec["orbit_sizes"]) != len(spec["orbit_species"]):
        issues.append("orbit_sizes and orbit_species length mismatch")
    
    # Check constraints reference valid orbits
    n_orbits = len(spec["orbit_sizes"])
    for c in spec["constraints"]:
        if "orbit" in c:
            if c["orbit"] >= n_orbits:
                issues.append(f"Constraint references invalid orbit {c['orbit']}")
        if "neighbor_orbit" in c:
            if c["neighbor_orbit"] >= n_orbits:
                issues.append(
                    f"Constraint references invalid neighbor_orbit {c['neighbor_orbit']}"
                )
    
    return issues


def validate_chemistry(chemistry: Dict) -> List[str]:
    """Validate chemistry input before generating specs."""
    issues = []
    
    if "cations" not in chemistry:
        issues.append("Missing 'cations' key")
    elif not chemistry["cations"]:
        issues.append("Empty cations list")
    else:
        for i, c in enumerate(chemistry["cations"]):
            if "element" not in c:
                issues.append(f"Cation {i}: missing 'element'")
            if "charge" not in c:
                issues.append(f"Cation {i}: missing 'charge'")
            elif c["charge"] <= 0:
                issues.append(f"Cation {i}: charge must be positive")
    
    if "anion" not in chemistry:
        issues.append("Missing 'anion' key")
    else:
        if "element" not in chemistry["anion"]:
            issues.append("Anion: missing 'element'")
        if "charge" not in chemistry["anion"]:
            issues.append("Anion: missing 'charge'")
        elif chemistry["anion"]["charge"] >= 0:
            issues.append("Anion: charge must be negative")
    
    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Demo / Testing
# ─────────────────────────────────────────────────────────────────────────────

def demo():
    """Run demonstration with common crystal structures."""
    
    examples = [
        # Perovskite: SrTiO3 (with list-based CN for Sr)
        {
            "name": "Perovskite (SrTiO3)",
            "chemistry": {
                "cations": [
                    {"element": "Sr", "charge": 2},
                    {"element": "Ti", "charge": 4}
                ],
                "anion": {"element": "O", "charge": -2},
                "overrides": {
                    "cation_cn": {"Ti": 6, "Sr": [8, 12]}  # Sr can be 8 or 12
                }
            }
        },
        # Vacancy-ordered double perovskite: Cs2SnBr6
        {
            "name": "Vacancy-ordered double perovskite (Cs2SnBr6)",
            "chemistry": {
                "cations": [
                    {"element": "Cs", "charge": 1, "count": 2},
                    {"element": "Sn", "charge": 4, "count": 1}
                ],
                "anion": {"element": "Br", "charge": -1},
                "overrides": {
                    "cation_cn": {"Sn": 6, "Cs": 12}
                }
            }
        },
        # Rutile: TiO2
        {
            "name": "Rutile (TiO2)",
            "chemistry": {
                "cations": [
                    {"element": "Ti", "charge": 4}
                ],
                "anion": {"element": "O", "charge": -2},
                "overrides": {
                    "cation_cn": {"Ti": 6}
                }
            }
        },
        # Spinel: MgAl2O4
        {
            "name": "Spinel (MgAl2O4)",
            "chemistry": {
                "cations": [
                    {"element": "Mg", "charge": 2, "count": 1},
                    {"element": "Al", "charge": 3, "count": 2}
                ],
                "anion": {"element": "O", "charge": -2},
                "overrides": {
                    "cation_cn": {"Mg": 4, "Al": 6}
                }
            }
        },
        # 4-cation system to test combinatorics cap
        {
            "name": "4-cation test (combinatorics cap)",
            "chemistry": {
                "cations": [
                    {"element": "A", "charge": 1, "count": 1},
                    {"element": "B", "charge": 2, "count": 1},
                    {"element": "C", "charge": 3, "count": 1},
                    {"element": "D", "charge": 4, "count": 1}
                ],
                "anion": {"element": "X", "charge": -2},
            }
        },
    ]
    
    for example in examples:
        print(f"\n{'='*70}")
        print(f"Example: {example['name']}")
        print('='*70)
        
        chemistry = example["chemistry"]
        
        # Validate
        issues = validate_chemistry(chemistry)
        if issues:
            print(f"Validation issues: {issues}")
            continue
        
        # Generate specs
        specs = generate_search_specs(chemistry, top_X=5)
        
        print(f"\nGenerated {len(specs)} SearchSpecs (showing top 5):\n")
        
        for i, spec in enumerate(specs):
            print(f"--- Spec {i+1} ---")
            print(f"  Label: {spec['label']}")
            print(f"  Orbit sizes: {spec['orbit_sizes']}")
            print(f"  Orbit species: {spec['orbit_species']}")
            print(f"  M_list: {spec['M_list']}")
            print(f"  Parents: {[p['name'] for p in spec['parent_lattices']]}")
            print(f"  CN targets (input): {spec['chemistry']['cn_targets']}")
            print(f"  Constraints ({len(spec['constraints'])}):")
            for c in spec['constraints']:
                print(f"    {c}")
            
            # Validate
            val_issues = validate_search_spec(spec)
            if val_issues:
                print(f"  ⚠ Validation issues: {val_issues}")


if __name__ == "__main__":
    demo()
