"""
lattice_search_integration.py

Integration layer connecting orbit_search_generator.py (upstream) to 
lattice_search.py (core) and preparing output for anion_calculator (downstream).

Pipeline:
    Chemistry Input 
        → orbit_search_generator.generate_search_specs() 
        → THIS MODULE: run_search_spec() 
        → Cation Configs (for anion_calculator)

This module:
1. Converts SearchSpec constraints to lattice_search Constraint objects
2. Handles parent lattice resolution (bravais vs params with scan)
3. Runs the search pipeline
4. Formats output for downstream consumption
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Import from lattice_search
from lattice_search import (
    # Lattice matrices
    SC_PRIMITIVE, BCC_PRIMITIVE, FCC_PRIMITIVE,
    lattice_matrix_from_params,
    
    # Core search
    search_and_analyze,
    filter_by_constraints,
    rank_configurations,
    
    # Constraint classes
    Constraint,
    CN1TotalConstraint,
    CN1RangeConstraint,
    CN1ByOrbitConstraint,
    CN2TotalConstraint,
    CN2RangeConstraint,
    CN2ByOrbitConstraint,
    FirstShellCompositionConstraint,
    SecondShellCompositionConstraint,
    FirstShellDistanceConstraint,
    MinSeparationConstraint,
    ShellGapConstraint,
    MajorityNeighborConstraint,
    DistinctCNConstraint,
    ShellRatioConstraint,
    UniformityConstraint,
    
    # Scoring
    ScoringWeights,
    
    # Types
    AnalyzedConfiguration,
)

# Import from orbit_search_generator
from orbit_search_generator import (
    generate_search_specs,
    validate_search_spec,
    validate_chemistry,
    SearchSpec,
)


# =============================================================================
# CONSTRAINT CONVERSION
# =============================================================================

def convert_constraint(constraint_dict: Dict) -> Constraint:
    """
    Convert a constraint dict from SearchSpec to a Constraint object.
    
    Mapping:
        CN1Total → CN1TotalConstraint
        CN2Total → CN2TotalConstraint
        CN1ByOrbit → CN1ByOrbitConstraint
        CN2ByOrbit → CN2ByOrbitConstraint
        MajorityNeighbor → MajorityNeighborConstraint
        MinSeparation → MinSeparationConstraint
        ShellGap → ShellGapConstraint
        OrbitUniform → UniformityConstraint
    """
    ctype = constraint_dict["type"]
    
    if ctype == "CN1Total":
        return CN1TotalConstraint(
            orbit_idx=constraint_dict["orbit"],
            target=constraint_dict["target"]
        )
    
    elif ctype == "CN2Total":
        return CN2TotalConstraint(
            orbit_idx=constraint_dict["orbit"],
            target=constraint_dict["target"]
        )
    
    elif ctype == "CN1ByOrbit":
        return CN1ByOrbitConstraint(
            orbit_idx=constraint_dict["orbit"],
            source_orbit=constraint_dict["neighbor_orbit"],
            target=constraint_dict["target"]
        )
    
    elif ctype == "CN2ByOrbit":
        return CN2ByOrbitConstraint(
            orbit_idx=constraint_dict["orbit"],
            source_orbit=constraint_dict["neighbor_orbit"],
            target=constraint_dict["target"]
        )
    
    elif ctype == "MajorityNeighbor":
        return MajorityNeighborConstraint(
            orbit_idx=constraint_dict["orbit"],
            neighbor_orbit=constraint_dict["neighbor_orbit"],
            min_fraction=constraint_dict.get("min_fraction", 0.5)
        )
    
    elif ctype == "MinSeparation":
        return MinSeparationConstraint(
            min_distance=constraint_dict["min_distance"]
        )
    
    elif ctype == "ShellGap":
        return ShellGapConstraint(
            min_gap=constraint_dict["min_gap"]
        )
    
    elif ctype == "OrbitUniform":
        return UniformityConstraint()
    
    elif ctype == "CN1Range":
        return CN1RangeConstraint(
            orbit_idx=constraint_dict["orbit"],
            min_cn=constraint_dict["min_cn"],
            max_cn=constraint_dict["max_cn"]
        )
    
    elif ctype == "CN2Range":
        return CN2RangeConstraint(
            orbit_idx=constraint_dict["orbit"],
            min_cn=constraint_dict["min_cn"],
            max_cn=constraint_dict["max_cn"]
        )
    
    else:
        raise ValueError(f"Unknown constraint type: {ctype}")


def convert_constraints(constraint_dicts: List[Dict]) -> List[Constraint]:
    """Convert list of constraint dicts to Constraint objects."""
    return [convert_constraint(c) for c in constraint_dicts]


# =============================================================================
# PARENT LATTICE RESOLUTION
# =============================================================================

# Predefined Bravais lattices
BRAVAIS_LATTICES = {
    "SC": SC_PRIMITIVE,
    "BCC": BCC_PRIMITIVE,
    "FCC": FCC_PRIMITIVE,
}


def resolve_parent_lattice(
    parent_spec: Dict
) -> List[Tuple[np.ndarray, str, Optional[Dict]]]:
    """
    Resolve a parent lattice specification to concrete lattice matrices.
    
    Args:
        parent_spec: Dict with keys:
            - name: str (e.g., "SC", "BCC", "FCC", "TRICLINIC")
            - metric_family: "bravais" or "params"
            - params: Optional dict with a,b,c,alpha,beta,gamma
            - scan: Optional dict with param and values for parameter sweep
    
    Returns:
        List of (lattice_matrix, name, cell_params) tuples.
        For scans, returns one tuple per scan value.
    """
    name = parent_spec["name"]
    metric_family = parent_spec["metric_family"]
    params = parent_spec.get("params")
    scan = parent_spec.get("scan")
    
    results = []
    
    if metric_family == "bravais":
        # Use predefined primitive matrix
        if name not in BRAVAIS_LATTICES:
            raise ValueError(f"Unknown Bravais lattice: {name}")
        
        lattice = BRAVAIS_LATTICES[name]
        # Cell params for cubic Bravais (a=1)
        cell_params = (1.0, 1.0, 1.0, 90.0, 90.0, 90.0)
        results.append((lattice, f"{name}-primitive", cell_params))
    
    elif metric_family == "params":
        # Build from parameters, possibly with scan
        if params is None:
            raise ValueError("metric_family='params' requires 'params' dict")
        
        if scan is None:
            # Single lattice from params
            lattice = lattice_matrix_from_params(
                params["a"], params["b"], params["c"],
                params["alpha"], params["beta"], params["gamma"]
            )
            cell_params = (
                params["a"], params["b"], params["c"],
                params["alpha"], params["beta"], params["gamma"]
            )
            results.append((lattice, name, cell_params))
        else:
            # Parameter sweep
            scan_param = scan["param"]
            scan_values = scan["values"]
            
            for val in scan_values:
                # Apply scan value
                p = params.copy()
                
                if scan_param == "c/a":
                    p["c"] = p["a"] * val
                elif scan_param == "b/a":
                    p["b"] = p["a"] * val
                elif scan_param == "alpha":
                    p["alpha"] = val
                elif scan_param == "beta":
                    p["beta"] = val
                elif scan_param == "gamma":
                    p["gamma"] = val
                else:
                    # Direct parameter
                    p[scan_param] = val
                
                lattice = lattice_matrix_from_params(
                    p["a"], p["b"], p["c"],
                    p["alpha"], p["beta"], p["gamma"]
                )
                cell_params = (
                    p["a"], p["b"], p["c"],
                    p["alpha"], p["beta"], p["gamma"]
                )
                scan_label = f"{name}({scan_param}={val})"
                results.append((lattice, scan_label, cell_params))
    
    else:
        raise ValueError(f"Unknown metric_family: {metric_family}")
    
    return results


# =============================================================================
# CATION CONFIG OUTPUT FORMAT
# =============================================================================

def format_cation_config(
    analyzed: AnalyzedConfiguration,
    orbit_species: List[str],
    score: float
) -> Dict:
    """
    Format an AnalyzedConfiguration for downstream consumption (anion calculator).
    
    Output format matches the expected CationConfig structure from the handoff doc.
    """
    # Build offsets (fractional coordinates per orbit)
    offsets = []
    for orbit_idx, orbit in enumerate(analyzed.config.orbits):
        frac_coords = analyzed.get_fractional_coords(orbit_idx)
        offsets.append({
            "orbit": orbit_idx,
            "species": orbit_species[orbit_idx] if orbit_idx < len(orbit_species) else f"X{orbit_idx}",
            "frac_coords": [fc.tolist() for fc in frac_coords]
        })
    
    # Build CN shells info
    cn1_list = []
    cn2_list = []
    cn1_by_orbit = []
    
    for orbit_idx in range(len(analyzed.config.orbits)):
        sig = analyzed.signatures.get(orbit_idx)
        if sig:
            cn1_list.append(sig.cn1)
            cn2_list.append(sig.cn2)
            cn1_by_orbit.append(dict(sig.cn1_by_orbit))
        else:
            cn1_list.append(0)
            cn2_list.append(0)
            cn1_by_orbit.append({})
    
    return {
        "lattice": {
            "name": analyzed.parent_basis_name,
            "params": {
                "a": analyzed.cell_params[0] if analyzed.cell_params else 1.0,
                "b": analyzed.cell_params[1] if analyzed.cell_params else 1.0,
                "c": analyzed.cell_params[2] if analyzed.cell_params else 1.0,
                "alpha": analyzed.cell_params[3] if analyzed.cell_params else 90.0,
                "beta": analyzed.cell_params[4] if analyzed.cell_params else 90.0,
                "gamma": analyzed.cell_params[5] if analyzed.cell_params else 90.0,
            } if analyzed.cell_params else None,
            "matrix": analyzed.lattice.tolist()
        },
        "M": len(analyzed.G),
        "orbit_sizes": list(analyzed.orbit_sizes),
        "orbit_species": orbit_species,
        "offsets": offsets,
        "cn_shells": {
            "CN1": cn1_list,
            "CN2": cn2_list,
            "CN1_by_orbit": cn1_by_orbit
        },
        "min_distance": analyzed.min_distance,
        "shell_gap": analyzed.shell_gap,
        "is_uniform": analyzed.is_uniform,
        "score": score
    }


# =============================================================================
# MAIN SEARCH RUNNER
# =============================================================================

def run_search_spec(
    spec: SearchSpec,
    top_K: int = 10,
    diagonal_only: bool = False,
    max_per_hnf: int = 30,
    verbose: bool = True
) -> List[Dict]:
    """
    Run lattice search for a single SearchSpec.
    
    This is the main integration function that:
    1. Resolves parent lattices
    2. Converts constraints
    3. Runs search_and_analyze for each (lattice, M) combination
    4. Filters and ranks results
    5. Formats output for anion calculator
    
    Args:
        spec: SearchSpec from orbit_search_generator
        top_K: Maximum number of results to return
        diagonal_only: If True, only search diagonal HNFs (faster)
        max_per_hnf: Max configurations to enumerate per HNF
        verbose: Print progress
    
    Returns:
        List of CationConfig dicts, ranked by score
    """
    # Validate spec
    issues = validate_search_spec(spec)
    if issues:
        raise ValueError(f"Invalid SearchSpec: {issues}")
    
    orbit_sizes = tuple(spec["orbit_sizes"])
    orbit_species = spec["orbit_species"]
    M_list = spec["M_list"]
    
    # Convert constraints
    constraints = convert_constraints(spec["constraints"])
    
    # Build scoring weights from constraint targets
    cn_targets = {}
    for c in spec["constraints"]:
        if c["type"] == "CN1Total":
            cn_targets[c["orbit"]] = c["target"]
    
    weights = ScoringWeights(
        shell_gap_weight=1.0,
        min_distance_penalty=-2.0,
        min_distance_threshold=0.3,
        cn_targets=cn_targets
    )
    
    all_results = []
    
    # Search each parent lattice
    for parent_spec in spec["parent_lattices"]:
        lattice_variants = resolve_parent_lattice(parent_spec)
        
        for lattice, lattice_name, cell_params in lattice_variants:
            if verbose:
                print(f"\nSearching {lattice_name} with M={M_list}...")
            
            # Run search
            configs = search_and_analyze(
                lattice=lattice,
                a=1.0,  # Normalized
                orbit_sizes=orbit_sizes,
                M_list=M_list,
                diagonal_only=diagonal_only,
                max_per_hnf=max_per_hnf,
                verbose=False,
                parent_basis_name=lattice_name,
                cell_params=cell_params
            )
            
            if verbose:
                print(f"  Found {len(configs)} raw configurations")
            
            # Filter by constraints
            filtered = filter_by_constraints(configs, constraints)
            
            if verbose:
                print(f"  After constraints: {len(filtered)}")
            
            # Rank
            ranked = rank_configurations(filtered, weights, top_n=top_K * 2)
            
            # Format and collect
            for analyzed, score in ranked:
                cation_config = format_cation_config(analyzed, orbit_species, score)
                all_results.append((cation_config, score))
    
    # Final ranking across all lattices
    all_results.sort(key=lambda x: -x[1])
    
    return [config for config, score in all_results[:top_K]]


def run_chemistry_search(
    chemistry: Dict,
    top_specs: int = 5,
    top_configs_per_spec: int = 10,
    diagonal_only: bool = False,
    verbose: bool = True
) -> Dict[str, List[Dict]]:
    """
    Complete pipeline from chemistry input to cation configurations.
    
    Args:
        chemistry: Chemistry dict for orbit_search_generator
        top_specs: Number of SearchSpecs to try
        top_configs_per_spec: Number of configs to return per spec
        diagonal_only: If True, only search diagonal HNFs
        verbose: Print progress
    
    Returns:
        Dict mapping spec label to list of CationConfig dicts
    """
    # Validate chemistry
    issues = validate_chemistry(chemistry)
    if issues:
        raise ValueError(f"Invalid chemistry: {issues}")
    
    # Generate search specs
    specs = generate_search_specs(chemistry, top_X=top_specs)
    
    if verbose:
        print(f"Generated {len(specs)} SearchSpecs")
    
    results = {}
    
    for i, spec in enumerate(specs):
        label = spec["label"]
        if verbose:
            print(f"\n{'='*70}")
            print(f"Spec {i+1}/{len(specs)}: {label}")
            print('='*70)
        
        try:
            configs = run_search_spec(
                spec,
                top_K=top_configs_per_spec,
                diagonal_only=diagonal_only,
                verbose=verbose
            )
            results[label] = configs
            
            if verbose:
                print(f"\n  → Found {len(configs)} valid configurations")
        
        except Exception as e:
            if verbose:
                print(f"\n  ⚠ Error: {e}")
            results[label] = []
    
    return results


# =============================================================================
# DEMO
# =============================================================================

def demo():
    """Demonstrate the integration pipeline."""
    
    print("=" * 70)
    print("LATTICE SEARCH INTEGRATION DEMO")
    print("=" * 70)
    
    # Example: SrTiO3 perovskite
    chemistry = {
        "cations": [
            {"element": "Sr", "charge": 2},
            {"element": "Ti", "charge": 4}
        ],
        "anion": {"element": "O", "charge": -2}
    }
    
    print("\nChemistry: SrTiO3 (perovskite)")
    print("-" * 50)
    
    # Generate specs
    specs = generate_search_specs(chemistry, top_X=3)
    
    print(f"\nGenerated {len(specs)} SearchSpecs:")
    for i, spec in enumerate(specs):
        print(f"  {i+1}. {spec['label']}")
    
    # Run search on first spec only (for demo speed)
    if specs:
        spec = specs[0]
        print(f"\n{'='*70}")
        print(f"Running search for: {spec['label']}")
        print("=" * 70)
        
        configs = run_search_spec(
            spec,
            top_K=5,
            diagonal_only=True,  # Fast demo
            verbose=True
        )
        
        print(f"\n{'='*70}")
        print(f"TOP {len(configs)} CATION CONFIGURATIONS")
        print("=" * 70)
        
        for i, config in enumerate(configs):
            print(f"\n--- Config {i+1} ---")
            print(f"  Lattice: {config['lattice']['name']}")
            print(f"  M: {config['M']}")
            print(f"  Orbit sizes: {config['orbit_sizes']}")
            print(f"  Species: {config['orbit_species']}")
            print(f"  CN1: {config['cn_shells']['CN1']}")
            print(f"  CN2: {config['cn_shells']['CN2']}")
            print(f"  Min distance: {config['min_distance']:.4f}")
            print(f"  Uniform: {config['is_uniform']}")
            print(f"  Score: {config['score']:.4f}")
            
            # Show fractional coords
            for offset in config['offsets']:
                orbit = offset['orbit']
                species = offset['species']
                coords = offset['frac_coords']
                print(f"  Orbit {orbit} ({species}): {len(coords)} sites")
                for j, fc in enumerate(coords[:3]):  # Show first 3
                    print(f"    [{fc[0]:.3f}, {fc[1]:.3f}, {fc[2]:.3f}]")
                if len(coords) > 3:
                    print(f"    ... ({len(coords)-3} more)")


if __name__ == "__main__":
    demo()
