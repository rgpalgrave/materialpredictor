"""
Crystal Coordination Calculator & Lattice Explorer
Streamlit application for calculating stoichiometry, anion CN, and minimum scale factors
"""
import streamlit as st
import numpy as np
import pandas as pd
import json
from math import gcd
from functools import reduce
from typing import List, Dict, Optional, Tuple
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ionic_radii import (
    get_ionic_radius, get_available_cns, get_available_charges,
    get_all_cations, get_all_anions, IONIC_RADII
)
from lattice_configs import (
    get_configs_for_n, get_all_lattice_types, get_all_bravais_types,
    LATTICE_COLORS, BRAVAIS_LABELS, LatticeConfig
)
from interstitial_engine import (
    compute_min_scale_for_cn, LatticeParams, Sublattice, find_threshold_s_for_N,
    scan_c_ratio_for_min_scale, batch_scan_c_ratio, lattice_vectors
)
from position_calculator import (
    calculate_complete_structure, generate_metal_positions, calculate_intersections,
    format_position_dict, format_xyz, format_metal_atoms_csv, format_intersections_csv,
    get_unique_intersections, calculate_weighted_counts, analyze_all_coordination_environments,
    find_optimal_half_filling, HalfFillingResult, calculate_stoichiometry_for_config
)

st.set_page_config(
    page_title="Crystal Coordination Calculator",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
.metric-card h3 { margin: 0; font-size: 0.9rem; opacity: 0.9; }
.metric-card h1 { margin: 0.5rem 0 0 0; font-size: 1.8rem; }
.config-card {
    padding: 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border: 2px solid;
}
.success-badge { background: #d4edda; border-color: #28a745; }
.warning-badge { background: #fff3cd; border-color: #ffc107; }
.info-badge { background: #d1ecf1; border-color: #17a2b8; }
</style>
""", unsafe_allow_html=True)


def reduce_ratio(numbers: List[int]) -> List[int]:
    """Reduce a list of integers by their GCD."""
    if not numbers or all(n == 0 for n in numbers):
        return numbers
    g = reduce(gcd, [n for n in numbers if n > 0])
    return [n // g for n in numbers]


def calculate_stoichiometry(metals: List[Dict], anion_charge: int) -> Tuple[List[int], int]:
    """
    Calculate stoichiometry from metal ratios and charges.
    Returns (metal_counts, anion_count) as integers.
    """
    total_positive_charge = sum(m['ratio'] * m['charge'] for m in metals)
    anion_count = total_positive_charge / anion_charge
    metal_counts = [m['ratio'] for m in metals]
    
    # Scale up if anion_count is not integer
    if not anion_count.is_integer():
        # Find LCD
        from fractions import Fraction
        frac = Fraction(total_positive_charge, anion_charge).limit_denominator(100)
        multiplier = frac.denominator
        metal_counts = [c * multiplier for c in metal_counts]
        anion_count = int(frac.numerator)
    else:
        anion_count = int(anion_count)
    
    # Reduce
    all_counts = metal_counts + [anion_count]
    reduced = reduce_ratio(all_counts)
    return reduced[:-1], reduced[-1]


def format_formula(metals: List[Dict], anion_symbol: str, metal_counts: List[int], anion_count: int) -> str:
    """Format chemical formula string."""
    formula = ""
    for m, count in zip(metals, metal_counts):
        formula += m['symbol']
        if count > 1:
            formula += f"₀₁₂₃₄₅₆₇₈₉"[count] if count < 10 else str(count)
    formula += anion_symbol
    if anion_count > 1:
        formula += f"₀₁₂₃₄₅₆₇₈₉"[anion_count] if anion_count < 10 else str(anion_count)
    return formula


def check_stoichiometry_match(
    calculated_metals: Dict[str, float],
    calculated_anions: float,
    expected_metals: Dict[str, float],
    expected_anions: float,
    tolerance: float = 0.1
) -> Tuple[bool, str]:
    """
    Check if calculated stoichiometry matches expected stoichiometry.
    
    Compares ratios rather than absolute values, with tolerance for rounding.
    Also checks for half-filling case (2× anions = common structural motif).
    
    Args:
        calculated_metals: Dict of symbol -> count from position calculation
        calculated_anions: Anion count from position calculation  
        expected_metals: Dict of symbol -> count from charge balance
        expected_anions: Anion count from charge balance
        tolerance: Relative tolerance for ratio comparison
    
    Returns:
        Tuple of (matches, match_type) where match_type is:
        - 'exact': Direct stoichiometry match
        - 'half': Half-filling match (calculated anions = 2× expected)
        - 'none': No match
    """
    # Handle edge cases
    if calculated_anions <= 0 or expected_anions <= 0:
        return False, 'none'
    
    # Check all expected metals are present
    for symbol in expected_metals:
        if symbol not in calculated_metals:
            return False, 'none'
    
    def check_ratios(calc_anions_effective: float) -> bool:
        """Check if metal ratios match given effective anion count."""
        calc_ratios = {}
        for symbol, count in calculated_metals.items():
            calc_ratios[symbol] = count / calc_anions_effective if calc_anions_effective > 0 else 0
        
        exp_ratios = {}
        for symbol, count in expected_metals.items():
            exp_ratios[symbol] = count / expected_anions if expected_anions > 0 else 0
        
        for symbol in expected_metals:
            calc_ratio = calc_ratios.get(symbol, 0)
            exp_ratio = exp_ratios.get(symbol, 0)
            
            if exp_ratio == 0:
                if calc_ratio != 0:
                    return False
            else:
                relative_diff = abs(calc_ratio - exp_ratio) / exp_ratio
                if relative_diff > tolerance:
                    return False
        return True
    
    # Check exact match first
    if check_ratios(calculated_anions):
        return True, 'exact'
    
    # Check half-filling case (calculated has 2× the anions)
    # This is common: zinc blende, fluorite, wurtzite, etc.
    half_anions = calculated_anions / 2.0
    if check_ratios(half_anions):
        return True, 'half'
    
    return False, 'none'


def run_full_analysis_chain(
    metals: List[Dict],
    anion_symbol: str,
    anion_charge: int,
    anion_radius: float,
    progress_callback=None
) -> Dict:
    """
    Run the complete analysis chain:
    1. Calculate stoichiometry and CN
    2. Find minimum scale factors for all configurations
    3. Calculate stoichiometries for successful configs
    4. Check for matches (exact, half-filling)
    5. Calculate regularity for exact matches
    5b. Optimize half-filling and calculate regularity for half-filling matches
    6. Generate 3D previews for all matches
    
    Args:
        metals: List of metal dictionaries with symbol, charge, ratio, cn, radius
        anion_symbol: Anion element symbol
        anion_charge: Anion charge (positive integer)
        anion_radius: Anion ionic radius
        progress_callback: Optional callback(step, total, message) for progress updates
    
    Returns:
        Dictionary with all results including:
        - stoichiometry: Target formula and CN
        - scale_results: Minimum s* for each config
        - matching_configs: Sorted list of exact, half, and non-matching configs
        - regularity_results: Coordination analysis for exact matches
        - half_filling_results: Optimization results for half-filling matches
        - preview_figures: 3D visualizations
    """
    results = {
        'success': False,
        'error': None,
        'stoichiometry': None,
        'scale_results': {},
        'stoich_results': {},
        'matching_configs': [],  # List of configs sorted by match quality
        'regularity_results': {},  # Config ID -> regularity data
        'half_filling_results': {},  # Config ID -> half-filling optimization data
        'preview_figures': {},  # Config ID -> Plotly figure
    }
    
    def update_progress(step, total, message):
        if progress_callback:
            progress_callback(step, total, message)
    
    try:
        num_metals = len(metals)
        
        # Step 1: Calculate stoichiometry
        update_progress(1, 6, "Calculating stoichiometry...")
        
        metal_counts, anion_count = calculate_stoichiometry(metals, anion_charge)
        total_metal_cn = sum(m['cn'] * c for m, c in zip(metals, metal_counts))
        anion_cn = total_metal_cn / anion_count
        target_cn = int(round(anion_cn))
        
        total_positive = sum(c * m['charge'] for m, c in zip(metals, metal_counts))
        total_negative = anion_count * anion_charge
        
        results['stoichiometry'] = {
            'metal_counts': metal_counts,
            'anion_count': anion_count,
            'anion_cn': anion_cn,
            'target_cn': target_cn,
            'charge_balanced': total_positive == total_negative,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'total_metal_cn': total_metal_cn,
            'formula': build_formula_string(metals, metal_counts, anion_symbol, anion_count)
        }
        
        # Step 2: Get configurations and calculate minimum scale factors
        update_progress(2, 6, "Finding minimum scale factors...")
        
        configs = get_configs_for_n(num_metals)
        arity0_configs = configs['arity0']
        
        # Compute coordination radii as (r_metal + r_anion) for each metal
        # This replaces the old alpha_ratio approach
        coord_radii = tuple(m['radius'] + anion_radius for m in metals)
        if len(coord_radii) == 1:
            coord_radii = coord_radii[0]  # Single value for N=1
        
        scale_results = {}
        for config in arity0_configs:
            if config.offsets is None:
                continue
            
            try:
                s_star = compute_min_scale_for_cn(
                    config.offsets,
                    target_cn,
                    config.lattice,
                    coord_radii,  # Now using (r_metal + r_anion) in Å
                    bravais_type=config.bravais_type
                )
                if s_star is not None:
                    scale_results[config.id] = {
                        's_star': s_star,
                        'status': 'found',
                        'lattice': config.lattice,
                        'pattern': config.pattern,
                        'bravais_type': config.bravais_type,
                        'offsets': config.offsets,
                        'coord_radii': coord_radii  # Store coordination radii
                    }
            except Exception:
                pass
        
        results['scale_results'] = scale_results
        
        if not scale_results:
            results['error'] = "No configurations could achieve the target CN"
            return results
        
        # Step 3: Calculate stoichiometries
        update_progress(3, 6, "Calculating stoichiometries...")
        
        stoich_results = {}
        for config_id, config_data in scale_results.items():
            try:
                stoich_result = calculate_stoichiometry_for_config(
                    config_id=config_id,
                    offsets=config_data['offsets'],
                    bravais_type=config_data['bravais_type'],
                    lattice_type=config_data['lattice'],
                    metals=metals,
                    anion_symbol=anion_symbol,
                    scale_s=config_data['s_star'],
                    target_cn=target_cn,
                    anion_radius=anion_radius,  # Pass anion radius for coordination calculation
                    cluster_eps_frac=0.05
                )
                stoich_results[config_id] = stoich_result
            except Exception:
                pass
        
        results['stoich_results'] = stoich_results
        
        # Step 4: Check matches and sort
        update_progress(4, 6, "Checking stoichiometry matches...")
        
        expected_metal_counts = dict(zip([m['symbol'] for m in metals], metal_counts))
        expected_anion_count = anion_count
        
        exact_matches = []
        half_matches = []
        non_matches = []
        
        for config_id, stoich_result in stoich_results.items():
            if not stoich_result.success:
                continue
            
            matches, match_type = check_stoichiometry_match(
                stoich_result.metal_counts,
                stoich_result.anion_count,
                expected_metal_counts,
                expected_anion_count
            )
            
            config_data = scale_results[config_id]
            entry = {
                'config_id': config_id,
                's_star': config_data['s_star'],
                'lattice': config_data['lattice'],
                'bravais_type': config_data['bravais_type'],
                'pattern': config_data.get('pattern', ''),
                'offsets': config_data['offsets'],
                'formula': stoich_result.formula,
                'match_type': match_type,
                'stoich_result': stoich_result
            }
            
            if match_type == 'exact':
                exact_matches.append(entry)
            elif match_type == 'half':
                half_matches.append(entry)
            else:
                non_matches.append(entry)
        
        # Sort each group by s_star
        exact_matches.sort(key=lambda x: x['s_star'])
        half_matches.sort(key=lambda x: x['s_star'])
        non_matches.sort(key=lambda x: x['s_star'])
        
        results['matching_configs'] = exact_matches + half_matches + non_matches
        
        # Step 5: Calculate regularity for exact matches
        update_progress(5, 6, "Analyzing coordination regularity...")
        
        regularity_results = {}
        for entry in exact_matches:
            config_id = entry['config_id']
            config_data = scale_results[config_id]
            
            try:
                # Build structure
                offsets = config_data['offsets']
                sublattice = Sublattice(
                    name='analysis',
                    offsets=tuple(tuple(o) for o in offsets),
                    alpha_ratio=coord_radii,  # Now using (r_metal + r_anion) in Å
                    bravais_type=config_data['bravais_type']
                )
                
                lattice_type = config_data['lattice']
                p_dict = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
                          'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
                if lattice_type == 'Hexagonal':
                    p_dict['gamma'] = 120.0
                    p_dict['c_ratio'] = 1.633
                
                p = LatticeParams(**p_dict)
                
                structure = calculate_complete_structure(
                    sublattices=[sublattice],
                    p=p,
                    scale_s=config_data['s_star'],
                    target_N=target_cn,
                    k_samples=32,
                    cluster_eps_frac=0.05,
                    include_boundary_equivalents=True
                )
                
                # Analyze coordination
                coord_result = analyze_all_coordination_environments(
                    structure=structure,
                    metals=metals,
                    max_sites=target_cn
                )
                
                regularity_results[config_id] = {
                    'structure': structure,
                    'coord_result': coord_result,
                    'mean_regularity': coord_result.summary.get('mean_overall_regularity', 0) if coord_result.success else 0
                }
            except Exception:
                regularity_results[config_id] = {
                    'structure': None,
                    'coord_result': None,
                    'mean_regularity': 0
                }
        
        results['regularity_results'] = regularity_results
        
        # Step 5b: Calculate half-filling optimization for half-filling matches
        half_filling_results = {}
        for entry in half_matches:
            config_id = entry['config_id']
            config_data = scale_results[config_id]
            
            try:
                # Build structure
                offsets = config_data['offsets']
                sublattice = Sublattice(
                    name='analysis',
                    offsets=tuple(tuple(o) for o in offsets),
                    alpha_ratio=coord_radii,
                    bravais_type=config_data['bravais_type']
                )
                
                lattice_type = config_data['lattice']
                p_dict = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
                          'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
                if lattice_type == 'Hexagonal':
                    p_dict['gamma'] = 120.0
                    p_dict['c_ratio'] = 1.633
                
                p = LatticeParams(**p_dict)
                
                structure = calculate_complete_structure(
                    sublattices=[sublattice],
                    p=p,
                    scale_s=config_data['s_star'],
                    target_N=target_cn,
                    k_samples=32,
                    cluster_eps_frac=0.05,
                    include_boundary_equivalents=True
                )
                
                # Run half-filling optimization
                half_result = find_optimal_half_filling(
                    structure=structure,
                    metals=metals,
                    max_coord_sites=target_cn,  # This will use per-metal CN internally
                    target_fraction=0.5
                )
                
                half_filling_results[config_id] = {
                    'structure': structure,
                    'half_result': half_result,
                    'mean_regularity_before': half_result.mean_regularity_before if half_result.success else 0,
                    'mean_regularity_after': half_result.mean_regularity_after if half_result.success else 0,
                    'kept_indices': half_result.kept_site_indices if half_result.success else [],
                    'per_metal_scores': half_result.per_metal_scores if half_result.success else []
                }
            except Exception:
                half_filling_results[config_id] = {
                    'structure': None,
                    'half_result': None,
                    'mean_regularity_before': 0,
                    'mean_regularity_after': 0,
                    'kept_indices': [],
                    'per_metal_scores': []
                }
        
        results['half_filling_results'] = half_filling_results
        
        # Step 6: Generate 3D previews for exact matches
        update_progress(6, 6, "Generating 3D previews...")
        
        preview_figures = {}
        for entry in exact_matches[:10]:  # Limit to first 10 for performance
            config_id = entry['config_id']
            reg_data = regularity_results.get(config_id, {})
            structure = reg_data.get('structure')
            
            if structure is not None:
                try:
                    fig = generate_preview_figure(structure, metals, config_id)
                    preview_figures[config_id] = fig
                except Exception:
                    pass
        
        # Generate previews for half-filling matches (showing kept sites only)
        for entry in half_matches[:10]:
            config_id = entry['config_id']
            hf_data = half_filling_results.get(config_id, {})
            structure = hf_data.get('structure')
            kept_indices = hf_data.get('kept_indices', [])
            
            if structure is not None and kept_indices:
                try:
                    fig = generate_preview_figure_half_filling(
                        structure, metals, kept_indices, config_id
                    )
                    preview_figures[config_id] = fig
                except Exception:
                    pass
        
        results['preview_figures'] = preview_figures
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
    
    return results


def build_formula_string(metals: List[Dict], metal_counts: List[int], 
                         anion_symbol: str, anion_count: int) -> str:
    """Build a chemical formula string."""
    parts = []
    for m, c in zip(metals, metal_counts):
        parts.append(f"{m['symbol']}{c if c > 1 else ''}")
    parts.append(f"{anion_symbol}{anion_count if anion_count > 1 else ''}")
    return ''.join(parts)


def generate_preview_figure(structure, metals: List[Dict], title: str = "") -> go.Figure:
    """Generate a small 3D preview figure for a structure."""
    fig = go.Figure()
    
    lat_vecs = structure.lattice_vectors
    
    # Draw unit cell edges
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    cart_corners = corners @ lat_vecs
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[cart_corners[i, 0], cart_corners[j, 0]],
            y=[cart_corners[i, 1], cart_corners[j, 1]],
            z=[cart_corners[i, 2], cart_corners[j, 2]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    metal_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Plot metal atoms
    if len(structure.metal_atoms.cartesian) > 0:
        unique_offsets = np.unique(structure.metal_atoms.offset_idx)
        
        for offset_idx in unique_offsets:
            mask = structure.metal_atoms.offset_idx == offset_idx
            coords = structure.metal_atoms.cartesian[mask]
            
            symbol = metals[offset_idx]['symbol'] if offset_idx < len(metals) else f'M{offset_idx+1}'
            color = metal_colors[offset_idx % len(metal_colors)]
            
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(size=6, color=color, opacity=0.9),
                name=symbol,
                hoverinfo='name'
            ))
    
    # Plot intersections
    if len(structure.intersections.cartesian) > 0:
        cart = structure.intersections.cartesian
        mult = structure.intersections.multiplicity
        
        fig.add_trace(go.Scatter3d(
            x=cart[:, 0], y=cart[:, 1], z=cart[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='diamond', opacity=0.7),
            name='Anions',
            hoverinfo='name'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            aspectmode='data'
        ),
        height=250,
        width=300,
        margin=dict(l=0, r=0, t=25, b=0),
        title=dict(text=title, font=dict(size=10)),
        showlegend=False
    )
    
    return fig


def generate_preview_figure_half_filling(structure, metals: List[Dict], 
                                         kept_indices: List[int], title: str = "") -> go.Figure:
    """Generate a small 3D preview figure showing only kept anion sites for half-filling."""
    from position_calculator import get_unique_intersections
    
    fig = go.Figure()
    
    lat_vecs = structure.lattice_vectors
    
    # Draw unit cell edges
    corners = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    cart_corners = corners @ lat_vecs
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    for i, j in edges:
        fig.add_trace(go.Scatter3d(
            x=[cart_corners[i, 0], cart_corners[j, 0]],
            y=[cart_corners[i, 1], cart_corners[j, 1]],
            z=[cart_corners[i, 2], cart_corners[j, 2]],
            mode='lines',
            line=dict(color='gray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    metal_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta']
    
    # Plot metal atoms
    if len(structure.metal_atoms.cartesian) > 0:
        unique_offsets = np.unique(structure.metal_atoms.offset_idx)
        
        for offset_idx in unique_offsets:
            mask = structure.metal_atoms.offset_idx == offset_idx
            coords = structure.metal_atoms.cartesian[mask]
            
            symbol = metals[offset_idx]['symbol'] if offset_idx < len(metals) else f'M{offset_idx+1}'
            color = metal_colors[offset_idx % len(metal_colors)]
            
            fig.add_trace(go.Scatter3d(
                x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                mode='markers',
                marker=dict(size=6, color=color, opacity=0.9),
                name=symbol,
                hoverinfo='name'
            ))
    
    # Plot ONLY kept intersection sites
    if len(structure.intersections.cartesian) > 0 and kept_indices:
        # Get unique intersections first
        unique_frac, unique_mult = get_unique_intersections(structure.intersections)
        
        if len(unique_frac) > 0 and len(kept_indices) > 0:
            # Filter to kept indices
            kept_frac = unique_frac[kept_indices]
            kept_cart = kept_frac @ lat_vecs
            
            # Kept sites in red (solid)
            fig.add_trace(go.Scatter3d(
                x=kept_cart[:, 0], y=kept_cart[:, 1], z=kept_cart[:, 2],
                mode='markers',
                marker=dict(size=5, color='red', symbol='diamond', opacity=0.9),
                name='Anions (kept)',
                hoverinfo='name'
            ))
            
            # Removed sites in gray (faded)
            all_indices = set(range(len(unique_frac)))
            removed_indices = list(all_indices - set(kept_indices))
            if removed_indices:
                removed_frac = unique_frac[removed_indices]
                removed_cart = removed_frac @ lat_vecs
                
                fig.add_trace(go.Scatter3d(
                    x=removed_cart[:, 0], y=removed_cart[:, 1], z=removed_cart[:, 2],
                    mode='markers',
                    marker=dict(size=3, color='gray', symbol='diamond', opacity=0.3),
                    name='Anions (removed)',
                    hoverinfo='name'
                ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title=''),
            aspectmode='data'
        ),
        height=250,
        width=300,
        margin=dict(l=0, r=0, t=25, b=0),
        title=dict(text=f"{title} (½)", font=dict(size=10)),
        showlegend=False
    )
    
    return fig


def main():
    st.title("🔬 Crystal Coordination & Lattice Explorer")
    st.markdown("Calculate stoichiometry, anion CN, and find minimum scale factors for target coordination")
    
    # Initialize session state
    if 'metals' not in st.session_state:
        st.session_state.metals = [
            {'symbol': 'Mg', 'charge': 2, 'ratio': 1, 'cn': 6, 'radius': 0.72}
        ]
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'scale_results' not in st.session_state:
        st.session_state.scale_results = {}
    if 'chain_results' not in st.session_state:
        st.session_state.chain_results = None
    if 'stoichiometry_results' not in st.session_state:
        st.session_state.stoichiometry_results = {}
    
    # Sidebar for examples
    with st.sidebar:
        st.header("📚 Quick Examples")
        examples = {
            'MgO (Rocksalt)': {'metals': [{'symbol': 'Mg', 'charge': 2, 'ratio': 1, 'cn': 6, 'radius': 0.72}],
                              'anion': 'O', 'anion_charge': 2},
            'Al₂O₃ (Corundum)': {'metals': [{'symbol': 'Al', 'charge': 3, 'ratio': 2, 'cn': 6, 'radius': 0.535}],
                                 'anion': 'O', 'anion_charge': 2},
            'TiO₂ (Rutile)': {'metals': [{'symbol': 'Ti', 'charge': 4, 'ratio': 1, 'cn': 6, 'radius': 0.605}],
                             'anion': 'O', 'anion_charge': 2},
            'MgAl₂O₄ (Spinel)': {'metals': [
                {'symbol': 'Mg', 'charge': 2, 'ratio': 1, 'cn': 4, 'radius': 0.57},
                {'symbol': 'Al', 'charge': 3, 'ratio': 2, 'cn': 6, 'radius': 0.535}
            ], 'anion': 'O', 'anion_charge': 2},
            'CaTiO₃ (Perovskite)': {'metals': [
                {'symbol': 'Ca', 'charge': 2, 'ratio': 1, 'cn': 12, 'radius': 1.34},
                {'symbol': 'Ti', 'charge': 4, 'ratio': 1, 'cn': 6, 'radius': 0.605}
            ], 'anion': 'O', 'anion_charge': 2},
            'Ca₃Al₂Si₃O₁₂ (Garnet)': {'metals': [
                {'symbol': 'Ca', 'charge': 2, 'ratio': 3, 'cn': 8, 'radius': 1.12},
                {'symbol': 'Al', 'charge': 3, 'ratio': 2, 'cn': 6, 'radius': 0.535},
                {'symbol': 'Si', 'charge': 4, 'ratio': 3, 'cn': 4, 'radius': 0.26}
            ], 'anion': 'O', 'anion_charge': 2},
            'CaF₂ (Fluorite)': {'metals': [{'symbol': 'Ca', 'charge': 2, 'ratio': 1, 'cn': 8, 'radius': 1.12}],
                               'anion': 'F', 'anion_charge': 1},
        }
        
        for name, config in examples.items():
            if st.button(name, use_container_width=True):
                st.session_state.metals = config['metals'].copy()
                st.session_state.anion_symbol = config['anion']
                st.session_state.anion_charge = config['anion_charge']
                st.session_state.results = None
                st.session_state.scale_results = {}
                st.session_state.chain_results = None
                st.session_state.stoichiometry_results = {}
                st.rerun()
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("⚙️ Composition Setup")
        
        # Anion section
        st.subheader("Anion Properties")
        anion_cols = st.columns(3)
        with anion_cols[0]:
            anion_symbol = st.text_input("Symbol", value=st.session_state.get('anion_symbol', 'O'), key='anion_sym')
        with anion_cols[1]:
            anion_charge = st.number_input("Charge", min_value=1, max_value=7, 
                                           value=st.session_state.get('anion_charge', 2), key='anion_chg')
        with anion_cols[2]:
            default_anion_radius = get_ionic_radius(anion_symbol, -anion_charge, 6)
            anion_radius = st.number_input("Radius (Å)", min_value=0.1, max_value=3.0,
                                           value=default_anion_radius or 1.40, step=0.01, key='anion_rad')
        
        # Number of metal types
        num_metals = st.number_input("Number of metal types (N)", min_value=1, max_value=8, 
                                     value=len(st.session_state.metals), key='num_metals')
        
        # Adjust metals list
        while len(st.session_state.metals) < num_metals:
            defaults = [
                {'symbol': 'Mg', 'charge': 2, 'ratio': 1, 'cn': 6, 'radius': 0.72},
                {'symbol': 'Al', 'charge': 3, 'ratio': 2, 'cn': 6, 'radius': 0.535},
                {'symbol': 'Fe', 'charge': 3, 'ratio': 1, 'cn': 6, 'radius': 0.645},
                {'symbol': 'Ti', 'charge': 4, 'ratio': 1, 'cn': 6, 'radius': 0.605},
                {'symbol': 'Ca', 'charge': 2, 'ratio': 1, 'cn': 8, 'radius': 1.12},
                {'symbol': 'Na', 'charge': 1, 'ratio': 1, 'cn': 6, 'radius': 1.02},
                {'symbol': 'K', 'charge': 1, 'ratio': 1, 'cn': 8, 'radius': 1.51},
                {'symbol': 'Zn', 'charge': 2, 'ratio': 1, 'cn': 4, 'radius': 0.60}
            ]
            idx = len(st.session_state.metals)
            st.session_state.metals.append(defaults[idx % len(defaults)])
        
        # Clean up tracking for removed metals
        if len(st.session_state.metals) > num_metals:
            st.session_state.metals = st.session_state.metals[:num_metals]
            # Clean up tracking dict
            if 'metal_last_params' in st.session_state:
                st.session_state.metal_last_params = {
                    k: v for k, v in st.session_state.metal_last_params.items() 
                    if k < num_metals
                }
        
        # Metal inputs
        st.subheader("Metal Cations")
        
        # Initialize tracking for auto-radius updates
        if 'metal_last_params' not in st.session_state:
            st.session_state.metal_last_params = {}
        
        for i, metal in enumerate(st.session_state.metals):
            with st.expander(f"Metal {i+1}: {metal['symbol']}⁺{metal['charge']}", expanded=True):
                mcols = st.columns(5)
                with mcols[0]:
                    new_symbol = st.text_input("Symbol", value=metal['symbol'], key=f'm_sym_{i}')
                with mcols[1]:
                    new_charge = st.number_input("Charge", min_value=1, max_value=7, 
                                                  value=metal['charge'], key=f'm_chg_{i}')
                with mcols[2]:
                    metal['ratio'] = st.number_input("Ratio", min_value=1, max_value=20,
                                                     value=metal['ratio'], key=f'm_rat_{i}')
                with mcols[3]:
                    avail_cns = get_available_cns(new_symbol, new_charge)
                    new_cn = st.number_input("CN", min_value=1, max_value=12,
                                              value=metal['cn'], key=f'm_cn_{i}')
                    if avail_cns:
                        st.caption(f"Available: {', '.join(map(str, avail_cns))}")
                
                # Check if symbol, charge, or CN changed - if so, try to auto-update radius
                last_params = st.session_state.metal_last_params.get(i, {})
                params_changed = (
                    last_params.get('symbol') != new_symbol or
                    last_params.get('charge') != new_charge or
                    last_params.get('cn') != new_cn
                )
                
                # Update metal dict with new values
                metal['symbol'] = new_symbol
                metal['charge'] = new_charge
                metal['cn'] = new_cn
                
                # Determine radius value to show
                current_radius = metal.get('radius', 0.7)
                if params_changed:
                    # Try to get radius from database
                    db_radius = get_ionic_radius(new_symbol, new_charge, new_cn)
                    if db_radius is not None:
                        current_radius = db_radius
                    # Store new params
                    st.session_state.metal_last_params[i] = {
                        'symbol': new_symbol,
                        'charge': new_charge,
                        'cn': new_cn
                    }
                
                with mcols[4]:
                    db_radius = get_ionic_radius(new_symbol, new_charge, new_cn)
                    metal['radius'] = st.number_input(
                        "Radius (Å)", 
                        min_value=0.1, 
                        max_value=2.5,
                        value=current_radius,
                        step=0.01, 
                        key=f'm_rad_{i}',
                        help=f"Database: {db_radius:.3f} Å" if db_radius else "No database value"
                    )
        
        # Calculate button - runs the full analysis chain
        if st.button("🧮 Calculate Stoichiometry & CN", type="primary", use_container_width=True):
            # Create a progress container
            progress_container = st.container()
            
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(step, total, message):
                    progress_bar.progress(step / total)
                    status_text.text(message)
                
                # Run the full analysis chain
                chain_results = run_full_analysis_chain(
                    metals=st.session_state.metals,
                    anion_symbol=anion_symbol,
                    anion_charge=anion_charge,
                    anion_radius=anion_radius,
                    progress_callback=update_progress
                )
                
                progress_bar.empty()
                status_text.empty()
            
            # Store results
            st.session_state.chain_results = chain_results
            
            if chain_results['success'] and chain_results['stoichiometry']:
                st.session_state.results = chain_results['stoichiometry']
                st.session_state.scale_results = chain_results['scale_results']
                st.session_state.stoichiometry_results = chain_results['stoich_results']
            
            st.rerun()
        
        # Display basic stoichiometry results
        if st.session_state.results:
            r = st.session_state.results
            st.markdown("---")
            st.subheader("📊 Stoichiometry Results")
            
            # Formula display
            formula_parts = []
            for m, c in zip(st.session_state.metals, r['metal_counts']):
                formula_parts.append(f"{m['symbol']}{c if c > 1 else ''}")
            formula_parts.append(f"{anion_symbol}{r['anion_count'] if r['anion_count'] > 1 else ''}")
            formula = ''.join(formula_parts)
            
            st.markdown(f"### Formula: **{formula}**")
            
            # Metrics row
            rcols = st.columns(4)
            with rcols[0]:
                st.metric("Anion CN", f"{r['anion_cn']:.2f}")
            with rcols[1]:
                st.metric("Charge Balance", "✓ Balanced" if r['charge_balanced'] else "✗ Imbalanced")
            with rcols[2]:
                st.metric("Total +", f"+{r['total_positive']}")
            with rcols[3]:
                st.metric("Total −", f"−{r['total_negative']}")
    
    with col2:
        st.header("🔷 Lattice Configurations")
        
        # Filters
        fcols = st.columns(2)
        with fcols[0]:
            lattice_filter = st.selectbox("Lattice Type", ['all'] + get_all_lattice_types())
        with fcols[1]:
            arity_filter = st.selectbox("Arity", ['all', '0 (Fixed)', '1 (Parametric)'])
            arity_filter = arity_filter.split()[0] if arity_filter != 'all' else 'all'
        
        configs = get_configs_for_n(num_metals, lattice_filter, arity_filter)
        total_configs = len(configs['arity0']) + len(configs['arity1'])
        
        st.caption(f"Showing {total_configs} configurations for N={num_metals}")
        
        # Display configurations
        config_container = st.container()
        with config_container:
            if configs['arity0']:
                st.markdown("#### Arity 0 — Fixed Offsets")
                for config in configs['arity0']:
                    color = LATTICE_COLORS.get(config.lattice, '#f0f0f0')
                    with st.container():
                        st.markdown(f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #ccc;">
                            <b>{config.id}</b> — {config.lattice}
                            {f'<span style="float: right; font-size: 0.8em;">{config.pattern}</span>' if config.pattern else ''}
                            <br><code style="font-size: 0.85em;">{', '.join(str(o) for o in config.offsets)}</code>
                            <br><small>Params: {config.params}</small>
                        </div>
                        """, unsafe_allow_html=True)
            
            if configs['arity1']:
                st.markdown("#### Arity 1 — Parametric")
                for config in configs['arity1']:
                    color = LATTICE_COLORS.get(config.lattice, '#f0f0f0')
                    st.markdown(f"""
                    <div style="background-color: {color}; padding: 10px; border-radius: 8px; margin-bottom: 8px; border: 1px solid #ccc; opacity: 0.85;">
                        <b>{config.id}</b> — {config.lattice}
                        <span style="float: right; font-family: monospace; font-size: 0.85em;">{config.free_param}</span>
                        <br><small>Params: {config.params}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ============================================
    # UNIFIED RESULTS SECTION
    # ============================================
    if 'chain_results' in st.session_state and st.session_state.chain_results.get('success'):
        chain = st.session_state.chain_results
        
        st.markdown("---")
        st.header("🎯 Matching Lattice Configurations")
        
        matching_configs = chain.get('matching_configs', [])
        regularity_results = chain.get('regularity_results', {})
        preview_figures = chain.get('preview_figures', {})
        
        # Count matches
        exact_matches = [c for c in matching_configs if c['match_type'] == 'exact']
        half_matches = [c for c in matching_configs if c['match_type'] == 'half']
        non_matches = [c for c in matching_configs if c['match_type'] == 'none']
        
        # Summary metrics
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("Total Configurations", len(matching_configs))
        with summary_cols[1]:
            st.metric("Exact Matches ✓", len(exact_matches))
        with summary_cols[2]:
            st.metric("Half-Filling Matches ½", len(half_matches))
        with summary_cols[3]:
            target_cn = chain['stoichiometry'].get('target_cn', 0)
            st.metric("Target CN", target_cn)
        
        # Display exact matches with regularity and 3D preview
        if exact_matches:
            st.subheader("✓ Exact Stoichiometry Matches")
            st.markdown("These configurations produce the expected chemical formula with regular coordination environments.")
            
            for entry in exact_matches:
                config_id = entry['config_id']
                s_star = entry['s_star']
                lattice = entry['lattice']
                formula = entry['formula']
                pattern = entry.get('pattern', '')
                
                # Get regularity data
                reg_data = regularity_results.get(config_id, {})
                mean_reg = reg_data.get('mean_regularity', 0)
                coord_result = reg_data.get('coord_result')
                
                # Create expander for each match
                with st.expander(f"**{config_id}** — {lattice} — s*={s_star:.4f} — Regularity: {mean_reg:.2f}", expanded=True):
                    
                    # Two columns: info + 3D preview
                    info_col, preview_col = st.columns([2, 1])
                    
                    with info_col:
                        # Key metrics
                        met_cols = st.columns(4)
                        with met_cols[0]:
                            st.metric("Scale Factor (s*)", f"{s_star:.4f}")
                        with met_cols[1]:
                            st.metric("Lattice", lattice)
                        with met_cols[2]:
                            st.metric("Formula", formula)
                        with met_cols[3]:
                            st.metric("Regularity", f"{mean_reg:.3f}")
                        
                        if pattern:
                            st.caption(f"Pattern: {pattern}")
                        
                        # Per-metal regularity details
                        if coord_result and coord_result.success:
                            st.markdown("**Coordination Environment Details:**")
                            env_data = []
                            for env in coord_result.environments:
                                env_data.append({
                                    'Metal': env.metal_symbol,
                                    'CN': len(env.coordination_sites),
                                    'Mean Dist (Å)': f"{env.mean_distance:.3f}",
                                    'Dist CV': f"{env.cv_distance:.3f}",
                                    'Ideal Polyhedron': env.ideal_polyhedron.replace('_', ' ').title(),
                                    'Angle Dev (°)': f"{env.angle_deviation:.1f}",
                                    'Regularity': f"{env.overall_regularity:.3f}"
                                })
                            env_df = pd.DataFrame(env_data)
                            st.dataframe(env_df, use_container_width=True, hide_index=True)
                    
                    with preview_col:
                        # 3D preview
                        if config_id in preview_figures:
                            st.plotly_chart(preview_figures[config_id], use_container_width=True)
                        else:
                            st.info("Preview not available")
        
        # Display half-filling matches with full details
        if half_matches:
            st.subheader("½ Half-Filling Matches")
            st.markdown("""
            These configurations match when only **half** the anion sites are occupied 
            (e.g., zinc blende from fluorite, wurtzite, anti-fluorite structures).
            The optimization finds which sites to remove for maximum coordination regularity.
            """)
            
            half_filling_results = chain_results.get('half_filling_results', {})
            
            for entry in half_matches:
                config_id = entry['config_id']
                s_star = entry['s_star']
                lattice = entry['lattice']
                formula = entry['formula']
                pattern = entry.get('pattern', '')
                
                # Get half-filling optimization data
                hf_data = half_filling_results.get(config_id, {})
                reg_before = hf_data.get('mean_regularity_before', 0)
                reg_after = hf_data.get('mean_regularity_after', 0)
                per_metal = hf_data.get('per_metal_scores', [])
                
                improvement = reg_after - reg_before
                delta_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
                
                # Create expander for each half-filling match
                with st.expander(f"**{config_id}** — {lattice} — s*={s_star:.4f} — Regularity: {reg_after:.2f} (after ½)", expanded=True):
                    
                    # Two columns: info + 3D preview
                    info_col, preview_col = st.columns([2, 1])
                    
                    with info_col:
                        # Key metrics - 5 columns for half-filling
                        met_cols = st.columns(5)
                        with met_cols[0]:
                            st.metric("Scale Factor (s*)", f"{s_star:.4f}")
                        with met_cols[1]:
                            st.metric("Lattice", lattice)
                        with met_cols[2]:
                            # Show half-filled formula
                            # Parse formula to halve anion count
                            half_formula = formula.replace('₂', '₁').replace('₄', '₂').replace('₆', '₃').replace('₈', '₄')
                            st.metric("Formula (½)", half_formula)
                        with met_cols[3]:
                            st.metric("Reg. Before", f"{reg_before:.3f}")
                        with met_cols[4]:
                            st.metric("Reg. After", f"{reg_after:.3f}", delta=delta_str)
                        
                        if pattern:
                            st.caption(f"Pattern: {pattern}")
                        
                        # Per-metal coordination details
                        if per_metal:
                            st.markdown("**Per-Metal Coordination (after half-filling):**")
                            metal_data = []
                            for m in per_metal:
                                metal_data.append({
                                    'Metal': m['symbol'],
                                    'CN': m['cn'],
                                    'Regularity': f"{m['regularity']:.3f}"
                                })
                            metal_df = pd.DataFrame(metal_data)
                            st.dataframe(metal_df, use_container_width=True, hide_index=True)
                        
                        st.info("💡 Half-filling optimization selects which anion sites to remove for maximum coordination regularity.")
                    
                    with preview_col:
                        # 3D preview (shows kept sites in red, removed in gray)
                        if config_id in preview_figures:
                            st.plotly_chart(preview_figures[config_id], use_container_width=True)
                        else:
                            st.info("Preview not available")
        
        # Display non-matches (collapsed)
        if non_matches:
            with st.expander(f"Other Configurations ({len(non_matches)} configs - no stoichiometry match)"):
                other_data = []
                for entry in non_matches[:20]:  # Limit display
                    other_data.append({
                        'Configuration': entry['config_id'],
                        's*': f"{entry['s_star']:.4f}",
                        'Lattice': entry['lattice'],
                        'Formula': entry['formula']
                    })
                other_df = pd.DataFrame(other_data)
                st.dataframe(other_df, use_container_width=True, hide_index=True)
                
                if len(non_matches) > 20:
                    st.caption(f"Showing first 20 of {len(non_matches)} configurations")
    
    # Keep the old detailed sections below for advanced users
    # but hide them in expanders
    with st.expander("🔧 Advanced: Manual Scale Factor & c/a Optimization", expanded=False):
        # Scale factor calculation section
        st.markdown("---")
        st.header("📐 Minimum Scale Factor Calculator")
        
        if st.session_state.results:
            target_cn = int(round(st.session_state.results['anion_cn']))
            st.info(f"Target CN (from stoichiometry): **{target_cn}**")
            
            # Compute per-metal alpha ratios from radii
            # Alpha ratio: r = alpha * s * a, so alpha = r / (s * a)
            # We normalize so the average alpha is around the user's base value
            metals = st.session_state.metals
            metal_radii = [m['radius'] for m in metals]
            
            # Get anion radius from session state or use default
            anion_rad = st.session_state.get('anion_rad', 1.40)
            
            # Coordination radius input (scale factor for r_metal + r_anion)
            alpha_cols = st.columns(2)
            with alpha_cols[0]:
                # Compute coordination radii as (r_metal + r_anion)
                coord_radii_list = [r + anion_rad for r in metal_radii]
                st.markdown(f"**Coordination radii (r_M + r_X):**")
                for m, cr in zip(metals, coord_radii_list):
                    st.caption(f"  {m['symbol']}: {m['radius']:.3f} + {anion_rad:.3f} = {cr:.3f} Å")
            
            # Create coord_radii tuple
            if len(coord_radii_list) > 1:
                coord_radii = tuple(coord_radii_list)
            else:
                coord_radii = coord_radii_list[0]
            
            with alpha_cols[1]:
                st.markdown("")
                st.markdown("")
                if st.button("🔍 Find Minimum Scale Factors", type="primary", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    arity0_configs = configs['arity0']
                    results = {}
                    
                    for i, config in enumerate(arity0_configs):
                        status_text.text(f"Computing {config.id}...")
                        progress_bar.progress((i + 1) / len(arity0_configs))
                        
                        if config.offsets is None:
                            results[config.id] = {'s_star': None, 'status': 'parametric'}
                            continue
                        
                        try:
                            s_star = compute_min_scale_for_cn(
                                config.offsets,
                                target_cn,
                                config.lattice,
                                coord_radii,  # Using (r_metal + r_anion) in Å
                                bravais_type=config.bravais_type
                            )
                            if s_star is not None:
                                results[config.id] = {
                                    's_star': s_star,
                                    'status': 'found',
                                    'lattice': config.lattice,
                                    'pattern': config.pattern,
                                    'bravais_type': config.bravais_type,
                                    'offsets': config.offsets,
                                    'coord_radii': coord_radii
                                }
                            else:
                                results[config.id] = {'s_star': None, 'status': 'not_achievable'}
                        except Exception as e:
                            results[config.id] = {'s_star': None, 'status': f'error: {str(e)}'}
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.session_state.scale_results = results
            
            # Display scale results
            if st.session_state.scale_results:
                st.subheader(f"Results for CN = {target_cn}")
                
                # Sort by s_star
                sorted_results = sorted(
                    [(k, v) for k, v in st.session_state.scale_results.items() if v.get('s_star') is not None],
                    key=lambda x: x[1]['s_star']
                )
                
                if sorted_results:
                    # Best result highlight
                    best_id, best_data = sorted_results[0]
                    st.success(f"**Best configuration: {best_id}** — s* = {best_data['s_star']:.4f}")
                    
                    # Results table
                    df_data = []
                    for config_id, data in sorted_results:
                        bravais = data.get('bravais_type', '')
                        bravais_label = BRAVAIS_LABELS.get(bravais, bravais)
                        df_data.append({
                            'Configuration': config_id,
                            's*': f"{data['s_star']:.4f}",
                            'Lattice': data.get('lattice', ''),
                            'Bravais': bravais_label,
                            'Pattern': data.get('pattern', '')
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Plot
                    if len(sorted_results) > 1:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=[r[0] for r in sorted_results],
                            y=[r[1]['s_star'] for r in sorted_results],
                            marker_color=[LATTICE_COLORS.get(r[1].get('lattice', 'Cubic'), '#6366f1') 
                                         for r in sorted_results],
                            text=[f"{r[1]['s_star']:.3f}" for r in sorted_results],
                            textposition='outside'
                        ))
                        fig.update_layout(
                            title=f"Minimum Scale Factors for CN={target_cn}",
                            xaxis_title="Configuration",
                            yaxis_title="s*",
                            height=400,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Show failures
                failures = [(k, v) for k, v in st.session_state.scale_results.items() 
                           if v.get('s_star') is None and v.get('status') != 'parametric']
                if failures:
                    with st.expander(f"⚠️ {len(failures)} configurations could not achieve CN={target_cn}"):
                        for config_id, data in failures:
                            st.text(f"  • {config_id}: {data.get('status', 'unknown')}")
                
                # Stoichiometry Calculation Section
                st.markdown("---")
                st.subheader("🧮 Calculate Stoichiometries")
                st.markdown("Calculate the chemical formula for each successful configuration based on weighted atom counts.")
                
                # Initialize stoichiometry results in session state
                if 'stoichiometry_results' not in st.session_state:
                    st.session_state.stoichiometry_results = {}
                
                if st.button("🧮 Calculate Stoichiometries for All Configs", type="secondary"):
                    from position_calculator import calculate_stoichiometry_for_config
                    
                    # Get successful configs
                    successful_configs = [(k, v) for k, v in st.session_state.scale_results.items() 
                                         if v.get('s_star') is not None]
                    
                    if successful_configs:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        stoich_results = {}
                        metals = st.session_state.metals
                        anion_symbol = st.session_state.get('anion_symbol', 'O')
                        anion_rad = st.session_state.get('anion_rad', 1.40)
                        
                        for i, (config_id, config_data) in enumerate(successful_configs):
                            status_text.text(f"Calculating stoichiometry for {config_id}...")
                            progress_bar.progress((i + 1) / len(successful_configs))
                            
                            result = calculate_stoichiometry_for_config(
                                config_id=config_id,
                                offsets=config_data['offsets'],
                                bravais_type=config_data['bravais_type'],
                                lattice_type=config_data['lattice'],
                                metals=metals,
                                anion_symbol=anion_symbol,
                                scale_s=config_data['s_star'],
                                target_cn=target_cn,
                                anion_radius=anion_rad,
                                cluster_eps_frac=0.05
                            )
                            stoich_results[config_id] = result
                        
                        progress_bar.empty()
                        status_text.empty()
                        st.session_state.stoichiometry_results = stoich_results
                    else:
                        st.warning("No successful configurations to calculate stoichiometry for.")
                
                # Display stoichiometry results
                if st.session_state.stoichiometry_results:
                    st.markdown("#### Stoichiometry Results")
                    
                    # Get expected stoichiometry from initial calculation
                    expected_metal_counts = dict(zip(
                        [m['symbol'] for m in st.session_state.metals],
                        st.session_state.results['metal_counts']
                    ))
                    expected_anion_count = st.session_state.results['anion_count']
                    
                    # Filter option
                    filter_col1, filter_col2 = st.columns([1, 3])
                    with filter_col1:
                        filter_matching = st.checkbox(
                            "Show only matching stoichiometry",
                            value=False,
                            help="Filter to show only configurations that match the expected formula"
                        )
                    with filter_col2:
                        # Show expected formula
                        expected_parts = [f"{sym}{cnt if cnt > 1 else ''}" 
                                         for sym, cnt in expected_metal_counts.items()]
                        expected_parts.append(f"{st.session_state.get('anion_symbol', 'O')}{expected_anion_count if expected_anion_count > 1 else ''}")
                        st.caption(f"Expected formula: **{''.join(expected_parts)}**")
                    
                    # Build results table
                    stoich_data = []
                    matching_count = 0
                    
                    for config_id, result in st.session_state.stoichiometry_results.items():
                        if result.success:
                            # Get s* from scale results
                            s_star = st.session_state.scale_results.get(config_id, {}).get('s_star', 0)
                            
                            # Check if stoichiometry matches expected
                            matches, match_type = check_stoichiometry_match(
                                result.metal_counts, 
                                result.anion_count,
                                expected_metal_counts,
                                expected_anion_count
                            )
                            
                            if matches:
                                matching_count += 1
                            
                            # Skip if filtering and doesn't match
                            if filter_matching and not matches:
                                continue
                            
                            # Build metal counts string
                            metal_str = ', '.join(f"{sym}={cnt:.1f}" for sym, cnt in result.metal_counts.items())
                            
                            # Format match indicator with type
                            if match_type == 'exact':
                                match_str = '✓'
                            elif match_type == 'half':
                                match_str = '½'  # Half-filling indicator
                            else:
                                match_str = '✗'
                            
                            stoich_data.append({
                                'Config': config_id,
                                'Formula': result.formula,
                                'Match': match_str,
                                's*': f"{s_star:.4f}",
                                'Metals': metal_str,
                                'Anions': f"{result.anion_count:.1f}",
                                'Ratio': result.ratio_formula
                            })
                        else:
                            if not filter_matching:  # Don't show errors when filtering
                                stoich_data.append({
                                    'Config': config_id,
                                    'Formula': f"Error: {result.error}",
                                    'Match': '',
                                    's*': '',
                                    'Metals': '',
                                    'Anions': '',
                                    'Ratio': ''
                                })
                    
                    # Count match types for summary
                    exact_count = sum(1 for r in stoich_data if r.get('Match') == '✓')
                    half_count = sum(1 for r in stoich_data if r.get('Match') == '½')
                    
                    # Show match summary
                    total_successful = sum(1 for r in st.session_state.stoichiometry_results.values() if r.success)
                    if matching_count > 0:
                        summary_parts = []
                        if exact_count > 0:
                            summary_parts.append(f"{exact_count} exact (✓)")
                        if half_count > 0:
                            summary_parts.append(f"{half_count} half-filling (½)")
                        st.success(f"**{matching_count}** of **{total_successful}** configurations match: {', '.join(summary_parts)}")
                        if half_count > 0:
                            st.caption("½ = Half-filling: only half the anion sites are occupied (e.g., zinc blende, fluorite)")
                    else:
                        st.warning(f"No configurations match the expected stoichiometry (0 of {total_successful})")
                    
                    if stoich_data:
                        df_stoich = pd.DataFrame(stoich_data)
                        st.dataframe(df_stoich, use_container_width=True, hide_index=True)
                        
                        # Summary of unique formulas
                        successful_formulas = [r['Formula'] for r in stoich_data if not r['Formula'].startswith('Error')]
                        if successful_formulas:
                            unique_formulas = list(set(successful_formulas))
                            st.info(f"**Unique formulas found:** {', '.join(unique_formulas)}")
            
            # c/a Ratio Scanning Section
            st.markdown("---")
            st.header("📊 c/a Ratio Optimization")
            st.markdown("Scan c/a ratios to find optimal parameters for tetragonal, hexagonal, and orthorhombic lattices.")
            
            # Initialize c/a scan results in session state
            if 'ca_scan_results' not in st.session_state:
                st.session_state.ca_scan_results = {}
            
            # c/a scan settings - row 1
            ca_cols = st.columns(4)
            with ca_cols[0]:
                c_ratio_min = st.number_input("c/a min", min_value=0.1, max_value=1.5, 
                                              value=0.5, step=0.1, key='ca_min')
            with ca_cols[1]:
                c_ratio_max = st.number_input("c/a max", min_value=0.5, max_value=3.0,
                                              value=2.0, step=0.1, key='ca_max')
            with ca_cols[2]:
                scan_level = st.selectbox("Scan Level", 
                                         ['coarse', 'medium', 'fine', 'ultrafine'],
                                         index=2, key='scan_level')
            with ca_cols[3]:
                optimize_metric = st.selectbox("Optimize For",
                                              ['s³/Volume', 's* only'],
                                              index=0, key='optimize_metric',
                                              help="s³/V ∝ packing fraction (minimizes atom volume per unit cell); s* only finds minimum scale factor")
            
            # Convert UI selection to engine parameter
            metric_param = 's3_over_volume' if optimize_metric == 's³/Volume' else 's_star'
            
            # Scan button
            scannable_lattices = {'Tetragonal', 'Hexagonal', 'Orthorhombic'}
            scannable_configs = [c for c in configs['arity0'] 
                                if c.lattice in scannable_lattices and c.offsets is not None]
            
            scan_disabled = len(scannable_configs) == 0
            if st.button("🔬 Scan c/a Ratios", type="primary", use_container_width=True, 
                        disabled=scan_disabled):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                ca_results = {}
                
                for i, config in enumerate(scannable_configs):
                    status_text.text(f"Scanning {config.id}...")
                    progress_bar.progress((i + 1) / len(scannable_configs))
                    
                    try:
                        scan_result = scan_c_ratio_for_min_scale(
                            config.offsets,
                            target_cn,
                            config.lattice,
                            coord_radii,  # Use coordination radii (r_metal + r_anion)
                            bravais_type=config.bravais_type,
                            c_ratio_min=c_ratio_min,
                            c_ratio_max=c_ratio_max,
                            scan_level=scan_level,
                            optimize_metric=metric_param
                        )
                        ca_results[config.id] = {
                            **scan_result,
                            'lattice': config.lattice,
                            'bravais_type': config.bravais_type,
                            'pattern': config.pattern
                        }
                    except Exception as e:
                        ca_results[config.id] = {
                            'best_c_ratio': None,
                            'best_s_star': None,
                            'best_volume': None,
                            'best_metric': None,
                            'error': str(e)
                        }
                
                progress_bar.empty()
                status_text.empty()
                st.session_state.ca_scan_results = ca_results
            
            if scan_disabled:
                st.caption("No tetragonal, hexagonal, or orthorhombic configurations available for c/a scanning.")
            
            # Display c/a scan results
            if st.session_state.ca_scan_results:
                st.subheader(f"c/a Scan Results for CN = {target_cn}")
                
                # Sort by best metric
                sorted_ca_results = sorted(
                    [(k, v) for k, v in st.session_state.ca_scan_results.items() 
                     if v.get('best_metric') is not None],
                    key=lambda x: x[1]['best_metric']
                )
                
                if sorted_ca_results:
                    # Best result
                    best_id, best_data = sorted_ca_results[0]
                    opt_metric = best_data.get('optimize_metric', 's3_over_volume')
                    if opt_metric == 's3_over_volume':
                        st.success(f"**Best: {best_id}** — c/a = {best_data['best_c_ratio']:.4f}, "
                                  f"s* = {best_data['best_s_star']:.4f}, "
                                  f"V = {best_data['best_volume']:.2f}, "
                                  f"s³/V = {best_data['best_metric']:.6f}")
                    else:
                        st.success(f"**Best: {best_id}** — c/a = {best_data['best_c_ratio']:.4f}, "
                                  f"s* = {best_data['best_s_star']:.4f}")
                    
                    # Results table
                    ca_df_data = []
                    for config_id, data in sorted_ca_results:
                        bravais = data.get('bravais_type', '')
                        bravais_label = BRAVAIS_LABELS.get(bravais, bravais)
                        row = {
                            'Configuration': config_id,
                            'Optimal c/a': f"{data['best_c_ratio']:.4f}",
                            's*': f"{data['best_s_star']:.4f}",
                            'Volume': f"{data['best_volume']:.2f}",
                            'Lattice': data.get('lattice', ''),
                            'Bravais': bravais_label
                        }
                        if data.get('optimize_metric') == 's3_over_volume':
                            row['s³/V'] = f"{data['best_metric']:.6f}"
                        ca_df_data.append(row)
                    
                    ca_df = pd.DataFrame(ca_df_data)
                    st.dataframe(ca_df, use_container_width=True, hide_index=True)
                    
                    # Detailed view for selected configuration
                    st.markdown("#### Detailed Scan Results")
                    selected_config = st.selectbox(
                        "Select configuration to view scan details:",
                        [r[0] for r in sorted_ca_results],
                        key='ca_detail_select'
                    )
                    
                    if selected_config:
                        detail_data = st.session_state.ca_scan_results[selected_config]
                        scan_results = detail_data.get('scan_results', [])
                        opt_metric = detail_data.get('optimize_metric', 's_over_volume')
                        
                        if scan_results:
                            # Create scan plot - now with 4 values per tuple
                            valid_scans = [(c, s, v, m) for c, s, v, m in scan_results if m is not None]
                            if valid_scans:
                                c_vals = [x[0] for x in valid_scans]
                                s_vals = [x[1] for x in valid_scans]
                                v_vals = [x[2] for x in valid_scans]
                                m_vals = [x[3] for x in valid_scans]
                                
                                # Plot metric vs c/a
                                fig_scan = go.Figure()
                                
                                y_vals = m_vals if opt_metric == 's3_over_volume' else s_vals
                                y_label = "s³/Volume" if opt_metric == 's3_over_volume' else "s*"
                                
                                fig_scan.add_trace(go.Scatter(
                                    x=c_vals,
                                    y=y_vals,
                                    mode='markers+lines',
                                    marker=dict(size=8, color='#667eea'),
                                    line=dict(width=1, color='#667eea', dash='dot'),
                                    name='Scan points',
                                    hovertemplate='c/a: %{x:.3f}<br>' + y_label + ': %{y:.4f}<extra></extra>'
                                ))
                                
                                # Mark the optimum
                                if detail_data.get('best_c_ratio') and detail_data.get('best_metric'):
                                    best_y = detail_data['best_metric'] if opt_metric == 's_over_volume' else detail_data['best_s_star']
                                    fig_scan.add_trace(go.Scatter(
                                        x=[detail_data['best_c_ratio']],
                                        y=[best_y],
                                        mode='markers',
                                        marker=dict(size=15, color='#22c55e', symbol='star'),
                                        name=f"Optimum (c/a={detail_data['best_c_ratio']:.3f})"
                                    ))
                                
                                fig_scan.update_layout(
                                    title=f"{y_label} vs c/a for {selected_config}",
                                    xaxis_title="c/a ratio",
                                    yaxis_title=y_label,
                                    height=350,
                                    showlegend=True
                                )
                                st.plotly_chart(fig_scan, use_container_width=True)
                            
                            # Show scan history
                            scan_history = detail_data.get('scan_history', {})
                            if scan_history:
                                with st.expander("View scan history by level"):
                                    for level, level_results in scan_history.items():
                                        valid_level = [(c, s, v, m) for c, s, v, m in level_results if m is not None]
                                        if valid_level:
                                            st.markdown(f"**{level.capitalize()}** ({len(valid_level)} points)")
                                            level_df = pd.DataFrame(valid_level, columns=['c/a', 's*', 'Volume', 'Metric'])
                                            level_df['c/a'] = level_df['c/a'].apply(lambda x: f"{x:.4f}")
                                            level_df['s*'] = level_df['s*'].apply(lambda x: f"{x:.4f}")
                                            level_df['Volume'] = level_df['Volume'].apply(lambda x: f"{x:.2f}")
                                            level_df['Metric'] = level_df['Metric'].apply(lambda x: f"{x:.6f}")
                                            st.dataframe(level_df, use_container_width=True, hide_index=True)
                
                # Show failures
                ca_failures = [(k, v) for k, v in st.session_state.ca_scan_results.items() 
                              if v.get('best_metric') is None]
                if ca_failures:
                    with st.expander(f"⚠️ {len(ca_failures)} configurations could not achieve CN={target_cn}"):
                        for config_id, data in ca_failures:
                            error = data.get('error', 'not achievable in c/a range')
                            st.text(f"  • {config_id}: {error}")
        
            # ============================================
            # STOICHIOMETRY-BASED c/a SCANNING
            # ============================================
            st.markdown("---")
            st.subheader("🎯 c/a Scan for Target Stoichiometry")
            st.markdown("""
            Scan c/a ratios to find regions where the calculated stoichiometry matches the target formula.
            This searches for c/a values where the M/X ratio (metals/anions) equals the expected value.
            """)
        
            # Initialize stoichiometry scan results
            if 'stoich_ca_scan_results' not in st.session_state:
                st.session_state.stoich_ca_scan_results = {}
        
            # Calculate target M/X ratio from results
            total_metals = sum(st.session_state.results['metal_counts'])
            total_anions = st.session_state.results['anion_count']
            target_mx = total_metals / total_anions if total_anions > 0 else 1.0
        
            st.info(f"**Target M/X ratio:** {target_mx:.4f} (from {total_metals} metals : {total_anions} anions)")
        
            # Scan settings
            stoich_cols = st.columns(4)
            with stoich_cols[0]:
                stoich_c_min = st.number_input("c/a min", min_value=0.1, max_value=1.5, 
                                               value=0.5, step=0.1, key='stoich_ca_min')
                with stoich_cols[1]:
                    stoich_c_max = st.number_input("c/a max", min_value=0.5, max_value=3.0,
                                                   value=2.0, step=0.1, key='stoich_ca_max')
                with stoich_cols[2]:
                    stoich_n_points = st.number_input("Scan points", min_value=20, max_value=200,
                                                      value=50, step=10, key='stoich_n_points')
                with stoich_cols[3]:
                    stoich_tolerance = st.number_input("M/X tolerance", min_value=0.01, max_value=0.5,
                                                       value=0.1, step=0.01, key='stoich_tolerance',
                                                       help="Relative tolerance for M/X match (0.1 = 10%)")
            
                # Check half-filling option
                check_half = st.checkbox("Check half-filling", value=True, key='stoich_check_half',
                                        help="Also check if half the anion sites gives correct stoichiometry")
            
                # Scan button
                scannable_configs = [c for c in configs['arity0'] 
                                    if c.lattice in {'Tetragonal', 'Hexagonal', 'Orthorhombic'} 
                                    and c.offsets is not None]
            
                if st.button("🎯 Scan c/a for Stoichiometry", type="secondary", 
                            disabled=len(scannable_configs) == 0):
                    from position_calculator import scan_ca_for_stoichiometry
                
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                    stoich_ca_results = {}
                    metals = st.session_state.metals
                
                    for i, config in enumerate(scannable_configs):
                        status_text.text(f"Scanning {config.id} for stoichiometry...")
                        progress_bar.progress((i + 1) / len(scannable_configs))
                    
                        result = scan_ca_for_stoichiometry(
                            config_id=config.id,
                            offsets=config.offsets,
                            bravais_type=config.bravais_type,
                            lattice_type=config.lattice,
                            metals=metals,
                            target_mx_ratio=target_mx,
                            target_cn=target_cn,
                            anion_radius=st.session_state.get('anion_rad', 1.40),
                            c_ratio_min=stoich_c_min,
                            c_ratio_max=stoich_c_max,
                            n_points=stoich_n_points,
                            tolerance=stoich_tolerance,
                            check_half_filling=check_half
                        )
                        stoich_ca_results[config.id] = result
                
                    progress_bar.empty()
                    status_text.empty()
                    st.session_state.stoich_ca_scan_results = stoich_ca_results
            
                # Display results
                if st.session_state.stoich_ca_scan_results:
                    st.markdown("#### Stoichiometry Scan Results")
                
                    # Sort by best M/X error
                    sorted_results = sorted(
                        [(k, v) for k, v in st.session_state.stoich_ca_scan_results.items() if v.success],
                        key=lambda x: x[1].best_mx_error if x[1].best_mx_error is not None else float('inf')
                    )
                
                    if sorted_results:
                        # Count matches
                        matches = [(k, v) for k, v in sorted_results if v.best_mx_error is not None and v.best_mx_error <= stoich_tolerance]
                    
                        if matches:
                            st.success(f"**{len(matches)}** configurations have c/a ranges with matching stoichiometry!")
                        else:
                            st.warning("No configurations found with matching stoichiometry in the scanned range.")
                    
                        # Results table
                        stoich_df_data = []
                        for config_id, result in sorted_results:
                            match_str = "✓" if result.best_mx_error is not None and result.best_mx_error <= stoich_tolerance else "✗"
                            ranges_str = ", ".join([f"{r[0]:.3f}-{r[1]:.3f}" for r in result.matching_ranges]) if result.matching_ranges else "—"
                        
                            stoich_df_data.append({
                                'Config': config_id,
                                'Match': match_str,
                                'Best c/a': f"{result.best_c_ratio:.4f}" if result.best_c_ratio else "—",
                                's*': f"{result.best_s_star:.4f}" if result.best_s_star else "—",
                                'M/X': f"{result.best_mx_ratio:.4f}" if result.best_mx_ratio else "—",
                                'Error': f"{result.best_mx_error:.4f}" if result.best_mx_error is not None else "—",
                                'Matching c/a ranges': ranges_str
                            })
                    
                        stoich_df = pd.DataFrame(stoich_df_data)
                        st.dataframe(stoich_df, use_container_width=True, hide_index=True)
                    
                        # Detailed view
                        st.markdown("#### Scan Details")
                        selected_stoich_config = st.selectbox(
                            "Select configuration to view scan plot:",
                            [r[0] for r in sorted_results],
                            key='stoich_detail_select'
                        )
                    
                        if selected_stoich_config:
                            result = st.session_state.stoich_ca_scan_results[selected_stoich_config]
                        
                            # Filter valid scan data
                            valid_data = [(c, s, mx, err) for c, s, mx, err in result.scan_data 
                                         if mx is not None and err is not None]
                        
                            if valid_data:
                                c_vals = [x[0] for x in valid_data]
                                mx_vals = [x[2] for x in valid_data]
                                err_vals = [x[3] for x in valid_data]
                            
                                # Create plot
                                fig_stoich = go.Figure()
                            
                                # M/X ratio vs c/a
                                fig_stoich.add_trace(go.Scatter(
                                    x=c_vals,
                                    y=mx_vals,
                                    mode='lines+markers',
                                    name='M/X ratio',
                                    line=dict(color='blue'),
                                    marker=dict(size=4)
                                ))
                            
                                # Target line
                                fig_stoich.add_hline(
                                    y=target_mx, 
                                    line_dash="dash", 
                                    line_color="green",
                                    annotation_text=f"Target M/X = {target_mx:.4f}"
                                )
                            
                                # Tolerance band
                                fig_stoich.add_hrect(
                                    y0=target_mx * (1 - stoich_tolerance),
                                    y1=target_mx * (1 + stoich_tolerance),
                                    fillcolor="green",
                                    opacity=0.1,
                                    line_width=0
                                )
                            
                                # Highlight matching ranges
                                for r_start, r_end in result.matching_ranges:
                                    fig_stoich.add_vrect(
                                        x0=r_start,
                                        x1=r_end,
                                        fillcolor="green",
                                        opacity=0.2,
                                        line_width=0
                                    )
                            
                                fig_stoich.update_layout(
                                    title=f"M/X Ratio vs c/a for {selected_stoich_config}",
                                    xaxis_title="c/a ratio",
                                    yaxis_title="M/X ratio",
                                    height=400
                                )
                            
                                st.plotly_chart(fig_stoich, use_container_width=True)
                            
                                # Show matching ranges info
                                if result.matching_ranges:
                                    st.success(f"Matching c/a ranges: {', '.join([f'{r[0]:.3f} - {r[1]:.3f}' for r in result.matching_ranges])}")
        else:
            st.info("👆 Calculate stoichiometry first to determine target CN")
    
    # ============================================
    # UNIT CELL VIEWER SECTION
    # ============================================
    st.header("🔮 Unit Cell Viewer")
    st.markdown("""
    Visualize metal ion and interstitial site positions in one unit cell.  
    Sites on boundaries (e.g., at x=0) also appear at the opposite face (x=1) to show translational symmetry.
    """)
    
    # Get available results from scale factor calculations
    available_configs = []
    if 'scale_results' in st.session_state and st.session_state.scale_results:
        for config_id, data in st.session_state.scale_results.items():
            if data.get('s_star') is not None:
                available_configs.append({
                    'id': config_id,
                    's_star': data['s_star'],
                    'lattice': data.get('lattice', ''),
                    'bravais': data.get('bravais_type', ''),
                    'offsets': data.get('offsets', [])
                })
    
    if available_configs:
        uc_cols = st.columns([2, 1, 1, 1])
        
        with uc_cols[0]:
            config_options = [f"{c['id']} (s*={c['s_star']:.4f})" for c in available_configs]
            selected_config_idx = st.selectbox(
                "Select configuration to visualize",
                range(len(config_options)),
                format_func=lambda i: config_options[i],
                key='uc_config_select'
            )
            selected_config = available_configs[selected_config_idx]
            
            # Update scale factor when config changes
            if 'uc_last_config' not in st.session_state or st.session_state.uc_last_config != selected_config['id']:
                st.session_state.uc_last_config = selected_config['id']
                st.session_state.uc_scale_s = float(selected_config['s_star'])
                st.rerun()
        
        with uc_cols[1]:
            uc_scale_s = st.number_input(
                "Scale factor s",
                min_value=0.1,
                max_value=2.0,
                value=float(selected_config['s_star']),
                step=0.01,
                key='uc_scale_s'
            )
        
        with uc_cols[2]:
            # Get target CN from results if available
            default_cn = 4
            if 'results' in st.session_state and st.session_state.results:
                default_cn = int(round(st.session_state.results.get('anion_cn', 4)))
            
            uc_target_n = st.number_input(
                "Min. multiplicity",
                min_value=2,
                max_value=12,
                value=default_cn,
                step=1,
                key='uc_target_n'
            )
        
        with uc_cols[3]:
            show_boundary_equiv = st.checkbox(
                "Show boundary equivalents",
                value=True,
                help="Show equivalent positions at unit cell boundaries",
                key='uc_boundary'
            )
        
        # Second row for clustering control
        uc_cols2 = st.columns([2, 2])
        with uc_cols2[0]:
            cluster_eps = st.select_slider(
                "Merge tolerance (fractional)",
                options=[0.01, 0.02, 0.03, 0.05, 0.075, 0.10],
                value=0.05,
                help="Sites within this distance (in fractional coords) are merged. Larger = fewer sites.",
                key='uc_cluster_eps'
            )
        
        if st.button("🔮 Generate Unit Cell View", type="primary", key='uc_generate'):
            with st.spinner("Calculating positions..."):
                # Build sublattice from selected config
                config_data = None
                for config_id, data in st.session_state.scale_results.items():
                    if config_id == selected_config['id']:
                        config_data = data
                        break
                
                if config_data:
                    # Create sublattice
                    offsets = config_data.get('offsets', [(0, 0, 0)])
                    bravais = config_data.get('bravais_type', 'cubic_P')
                    
                    # Get coordination radii (r_metal + r_anion) - with fallback for backward compatibility
                    stored_coord_radii = config_data.get('coord_radii')
                    if stored_coord_radii is None:
                        # Fallback: compute from metal radii + anion radius
                        metals = st.session_state.get('metals', [{'symbol': 'M', 'radius': 0.7}])
                        anion_rad = st.session_state.get('anion_rad', 1.40)
                        metal_radii = [m['radius'] for m in metals]
                        if len(metal_radii) > 1:
                            stored_coord_radii = tuple(r + anion_rad for r in metal_radii)
                        else:
                            stored_coord_radii = metal_radii[0] + anion_rad
                    
                    sublattice = Sublattice(
                        name='M',
                        offsets=tuple(tuple(o) for o in offsets),
                        alpha_ratio=stored_coord_radii,  # Now using coordination radii
                        bravais_type=bravais
                    )
                    
                    # Determine lattice type and set parameters
                    lattice_type = config_data.get('lattice', 'Cubic')
                    p_dict = {'a': 5.0, 'b_ratio': 1.0, 'c_ratio': 1.0,
                              'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0}
                    
                    if lattice_type == 'Hexagonal':
                        p_dict['gamma'] = 120.0
                        p_dict['c_ratio'] = 1.633  # Ideal HCP
                    
                    # Check if we have c/a scan results for this config
                    if 'ca_scan_results' in st.session_state and selected_config['id'] in st.session_state.ca_scan_results:
                        ca_result = st.session_state.ca_scan_results[selected_config['id']]
                        if ca_result.get('best_c_ratio') is not None:
                            p_dict['c_ratio'] = ca_result['best_c_ratio']
                    
                    p = LatticeParams(**p_dict)
                    
                    # Calculate structure
                    structure = calculate_complete_structure(
                        sublattices=[sublattice],
                        p=p,
                        scale_s=uc_scale_s,
                        target_N=uc_target_n,
                        k_samples=32,
                        cluster_eps_frac=cluster_eps,
                        include_boundary_equivalents=show_boundary_equiv
                    )
                    
                    st.session_state['uc_structure'] = structure
                    st.session_state['uc_lattice_params'] = p
        
        # Display results if available
        if 'uc_structure' in st.session_state:
            structure = st.session_state['uc_structure']
            p = st.session_state['uc_lattice_params']
            
            # Calculate weighted counts for correct stoichiometry
            weighted = calculate_weighted_counts(structure)
            unique_frac, unique_mult = get_unique_intersections(structure.intersections)
            
            uc_metric_cols = st.columns(4)
            with uc_metric_cols[0]:
                metal_count = weighted['metal_count']
                # Display as integer if whole number, else 1 decimal
                if abs(metal_count - round(metal_count)) < 0.01:
                    st.metric("Metal atoms/cell", f"{int(round(metal_count))}")
                else:
                    st.metric("Metal atoms/cell", f"{metal_count:.2f}")
            with uc_metric_cols[1]:
                site_count = weighted['intersection_count']
                if abs(site_count - round(site_count)) < 0.01:
                    st.metric("Interstitial sites/cell", f"{int(round(site_count))}")
                else:
                    st.metric("Interstitial sites/cell", f"{site_count:.2f}")
            with uc_metric_cols[2]:
                st.metric("Unique sites", len(unique_frac))
            with uc_metric_cols[3]:
                if len(structure.intersections.multiplicity) > 0:
                    st.metric("Max multiplicity", int(np.max(structure.intersections.multiplicity)))
                else:
                    st.metric("Max multiplicity", 0)
            
            # 3D visualization
            st.subheader("3D Unit Cell")
            
            # Half-filling options (for zinc blende, etc.)
            viz_cols = st.columns([1, 1, 1, 1])
            with viz_cols[0]:
                show_half_filling = st.checkbox(
                    "Half-filling mode",
                    value=False,
                    help="Keep only half the anion sites, optimized for maximum coordination regularity",
                    key='uc_half_filling'
                )
            
            with viz_cols[1]:
                if show_half_filling:
                    half_fill_fraction = st.select_slider(
                        "Fraction to keep",
                        options=[0.25, 0.33, 0.5, 0.67, 0.75],
                        value=0.5,
                        key='uc_half_fraction'
                    )
                else:
                    half_fill_fraction = 0.5
            
            # Get expected CN from stoichiometry calculation
            expected_cn = 6  # Default
            if 'results' in st.session_state and st.session_state.results:
                expected_cn = int(round(st.session_state.results.get('anion_cn', 6)))
            
            # Calculate optimized half-filling if enabled
            half_filling_result = None
            if show_half_filling:
                with st.spinner("Optimizing site selection for maximum regularity..."):
                    metals = st.session_state.get('metals', [{'symbol': 'M'}])
                    half_filling_result = find_optimal_half_filling(
                        structure=structure,
                        metals=metals,
                        max_coord_sites=expected_cn,  # Use expected CN from stoichiometry
                        target_fraction=half_fill_fraction
                    )
                    st.session_state['half_filling_result'] = half_filling_result
                
                if half_filling_result and half_filling_result.success:
                    # Show improvement metrics
                    improvement = half_filling_result.mean_regularity_after - half_filling_result.mean_regularity_before
                    delta_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
                    
                    hf_metric_cols = st.columns(5)
                    with hf_metric_cols[0]:
                        st.metric("Sites kept", f"{half_filling_result.kept_count}/{half_filling_result.original_count}")
                    with hf_metric_cols[1]:
                        st.metric("Target CN", expected_cn)
                    with hf_metric_cols[2]:
                        st.metric("Regularity (before)", f"{half_filling_result.mean_regularity_before:.3f}")
                    with hf_metric_cols[3]:
                        st.metric("Regularity (after)", f"{half_filling_result.mean_regularity_after:.3f}", delta=delta_str)
                    with hf_metric_cols[4]:
                        # Show per-metal CNs
                        cn_strs = [f"{m['symbol']}:{m['cn']}" for m in half_filling_result.per_metal_scores]
                        st.metric("CNs", ", ".join(cn_strs))
                    
                    # Show per-metal details in expander
                    with st.expander("Per-metal regularity details"):
                        for m in half_filling_result.per_metal_scores:
                            st.write(f"**{m['symbol']}**: CN = {m['cn']}, Regularity = {m['regularity']:.3f}")
            
            # Madelung Energy Calculation
            with st.expander("⚡ Madelung Energy Calculation", expanded=False):
                st.markdown("""
                Calculate approximate electrostatic (Madelung) energy using ionic charges.
                For accurate results, enter the experimental lattice parameter.
                """)
                
                mad_cols = st.columns([1, 1, 1])
                with mad_cols[0]:
                    # Default lattice param from current structure
                    default_a = p.a if hasattr(p, 'a') else 5.0
                    mad_lattice_a = st.number_input(
                        "Lattice parameter a (Å)",
                        min_value=2.0, max_value=20.0,
                        value=default_a, step=0.01,
                        help="Use experimental value for accurate Madelung constant",
                        key='mad_lattice_a'
                    )
                
                with mad_cols[1]:
                    # Structure type for scale factor calculation
                    struct_types = ['rocksalt', 'fluorite', 'zincblende', 'perovskite', 'rutile', 'custom']
                    mad_struct_type = st.selectbox(
                        "Structure type",
                        struct_types,
                        index=0,
                        help="Determines expected anion positions",
                        key='mad_struct_type'
                    )
                
                with mad_cols[2]:
                    mad_supercell = st.selectbox(
                        "Supercell size",
                        [5, 7, 9, 11],
                        index=1,
                        help="Larger = more accurate but slower",
                        key='mad_supercell'
                    )
                
                if st.button("⚡ Calculate Madelung Energy", key='calc_madelung'):
                    with st.spinner("Computing electrostatic energy..."):
                        from position_calculator import (
                            calculate_madelung_energy, 
                            compute_physical_scale_factor
                        )
                        
                        # Get metal and anion info
                        metals = st.session_state.get('metals', [{'symbol': 'M', 'charge': 2, 'radius': 0.7}])
                        anion_charge = st.session_state.get('anion_charge', -2)
                        anion_rad = st.session_state.get('anion_rad', 1.40)
                        
                        # Compute coordination radius
                        metal_radii = [m['radius'] for m in metals]
                        coord_radius = metal_radii[0] + anion_rad  # Use first metal for scale
                        
                        # Compute physical scale factor
                        if mad_struct_type != 'custom':
                            s_physical = compute_physical_scale_factor(
                                mad_lattice_a, coord_radius, mad_struct_type
                            )
                        else:
                            s_physical = uc_scale_s  # Use current scale
                        
                        # Rebuild structure with physical lattice parameter and scale
                        from interstitial_engine import Sublattice, LatticeParams
                        
                        # Get config data again
                        config_data = None
                        for cid, data in st.session_state.scale_results.items():
                            if cid == selected_config['id']:
                                config_data = data
                                break
                        
                        if config_data:
                            offsets = config_data.get('offsets', [(0, 0, 0)])
                            bravais = config_data.get('bravais_type', 'cubic_P')
                            
                            if len(metal_radii) > 1:
                                coord_radii = tuple(r + anion_rad for r in metal_radii)
                            else:
                                coord_radii = coord_radius
                            
                            sublattice_mad = Sublattice(
                                name='M',
                                offsets=tuple(tuple(o) for o in offsets),
                                alpha_ratio=coord_radii,
                                bravais_type=bravais
                            )
                            
                            # Create lattice params matching original but with user's a
                            lattice_type = config_data.get('lattice', 'Cubic')
                            p_mad = LatticeParams(
                                a=mad_lattice_a,
                                b_ratio=p.b_ratio,
                                c_ratio=p.c_ratio,
                                alpha=p.alpha,
                                beta=p.beta,
                                gamma=p.gamma
                            )
                            
                            structure_mad = calculate_complete_structure(
                                sublattices=[sublattice_mad],
                                p=p_mad,
                                scale_s=s_physical,
                                target_N=expected_cn,
                                k_samples=32,
                                cluster_eps_frac=0.05,
                                include_boundary_equivalents=True
                            )
                            
                            # Calculate Madelung energy
                            madelung_result = calculate_madelung_energy(
                                structure=structure_mad,
                                metals=metals,
                                anion_charge=anion_charge,
                                target_multiplicity=expected_cn,
                                supercell_size=mad_supercell
                            )
                            
                            if madelung_result.success:
                                st.success("Madelung calculation complete!")
                                
                                mad_metric_cols = st.columns(4)
                                with mad_metric_cols[0]:
                                    st.metric("Madelung Constant", f"{madelung_result.madelung_constant:.4f}")
                                with mad_metric_cols[1]:
                                    st.metric("Energy/formula", f"{madelung_result.energy_per_formula:.2f} eV")
                                with mad_metric_cols[2]:
                                    st.metric("Nearest M-X", f"{madelung_result.nearest_neighbor_dist:.3f} Å")
                                with mad_metric_cols[3]:
                                    st.metric("Formula units", f"{madelung_result.formula_units:.0f}")
                                
                                # Additional info
                                st.caption(f"Calculation used: {madelung_result.n_cations} cations, {madelung_result.n_anions} anions, s = {s_physical:.4f}")
                                
                                # Reference values
                                st.info("""
                                **Reference Madelung constants:**
                                - Rocksalt (NaCl, MgO): 1.748
                                - Fluorite (CaF₂): 2.519
                                - Zinc blende (ZnS): 1.638
                                - Wurtzite: 1.641
                                - Cesium chloride: 1.763
                                """)
                            else:
                                st.error(f"Calculation failed: {madelung_result.error}")
                        else:
                            st.error("Could not retrieve configuration data")
            
            # Create 3D plot
            fig_3d = go.Figure()
            
            # Get lattice vectors for drawing unit cell
            lat_vecs = structure.lattice_vectors
            
            # Draw unit cell edges
            corners = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
            ])
            cart_corners = corners @ lat_vecs
            
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
                (4, 5), (5, 6), (6, 7), (7, 4),  # Top
                (0, 4), (1, 5), (2, 6), (3, 7)   # Verticals
            ]
            
            for i, j in edges:
                fig_3d.add_trace(go.Scatter3d(
                    x=[cart_corners[i, 0], cart_corners[j, 0]],
                    y=[cart_corners[i, 1], cart_corners[j, 1]],
                    z=[cart_corners[i, 2], cart_corners[j, 2]],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Color palette for different metals
            metal_colors = ['blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink']
            
            # Plot metal atoms - grouped by offset_idx (which corresponds to metal type)
            if len(structure.metal_atoms.cartesian) > 0:
                # Get unique offset indices
                unique_offsets = np.unique(structure.metal_atoms.offset_idx)
                metals = st.session_state.get('metals', [{'symbol': 'M'}])
                
                for offset_idx in unique_offsets:
                    mask = structure.metal_atoms.offset_idx == offset_idx
                    coords = structure.metal_atoms.cartesian[mask]
                    frac_coords = structure.metal_atoms.fractional[mask]
                    
                    # Get metal symbol for this offset
                    if offset_idx < len(metals):
                        symbol = metals[offset_idx]['symbol']
                    else:
                        symbol = f"M{offset_idx+1}"
                    
                    color = metal_colors[offset_idx % len(metal_colors)]
                    
                    fig_3d.add_trace(go.Scatter3d(
                        x=coords[:, 0],
                        y=coords[:, 1],
                        z=coords[:, 2],
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=color,
                            opacity=0.8
                        ),
                        name=f'{symbol} atoms',
                        text=[f"{symbol}<br>({frac_coords[i][0]:.3f}, {frac_coords[i][1]:.3f}, {frac_coords[i][2]:.3f})"
                              for i in range(len(frac_coords))],
                        hoverinfo='text'
                    ))
            
            # Plot intersections
            if len(structure.intersections.cartesian) > 0:
                # Color by multiplicity
                mult = structure.intersections.multiplicity
                frac = structure.intersections.fractional
                cart = structure.intersections.cartesian
                
                # Apply optimized half-filling filter if enabled
                if show_half_filling and half_filling_result and half_filling_result.success:
                    # Get the kept site fractional coordinates
                    kept_fracs = half_filling_result.kept_site_fractions
                    
                    # Match intersection sites to kept sites (accounting for boundary equivalents)
                    keep_mask = np.zeros(len(frac), dtype=bool)
                    
                    for i in range(len(frac)):
                        site_frac = frac[i]
                        # Wrap to [0, 1) for comparison
                        site_wrapped = site_frac - np.floor(site_frac)
                        
                        for kept_frac in kept_fracs:
                            kept_wrapped = kept_frac - np.floor(kept_frac)
                            # Check if this is the same site (considering PBC)
                            diff = site_wrapped - kept_wrapped
                            diff = diff - np.round(diff)
                            if np.sum(diff**2) < 0.001:  # Tolerance
                                keep_mask[i] = True
                                break
                    
                    frac = frac[keep_mask]
                    cart = cart[keep_mask]
                    mult = mult[keep_mask]
                
                fig_3d.add_trace(go.Scatter3d(
                    x=cart[:, 0],
                    y=cart[:, 1],
                    z=cart[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=mult,
                        colorscale='Viridis',
                        colorbar=dict(title='N'),
                        symbol='diamond',
                        opacity=0.9
                    ),
                    name='Anion sites',
                    text=[f"Site {i}<br>N={mult[i]}<br>({frac[i][0]:.3f}, {frac[i][1]:.3f}, {frac[i][2]:.3f})"
                          for i in range(len(frac))],
                    hoverinfo='text'
                ))
            
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='x (Å)',
                    yaxis_title='y (Å)',
                    zaxis_title='z (Å)',
                    aspectmode='data'
                ),
                height=500,
                margin=dict(l=0, r=0, t=30, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Position tables
            tab_metals, tab_intersections = st.tabs(["Metal Positions", "Intersection Sites"])
            
            with tab_metals:
                if len(structure.metal_atoms.fractional) > 0:
                    metal_df = pd.DataFrame({
                        'Index': range(len(structure.metal_atoms.fractional)),
                        'Frac x': [f"{x:.4f}" for x in structure.metal_atoms.fractional[:, 0]],
                        'Frac y': [f"{x:.4f}" for x in structure.metal_atoms.fractional[:, 1]],
                        'Frac z': [f"{x:.4f}" for x in structure.metal_atoms.fractional[:, 2]],
                        'Cart x': [f"{x:.4f}" for x in structure.metal_atoms.cartesian[:, 0]],
                        'Cart y': [f"{x:.4f}" for x in structure.metal_atoms.cartesian[:, 1]],
                        'Cart z': [f"{x:.4f}" for x in structure.metal_atoms.cartesian[:, 2]],
                        'Radius (Å)': [f"{x:.4f}" for x in structure.metal_atoms.radius]
                    })
                    
                    # Add metal symbol column
                    metals = st.session_state.get('metals', [])
                    symbols = []
                    for idx in structure.metal_atoms.offset_idx:
                        if idx < len(metals):
                            symbols.append(metals[idx]['symbol'])
                        else:
                            symbols.append(f"M{idx+1}")
                    metal_df.insert(1, 'Symbol', symbols)
                    
                    st.dataframe(metal_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No metal atoms to display")
            
            with tab_intersections:
                if len(structure.intersections.fractional) > 0:
                    int_df = pd.DataFrame({
                        'Index': range(len(structure.intersections.fractional)),
                        'N': structure.intersections.multiplicity,
                        'Frac x': [f"{x:.4f}" for x in structure.intersections.fractional[:, 0]],
                        'Frac y': [f"{x:.4f}" for x in structure.intersections.fractional[:, 1]],
                        'Frac z': [f"{x:.4f}" for x in structure.intersections.fractional[:, 2]],
                        'Cart x': [f"{x:.4f}" for x in structure.intersections.cartesian[:, 0]],
                        'Cart y': [f"{x:.4f}" for x in structure.intersections.cartesian[:, 1]],
                        'Cart z': [f"{x:.4f}" for x in structure.intersections.cartesian[:, 2]],
                    })
                    st.dataframe(int_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No intersections found at this scale factor")
            
            # Export options
            st.subheader("📥 Export Data")
            export_cols = st.columns(4)
            
            with export_cols[0]:
                json_data = json.dumps(format_position_dict(structure), indent=2)
                st.download_button(
                    label="📄 JSON",
                    data=json_data,
                    file_name="unit_cell_positions.json",
                    mime="application/json"
                )
            
            with export_cols[1]:
                csv_metals = format_metal_atoms_csv(structure)
                st.download_button(
                    label="📊 Metals CSV",
                    data=csv_metals,
                    file_name="metal_positions.csv",
                    mime="text/csv"
                )
            
            with export_cols[2]:
                csv_intersections = format_intersections_csv(structure)
                st.download_button(
                    label="📊 Intersections CSV",
                    data=csv_intersections,
                    file_name="intersection_positions.csv",
                    mime="text/csv"
                )
            
            with export_cols[3]:
                xyz_data = format_xyz(structure, include_intersections=True)
                st.download_button(
                    label="🔬 XYZ File",
                    data=xyz_data,
                    file_name="structure.xyz",
                    mime="text/plain"
                )
            
            # ============================================
            # COORDINATION ENVIRONMENT ANALYSIS
            # ============================================
            st.markdown("---")
            st.subheader("🎯 Coordination Environment Analysis")
            st.markdown("""
            Analyze the regularity of coordination environments around each metal type.
            For each metal, finds the nearest intersection sites using periodic boundary conditions
            and calculates distance and angular metrics to assess polyhedron regularity.
            """)
            
            coord_cols = st.columns([1, 1, 2])
            with coord_cols[0]:
                # Get expected CN from stoichiometry as default
                default_coord_cn = 6
                if 'results' in st.session_state and st.session_state.results:
                    default_coord_cn = int(round(st.session_state.results.get('anion_cn', 6)))
                
                max_coord_sites = st.number_input(
                    "Max coordination sites",
                    min_value=4,
                    max_value=12,
                    value=min(12, max(4, default_coord_cn)),  # Use expected CN, clamped to 4-12
                    step=1,
                    help=f"Maximum nearest sites to consider (expected CN from stoichiometry: {default_coord_cn})",
                    key='coord_max_sites'
                )
            
            if st.button("🔍 Analyze Coordination Environments", type="secondary", key='coord_analyze'):
                with st.spinner("Analyzing coordination environments..."):
                    metals = st.session_state.get('metals', [{'symbol': 'M'}])
                    
                    coord_result = analyze_all_coordination_environments(
                        structure=structure,
                        metals=metals,
                        max_sites=max_coord_sites
                    )
                    
                    st.session_state['coord_analysis'] = coord_result
            
            # Display coordination analysis results
            if 'coord_analysis' in st.session_state:
                coord_result = st.session_state['coord_analysis']
                
                if coord_result.success and coord_result.environments:
                    # Overall summary
                    st.success(f"**Overall structure regularity: {coord_result.summary['mean_overall_regularity']:.2f}** (1.0 = ideal)")
                    
                    # Results for each metal type
                    for env in coord_result.environments:
                        with st.expander(f"**{env.metal_symbol}** — CN={len(env.coordination_sites)}, Regularity={env.overall_regularity:.2f}", expanded=True):
                            
                            # Metrics row
                            env_cols = st.columns(4)
                            with env_cols[0]:
                                st.metric("Coordination #", len(env.coordination_sites))
                            with env_cols[1]:
                                st.metric("Ideal Polyhedron", env.ideal_polyhedron.replace('_', ' ').title())
                            with env_cols[2]:
                                st.metric("Distance Regularity", f"{env.distance_regularity:.2f}")
                            with env_cols[3]:
                                st.metric("Angular Regularity", f"{env.angular_regularity:.2f}")
                            
                            # Distance statistics
                            st.markdown("**Distance Statistics**")
                            dist_cols = st.columns(5)
                            with dist_cols[0]:
                                st.metric("Mean", f"{env.mean_distance:.3f} Å")
                            with dist_cols[1]:
                                st.metric("Std Dev", f"{env.std_distance:.3f} Å")
                            with dist_cols[2]:
                                st.metric("Min", f"{env.min_distance:.3f} Å")
                            with dist_cols[3]:
                                st.metric("Max", f"{env.max_distance:.3f} Å")
                            with dist_cols[4]:
                                st.metric("CV", f"{env.cv_distance:.3f}")
                            
                            # Angular statistics
                            if len(env.angles) > 0:
                                st.markdown("**Angular Statistics**")
                                ang_cols = st.columns(4)
                                with ang_cols[0]:
                                    st.metric("Mean Angle", f"{env.mean_angle:.1f}°")
                                with ang_cols[1]:
                                    st.metric("Std Dev", f"{env.std_angle:.1f}°")
                                with ang_cols[2]:
                                    ideal_str = ", ".join([f"{a:.1f}°" for a in env.ideal_angles])
                                    st.metric("Ideal Angles", ideal_str)
                                with ang_cols[3]:
                                    st.metric("RMS Deviation", f"{env.angle_deviation:.1f}°")
                            
                            # Coordination sites table
                            if env.coordination_sites:
                                st.markdown("**Coordination Sites**")
                                site_data = []
                                for i, site in enumerate(env.coordination_sites):
                                    img_str = f"({site.image[0]},{site.image[1]},{site.image[2]})"
                                    site_data.append({
                                        '#': i + 1,
                                        'Distance (Å)': f"{site.distance:.4f}",
                                        'N': site.multiplicity,
                                        'Frac x': f"{site.fractional[0]:.4f}",
                                        'Frac y': f"{site.fractional[1]:.4f}",
                                        'Frac z': f"{site.fractional[2]:.4f}",
                                        'Image': img_str
                                    })
                                site_df = pd.DataFrame(site_data)
                                st.dataframe(site_df, use_container_width=True, hide_index=True)
                            
                            # 3D visualization of coordination environment
                            st.markdown("**3D Coordination Environment**")
                            
                            fig_coord = go.Figure()
                            
                            # Plot metal center
                            fig_coord.add_trace(go.Scatter3d(
                                x=[env.metal_cartesian[0]],
                                y=[env.metal_cartesian[1]],
                                z=[env.metal_cartesian[2]],
                                mode='markers',
                                marker=dict(size=12, color='blue', symbol='circle'),
                                name=f'{env.metal_symbol} center',
                                hovertext=f"{env.metal_symbol}<br>({env.metal_fractional[0]:.3f}, {env.metal_fractional[1]:.3f}, {env.metal_fractional[2]:.3f})"
                            ))
                            
                            # Plot coordination sites
                            if env.coordination_sites:
                                coord_x = [s.cartesian[0] for s in env.coordination_sites]
                                coord_y = [s.cartesian[1] for s in env.coordination_sites]
                                coord_z = [s.cartesian[2] for s in env.coordination_sites]
                                coord_dist = [s.distance for s in env.coordination_sites]
                                
                                fig_coord.add_trace(go.Scatter3d(
                                    x=coord_x,
                                    y=coord_y,
                                    z=coord_z,
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=coord_dist,
                                        colorscale='Viridis',
                                        colorbar=dict(title='Dist (Å)', x=1.02),
                                        symbol='diamond'
                                    ),
                                    name='Coordination sites',
                                    hovertext=[f"Site {i+1}<br>d={s.distance:.3f} Å<br>N={s.multiplicity}" 
                                              for i, s in enumerate(env.coordination_sites)]
                                ))
                                
                                # Draw lines from metal to each coordination site
                                for site in env.coordination_sites:
                                    fig_coord.add_trace(go.Scatter3d(
                                        x=[env.metal_cartesian[0], site.cartesian[0]],
                                        y=[env.metal_cartesian[1], site.cartesian[1]],
                                        z=[env.metal_cartesian[2], site.cartesian[2]],
                                        mode='lines',
                                        line=dict(color='rgba(100,100,100,0.4)', width=2),
                                        showlegend=False,
                                        hoverinfo='skip'
                                    ))
                            
                            fig_coord.update_layout(
                                scene=dict(
                                    xaxis_title='x (Å)',
                                    yaxis_title='y (Å)',
                                    zaxis_title='z (Å)',
                                    aspectmode='data'
                                ),
                                height=400,
                                margin=dict(l=0, r=0, t=30, b=0),
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_coord, use_container_width=True)
                    
                    # Summary table
                    st.markdown("---")
                    st.markdown("**Summary Table**")
                    summary_data = []
                    for env_summary in coord_result.summary['environments']:
                        summary_data.append({
                            'Metal': env_summary['symbol'],
                            'CN': env_summary['cn'],
                            'Mean Dist (Å)': f"{env_summary['mean_distance']:.3f}",
                            'CV': f"{env_summary['cv_distance']:.3f}",
                            'Ideal Polyhedron': env_summary['ideal_polyhedron'].replace('_', ' ').title(),
                            'Angle Dev (°)': f"{env_summary['angle_deviation']:.1f}",
                            'Dist. Reg.': f"{env_summary['distance_regularity']:.2f}",
                            'Ang. Reg.': f"{env_summary['angular_regularity']:.2f}",
                            'Overall': f"{env_summary['overall_regularity']:.2f}"
                        })
                    summary_df = pd.DataFrame(summary_data)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                elif coord_result.error:
                    st.error(f"Analysis failed: {coord_result.error}")
                else:
                    st.warning("No coordination environments could be analyzed.")
    else:
        st.info("👆 Calculate minimum scale factors first to visualize unit cell positions")
    
    # Footer
    st.markdown("---")
    st.caption("Crystal Coordination Calculator • Shannon Ionic Radii (1976) • Sphere Intersection Analysis")


if __name__ == "__main__":
    main()
