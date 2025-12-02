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
    get_unique_intersections, calculate_weighted_counts
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
        st.session_state.metals = st.session_state.metals[:num_metals]
        
        # Metal inputs
        st.subheader("Metal Cations")
        for i, metal in enumerate(st.session_state.metals):
            with st.expander(f"Metal {i+1}: {metal['symbol']}⁺{metal['charge']}", expanded=True):
                mcols = st.columns(5)
                with mcols[0]:
                    metal['symbol'] = st.text_input("Symbol", value=metal['symbol'], key=f'm_sym_{i}')
                with mcols[1]:
                    metal['charge'] = st.number_input("Charge", min_value=1, max_value=7, 
                                                      value=metal['charge'], key=f'm_chg_{i}')
                with mcols[2]:
                    metal['ratio'] = st.number_input("Ratio", min_value=1, max_value=20,
                                                     value=metal['ratio'], key=f'm_rat_{i}')
                with mcols[3]:
                    avail_cns = get_available_cns(metal['symbol'], metal['charge'])
                    metal['cn'] = st.number_input("CN", min_value=1, max_value=12,
                                                  value=metal['cn'], key=f'm_cn_{i}')
                    if avail_cns:
                        st.caption(f"Available: {', '.join(map(str, avail_cns))}")
                with mcols[4]:
                    db_radius = get_ionic_radius(metal['symbol'], metal['charge'], metal['cn'])
                    metal['radius'] = st.number_input("Radius (Å)", min_value=0.1, max_value=2.0,
                                                      value=db_radius or metal.get('radius', 0.7),
                                                      step=0.01, key=f'm_rad_{i}')
        
        # Calculate button
        if st.button("🧮 Calculate Stoichiometry & CN", type="primary", use_container_width=True):
            metal_counts, anion_count = calculate_stoichiometry(st.session_state.metals, anion_charge)
            
            # Calculate anion CN
            total_metal_cn = sum(m['cn'] * c for m, c in zip(st.session_state.metals, metal_counts))
            anion_cn = total_metal_cn / anion_count
            
            # Charge balance check
            total_positive = sum(c * m['charge'] for m, c in zip(st.session_state.metals, metal_counts))
            total_negative = anion_count * anion_charge
            
            st.session_state.results = {
                'metal_counts': metal_counts,
                'anion_count': anion_count,
                'anion_cn': anion_cn,
                'charge_balanced': total_positive == total_negative,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'total_metal_cn': total_metal_cn,
            }
            st.session_state.scale_results = {}
        
        # Display results
        if st.session_state.results:
            r = st.session_state.results
            st.markdown("---")
            st.subheader("📊 Results")
            
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
            
            # Detailed tables
            tcols = st.columns(2)
            with tcols[0]:
                st.markdown("**Stoichiometry**")
                stoich_df = pd.DataFrame({
                    'Species': [m['symbol'] for m in st.session_state.metals] + [anion_symbol],
                    'Count': r['metal_counts'] + [r['anion_count']],
                    'Charge': [m['charge'] for m in st.session_state.metals] + [-anion_charge],
                })
                st.dataframe(stoich_df, use_container_width=True, hide_index=True)
            
            with tcols[1]:
                st.markdown("**Coordination Numbers**")
                cn_df = pd.DataFrame({
                    'Species': [m['symbol'] for m in st.session_state.metals] + [anion_symbol],
                    'CN': [m['cn'] for m in st.session_state.metals] + [r['anion_cn']],
                    'Radius (Å)': [m['radius'] for m in st.session_state.metals] + [anion_radius],
                })
                st.dataframe(cn_df, use_container_width=True, hide_index=True)
    
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
        
        # Scale factor calculation section
        st.markdown("---")
        st.header("📐 Minimum Scale Factor Calculator")
        
        if st.session_state.results:
            target_cn = int(round(st.session_state.results['anion_cn']))
            st.info(f"Target CN (from stoichiometry): **{target_cn}**")
            
            # Alpha ratio input
            alpha_cols = st.columns(2)
            with alpha_cols[0]:
                alpha_ratio = st.number_input("α ratio (r = α·s·a)", min_value=0.1, max_value=1.0,
                                              value=0.5, step=0.05, key='alpha_ratio')
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
                                alpha_ratio,
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
                                    'alpha_ratio': alpha_ratio
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
                            alpha_ratio,
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
                    stored_alpha = config_data.get('alpha_ratio', 0.5)
                    
                    sublattice = Sublattice(
                        name='M',
                        offsets=tuple(tuple(o) for o in offsets),
                        alpha_ratio=stored_alpha,
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
            
            # Plot metal atoms
            if len(structure.metal_atoms.cartesian) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=structure.metal_atoms.cartesian[:, 0],
                    y=structure.metal_atoms.cartesian[:, 1],
                    z=structure.metal_atoms.cartesian[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='blue',
                        opacity=0.8
                    ),
                    name='Metal atoms',
                    text=[f"Metal {i}<br>({structure.metal_atoms.fractional[i][0]:.3f}, {structure.metal_atoms.fractional[i][1]:.3f}, {structure.metal_atoms.fractional[i][2]:.3f})"
                          for i in range(len(structure.metal_atoms.fractional))],
                    hoverinfo='text'
                ))
            
            # Plot intersections
            if len(structure.intersections.cartesian) > 0:
                # Color by multiplicity
                mult = structure.intersections.multiplicity
                
                fig_3d.add_trace(go.Scatter3d(
                    x=structure.intersections.cartesian[:, 0],
                    y=structure.intersections.cartesian[:, 1],
                    z=structure.intersections.cartesian[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=mult,
                        colorscale='Viridis',
                        colorbar=dict(title='N'),
                        symbol='diamond',
                        opacity=0.9
                    ),
                    name='Intersections',
                    text=[f"Site {i}<br>N={mult[i]}<br>({structure.intersections.fractional[i][0]:.3f}, {structure.intersections.fractional[i][1]:.3f}, {structure.intersections.fractional[i][2]:.3f})"
                          for i in range(len(structure.intersections.fractional))],
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
    else:
        st.info("👆 Calculate minimum scale factors first to visualize unit cell positions")
    
    # Footer
    st.markdown("---")
    st.caption("Crystal Coordination Calculator • Shannon Ionic Radii (1976) • Sphere Intersection Analysis")


if __name__ == "__main__":
    main()
