# Advanced Lattice Predictor Integration

## Overview

This integration uses a multi-source prediction strategy based on analysis of 1802 AFLOW structures:

1. **Observed patterns** - Empirical frequencies from the database
2. **Template filling** - Lattice-specific position filling rules
3. **N-decomposition** - Patterns from N = (in-plane) × (z-layers) decomposition

Always includes **cubic P/I/F** configurations regardless of predictor output.

## Files

| File | Purpose |
|------|---------|
| `app.py` | Modified app with advanced predictor integration |
| `advanced_predictor_integration.py` | Integration module |
| `sublattice_lookup.json` | Empirical lookup table (N=1-16) |
| `lattice_prediction_strategy.json` | Strategy rules and predictions |
| `offset_filling_database.json` | Filling order statistics by lattice |
| `z_fraction_N_correlation.json` | Z-coordinate analysis |
| `LATTICE_PREDICTION_STRATEGY.md` | Strategy documentation |

## Installation

Copy all files to your app directory:

```bash
cp advanced_predictor_integration.py /path/to/your/app/
cp *.json /path/to/your/app/
```

The predictor will automatically search for JSON files in:
- Current directory
- App directory  
- `/home/claude`
- `/mnt/user-data/outputs`

## Usage

```python
from advanced_predictor_integration import get_predictor

predictor = get_predictor()

# Get search configs for N=4
configs = predictor.get_search_configs(num_metals=4, top_k=15)

for c in configs:
    print(f"{c['id']}: {c['bravais_type']} ({c['source']})")
```

## Prediction Sources

Each config includes a `source` field indicating where it came from:

| Source | Description |
|--------|-------------|
| `advanced_observed` | Empirically observed pattern from AFLOW |
| `advanced_template` | Generated from lattice-specific filling rules |
| `advanced_decomposition` | Generated from N = (in-plane × z-layers) |
| `always_cubic` | Always-included cubic P/I/F configs |

## Example Output for N=4

```
ADV-N4-monoclinic_P-1: monoclinic_P (decomposition 6.0%)
ADV-N4-tetragonal_P-6: tetragonal_P (observed 1.7%)
ADV-N4-cubic_P-9: cubic_P (observed 1.2%)
ADV-N4-hexagonal_P-5: hexagonal_P (template 0.8%)
CUBIC-N4-C1: cubic_P (always)
```

## Key Strategy Insights

1. **N=2**: Two competing patterns dominate
   - z-doubled: `(0,0,0), (0,0,½)` 
   - wurtzite-like: `(0,0,0), (⅓,⅔,½)`

2. **N=4**: High diversity (90% unique patterns)
   - Use lattice type + decomposition
   - Tetragonal checkerboard common: `(0,0,0), (½,½,0), (0,½,0), (½,0,0)`

3. **N=8**: Progressive corner filling
   - Orthorhombic-P with all 8 corners most common (2.1%)

4. **Cubic always fast**: Include P/I/F regardless of probability

## Deployment

For Cloud Run / Docker deployment, ensure the JSON files are copied in your Dockerfile:

```dockerfile
COPY *.json .
COPY advanced_predictor_integration.py .
```
