# Crystal Coordination Calculator - Chemistry-Based Predictor Integration

## Overview

This update integrates a **chemistry-based lattice prediction system** into the Crystal Coordination Calculator. Instead of relying solely on empirical patterns from the AFLOW database, the app now uses **Pauling radius-ratio rules** and **bond allocation models** to predict likely lattice configurations directly from the input chemistry.

## What Changed

### New Prediction Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                     CHEMISTRY INPUT                               │
│  Metals: [{symbol, charge, ratio, CN, radius}, ...]              │
│  Anion: {symbol, charge}                                          │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              orbit_search_generator.py                            │
│  • Shannon ionic radii database                                   │
│  • Pauling radius-ratio → CN prediction                           │
│  • Bond allocation models (A, B, C, D)                            │
│  • Anion CN consistency check                                     │
│  • Constraint generation (CN targets, uniformity, etc.)           │
│  Output: SearchSpec objects                                       │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│          chemistry_predictor_integration.py                       │
│  • Converts SearchSpec → app config format                        │
│  • Ensures cubic P/I/F always included                            │
│  • Adds common lattices (tetragonal, hexagonal)                   │
│  Output: Config dicts for sphere intersection model               │
└─────────────────────────────┬────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                 Sphere Intersection Model                         │
│  • Find minimum scale factor for target CN                        │
│  • Calculate anion positions                                      │
│  • Optimize regularity and Madelung energy                        │
└──────────────────────────────────────────────────────────────────┘
```

### Key Benefits

1. **Chemistry-Driven**: Predictions based on ionic radii and coordination chemistry
2. **Pauling Rules**: Uses radius-ratio thresholds for CN prediction
3. **Bond Allocation Models**: 
   - Model A: CN_MX ≈ CN1 (shell 1 only)
   - Model B: CN_MX ≈ CN1 + CN2/2 (edge-sharing)
   - Model C: CN_MX ≈ CN1 + CN2 (full sharing)
   - Model D: CN_MX ≈ 2×CN1 (dense/face-sharing)
4. **Fallback Support**: Falls back to empirical predictor if chemistry modules unavailable

## Files

### New Files

| File | Lines | Purpose |
|------|-------|---------|
| `orbit_search_generator.py` | 1410 | Chemistry → SearchSpec generation |
| `lattice_search_integration.py` | 564 | SearchSpec → Config adapter |
| `lattice_search.py` | 3608 | HNF-based lattice enumeration |
| `chemistry_predictor_integration.py` | 550 | App integration layer |

### Modified Files

| File | Change |
|------|--------|
| `app.py` | Updated `get_default_search_configs()` to use chemistry predictor |

### Existing Files (unchanged)

| File | Purpose |
|------|---------|
| `interstitial_engine.py` | Sphere intersection calculations |
| `position_calculator.py` | Structure generation and analysis |
| `ionic_radii.py` | Shannon radii database |
| `lattice_configs.py` | Fallback lattice configurations |

## Installation

1. Copy all new Python files to your app directory:

```bash
cp orbit_search_generator.py /path/to/app/
cp lattice_search_integration.py /path/to/app/
cp lattice_search.py /path/to/app/
cp chemistry_predictor_integration.py /path/to/app/
cp app.py /path/to/app/  # Updated version
```

2. The app will automatically detect if the chemistry modules are available

## Usage

The app works exactly as before, but now uses chemistry-based predictions:

1. Enter metal cation properties (symbol, charge, ratio, CN, radius)
2. Enter anion properties (symbol, charge, radius)
3. Click "Calculate Stoichiometry & CN"
4. The app predicts lattice configurations based on:
   - Input coordination numbers
   - Pauling radius-ratio rules
   - Bond allocation models

### Sidebar Status

The sidebar now shows predictor status:
- ✅ **Active**: Chemistry predictor is working
- ⚠️ **Unavailable**: Using fallback empirical predictor

## Technical Details

### Pauling Radius-Ratio Thresholds

| r+/r- | CN |
|-------|-----|
| 0.155 | 2 |
| 0.225 | 3 |
| 0.414 | 4 |
| 0.732 | 6 |
| 1.000 | 8 |
| ∞ | 12 |

### Bond Allocation Models

For 2-species systems (e.g., SrTiO3):

- **Model A (shell-1 only)**: Total cation CN ≈ CN1 from radius ratio
- **Model B (edge-sharing)**: Total cation CN ≈ CN1 + CN2/2
- **Model C (full sharing)**: Total cation CN ≈ CN1 + CN2
- **Model D (face-sharing)**: Total cation CN ≈ 2×CN1

### Constraints Generated

1. **CN1TotalConstraint**: Target first-shell CN sum
2. **CN1RangeConstraint**: Allow CN variation for large cations
3. **MajorityNeighborConstraint**: Prefer cross-orbit neighbors (70%)
4. **MinSeparationConstraint**: Ensure sufficient cation-cation distance
5. **ShellGapConstraint**: Clear separation between shells
6. **OrbitUniformConstraint**: All sites in orbit have same CN

## Deployment (Cloud Run / Docker)

Add the new files to your Dockerfile:

```dockerfile
COPY orbit_search_generator.py .
COPY lattice_search_integration.py .
COPY lattice_search.py .
COPY chemistry_predictor_integration.py .
COPY app.py .
```

## Examples

### SrTiO3 (Perovskite)

Input:
- Sr: charge=2, ratio=1, CN=12
- Ti: charge=4, ratio=1, CN=6
- O: charge=-2

Predicted configs:
- `CHEM-SC-SrCN[12]_TiCN[6]-0`: Cubic-P with Sr CN=12, Ti CN=6
- Plus cubic I/F, tetragonal, hexagonal variants

### MgAl₂O₄ (Spinel)

Input:
- Mg: charge=2, ratio=1, CN=4
- Al: charge=3, ratio=2, CN=6
- O: charge=-2

Predicted configs:
- `CHEM-SC-MgCN[4]_AlCN[6]-0`: Mg tetrahedral, Al octahedral
- Plus cubic variants for FCC-based spinel structure

## Troubleshooting

### Chemistry predictor shows "Unavailable"

Check that all required files are present:
```bash
ls -la orbit_search_generator.py lattice_search_integration.py lattice_search.py chemistry_predictor_integration.py
```

### Import errors

Ensure all files are in the Python path:
```python
import sys
sys.path.insert(0, '/path/to/app')
```

### Performance

The chemistry predictor is fast (~100ms). For full HNF enumeration (slower but more accurate), set `run_lattice_search=True` in the predictor call.
