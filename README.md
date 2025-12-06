# AFLOW Predictor Integration

## Files

| File | Purpose |
|------|---------|
| `app.py` | Modified app with predictor integration in `get_default_search_configs()` |
| `predictor_integration.py` | Bridge module between AFLOW predictor and app |
| `sublattice_predictor_module.py` | Core predictor API (from your ZIP) |
| `sublattice_lookup.json` | AFLOW prediction data for N=1-16 |
| `lattice_configs.py` | Unchanged - existing config catalogue |

## What Changed

### `get_default_search_configs(num_metals, use_predictor=True)`

Now uses AFLOW-based predictions when available:

```python
# Uses predictor (default)
configs = get_default_search_configs(4)

# Force fallback behavior  
configs = get_default_search_configs(4, use_predictor=False)
```

### Prediction Strategy by N

| N | Predictability | Strategy |
|---|----------------|----------|
| 1, 5, 7 | HIGH | Uses exact offset patterns from AFLOW data |
| 2, 3 | MODERATE | Uses lattice types + top offset patterns |
| 4, 6, 8+ | LOW | Uses lattice type priors only |

## Config Output Format

Configs now include additional fields:

```python
{
    'id': 'PRED-N4-monoclinic_P-1',
    'lattice': 'Monoclinic',
    'bravais_type': 'monoclinic_P',
    'offsets': [(0,0,0), (0.5,0.5,0), ...],
    'pattern': 'AFLOW 30.1%',
    'c_ratio': 4.0,
    'probability': 0.301,        # NEW: from AFLOW stats
    'source': 'aflow_lattice_type'  # NEW: prediction source
}
```

## Installation

Place all files in your app directory alongside existing code. The predictor auto-discovers `sublattice_lookup.json` in common locations.

## Testing

```python
from predictor_integration import get_predictor

predictor = get_predictor()
print(f"Available: {predictor.is_available()}")
print(f"N=4 predictability: {predictor.get_predictability(4)}")

configs = predictor.get_search_configs_from_predictor(4)
for c in configs[:5]:
    print(f"{c['id']}: {c['probability']:.1%}")
```
