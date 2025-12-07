# Metal Sublattice Prediction Strategy

## Overview

This document provides a comprehensive strategy for generating the 10-20 most likely metal sublattice configurations for a given N (number of unique metal positions per primitive cell).

Based on analysis of 1802 AFLOW structures, this strategy combines:
1. Empirical pattern frequencies
2. Lattice-type-specific filling rules
3. N-decomposition into (in-plane) × (z-layers)
4. Special position preferences

---

## Quick Reference: Top Patterns by N

### N = 1
| Rank | Lattice | Pattern | Probability |
|------|---------|---------|-------------|
| 1 | Hexagonal-P | (0,0,0) | 28.6% |
| 2 | Triclinic-P | (0,0,0) | 16.1% |
| 3 | Monoclinic-C | (0,0,0) | 16.1% |
| 4 | Rhombohedral-R | (0,0,0) | 12.5% |
| 5 | Cubic-F | (0,0,0) | 10.7% |

### N = 2
| Rank | Lattice | Pattern | Probability |
|------|---------|---------|-------------|
| 1 | Hexagonal-P | (0,0,0), (0,0,½) | 5.3% |
| 2 | Hexagonal-P | (0,0,0), (⅓,⅔,½) | 5.0% |
| 3 | Monoclinic-C | (0,0,0), (0,0,½) | 4.3% |
| 4 | Rhombohedral-R | (0,0,0), (0,0,½) | 3.6% |
| 5 | Orthorhombic-C | (0,0,0), (0,0,½) | 3.3% |
| 6 | Tetragonal-P | (0,0,0), (½,½,0) | 3.0% |

### N = 4
| Rank | Lattice | Pattern | Probability |
|------|---------|---------|-------------|
| 1 | Tetragonal-P | (0,0,0), (0,½,0), (½,0,0), (½,½,0) | 1.7% |
| 2 | Cubic-P | (0,0,0), (0,½,½), (½,0,½), (½,½,0) | 1.2% |
| 3 | Orthorhombic-P | (0,0,0), (0,½,0), (½,0,½), (½,½,½) | 1.2% |
| 4 | Hexagonal-P | (0,0,0), (0,0,½), (⅔,⅓,0), (⅔,⅓,½) | 1.0% |

### N = 8
| Rank | Lattice | Pattern | Probability |
|------|---------|---------|-------------|
| 1 | Orthorhombic-P | All 8 corners: (0,0,0), (0,0,½), (0,½,0), (0,½,½), (½,0,0), (½,0,½), (½,½,0), (½,½,½) | 2.1% |
| 2 | Tetragonal-I | Complex 8-site pattern | 1.1% |
| 3 | Tetragonal-P | All 8 corners | 0.8% |

---

## Strategy Algorithm

### Step 1: Determine Likely Lattice Types

For a given N, the probability of each Bravais lattice type varies:

| N | Top Lattice Types (by probability) |
|---|-----------------------------------|
| 1 | Hex-P (29%), Tri-P (16%), Mon-C (16%), Rhl-R (13%) |
| 2 | Mon-C (19%), Hex-P (16%), Mon-P (11%), Rhl-R (8%) |
| 4 | Mon-P (30%), Mon-C (24%), Orc-P (14%), Tet-I (7%) |
| 8 | Orc-P (34%), Tet-I (11%), Mon-P (11%), Mon-C (9%) |

### Step 2: Apply Lattice-Specific Filling Rules

Each lattice type has a characteristic "filling order" - positions that appear as N increases:

#### Cubic-P
```
N=1: (0,0,0)
N=2: + (½,½,½)              ← BCC-like, 100% of N=2
N=3: + (½,½,0) or (½,0,½)   ← FCC face center
N=4: All three face centers  ← Complete FCC-like
```

#### Hexagonal-P
```
N=1: (0,0,0)
N=2: + (0,0,½)         34%  ← Simple z-doubling
  OR + (⅓,⅔,½)         32%  ← Wurtzite-like
N=4: Combine wurtzite + z-doubling
```

#### Tetragonal-P
```
N=1: (0,0,0)
N=2: + (½,½,0)              ← Checkerboard, 69% of N=2!
N=4: + (0,½,0), (½,0,0)     ← Complete square
N=8: + z=½ layer
```

#### Orthorhombic-P
```
N=1: (0,0,0)
N=4: + (0,½,½), (½,0,½), (½,½,0)  ← (0,½,½) strongly preferred
N=8: All 8 corner positions
```

#### Rhombohedral-R
```
Filling is 1D along z-axis:
N=2: (0,0,0), (0,0,½)
N=3: (0,0,0), (0,0,¼), (0,0,¾)  OR  (0,0,0), (0,0,⅓), (0,0,⅔)
```

### Step 3: Apply N-Decomposition

**Key Finding**: N = (in-plane multiplicity) × (z-layers) for 90% of structures.

| N | Common Decompositions | Implied z-fractions |
|---|----------------------|---------------------|
| 4 | 1×4 (48%), 2×2 (47%) | 0,¼,½,¾ or 0,½ |
| 6 | 3×2 (53%), 2×3 (26%) | 0,½ or 0,⅓,⅔ |
| 8 | 2×4 (64%), 4×2 (32%) | 0,¼,½,¾ or 0,½ |
| 12 | 3×4 (40%), 4×3 (30%) | 0,¼,½,¾ or 0,⅓,⅔ |

**Z-layer templates**:
- 2 layers: z = 0, ½
- 3 layers: z = 0, ⅓, ⅔
- 4 layers: z = 0, ¼, ½, ¾
- 6 layers: z = 0, ⅙, ⅓, ½, ⅔, ⅚

**In-plane templates** (at each z-layer):
- 1 in-plane: (0,0)
- 2 in-plane: (0,0), (½,½) or (0,0), (⅓,⅔)
- 4 in-plane: (0,0), (0,½), (½,0), (½,½)

### Step 4: Generate Candidate List

Combine the above rules to generate candidates, then rank by:

1. **Observed frequency** (highest priority) - patterns seen multiple times in database
2. **Template match** - patterns matching lattice-specific rules
3. **Decomposition match** - patterns from N-decomposition

---

## Detailed Lattice Rules

### Cubic Systems

| Lattice | Key Positions | Filling Character |
|---------|---------------|-------------------|
| Cubic-P | Origin, body-center (½,½,½), face-centers | BCC-like at N=2, FCC-like at N=4 |
| Cubic-F | Origin, body-center, diamond (¼,¼,¼) | Diamond positions appear early |
| Cubic-I | Origin, (0,½,½)-type positions | Garnet patterns at N=32 |

### Tetragonal Systems

| Lattice | Key Positions | Filling Character |
|---------|---------------|-------------------|
| Tetragonal-P | Origin, checkerboard (½,½,0), edges | Checkerboard dominates N=2 |
| Tetragonal-I | Origin, (0,½,¼)-shift | z=¼ preferred over z=½ |

### Hexagonal/Trigonal Systems

| Lattice | Key Positions | Filling Character |
|---------|---------------|-------------------|
| Hexagonal-P | Origin, z=½, wurtzite (⅓,⅔,½) | Two competing N=2 patterns |
| Rhombohedral-R | Origin, z-axis only | Almost purely 1D filling |

### Orthorhombic Systems

| Lattice | Key Positions | Filling Character |
|---------|---------------|-------------------|
| Orthorhombic-P | Origin, (0,½,½) strongly preferred | Progressive corner filling |
| Orthorhombic-C | Origin, (0,0,½), x=⅓,⅔ | C-centering adds ⅓ positions |

### Monoclinic Systems

| Lattice | Key Positions | Filling Character |
|---------|---------------|-------------------|
| Monoclinic-P | Origin, (0,0,½), (0,½,½) | Lower symmetry, varied patterns |
| Monoclinic-C | Origin, (0,0,½), (⅔,0,0) | C-centering effects |

---

## Example: Generating Predictions for N=4

**Step 1**: Likely lattice types for N=4:
- Monoclinic-P (30%), Monoclinic-C (24%), Orthorhombic-P (14%), Tetragonal-I (7%)

**Step 2**: Apply lattice rules:
- Tetragonal-P: (0,0,0), (0,½,0), (½,0,0), (½,½,0) - square in xy-plane
- Cubic-P: (0,0,0), (0,½,½), (½,0,½), (½,½,0) - FCC-like
- Orthorhombic-P: (0,0,0), (0,½,½), (½,0,½), (½,½,0)

**Step 3**: N-decomposition:
- 2×2: 2 in-plane positions at each of 2 z-layers
  - z=0: (0,0,0), (½,½,0); z=½: (0,0,½), (½,½,½)
- 1×4: 1 in-plane position at each of 4 z-layers
  - (0,0,0), (0,0,¼), (0,0,½), (0,0,¾)

**Final ranked list**:
1. Tet-P: (0,0,0), (0,½,0), (½,0,0), (½,½,0) - 1.7%
2. Cub-P: (0,0,0), (0,½,½), (½,0,½), (½,½,0) - 1.2%
3. Orc-P: variant patterns - 1.2%
4. Hex-P: (0,0,0), (0,0,½), (⅔,⅓,0), (⅔,⅓,½) - 1.0%
5. ... continue with templates and decompositions

---

## Key Insights

1. **Pattern diversity increases with N**: At N≥4, patterns become highly compound-specific. Only 5% of patterns appear more than once.

2. **Special positions dominate shared patterns**: Patterns appearing multiple times use only 0, ½, ⅓, ⅔, ¼, ¾.

3. **Lattice type is the strongest predictor**: Once you know the Bravais lattice, the filling order is quite constrained.

4. **z-layers correlate with N**: The number of distinct z-values is typically a divisor of N.

5. **Monoclinic/Triclinic = high diversity**: Lower symmetry → more possible patterns → harder to predict.

---

## Files Generated

| File | Description |
|------|-------------|
| `lattice_prediction_strategy.json` | Complete prediction database with strategy rules |
| `offset_filling_database.json` | Filling order statistics by lattice type |
| `z_fraction_N_correlation.json` | Z-coordinate fraction analysis |
| `sublattice_lookup.json` | Empirical lookup table by (N, Bravais) |

