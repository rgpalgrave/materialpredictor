"""
Lattice Configuration Catalogue
Complete set of metal sublattice configurations for N=1 through N=8
"""
from dataclasses import dataclass
from typing import Optional
import re


@dataclass
class LatticeConfig:
    """A single lattice configuration entry."""
    id: str
    lattice: str
    offsets: list[tuple[float, float, float]]
    params: str
    pattern: str = ""
    free_param: Optional[str] = None  # For arity-1 configs
    bravais_type: str = ""  # e.g., 'cubic_P', 'cubic_F', 'cubic_I'
    
    @property
    def is_parametric(self) -> bool:
        return self.free_param is not None
    
    @property
    def arity(self) -> int:
        return 1 if self.is_parametric else 0
    
    def __post_init__(self):
        # Set default bravais_type if not specified
        if not self.bravais_type:
            self.bravais_type = self.lattice.lower() + '_P'


def parse_offset(s: str) -> tuple[float, float, float]:
    """Parse offset string like '(0,0,0)', '(½,½,½)', '(⅓,⅔,0)' to float tuple."""
    s = s.strip().strip('()')
    parts = s.split(',')
    
    def parse_frac(f: str) -> float:
        f = f.strip()
        # Handle unicode fractions
        frac_map = {
            '½': 0.5, '⅓': 1/3, '⅔': 2/3, '¼': 0.25, '¾': 0.75,
            '⅕': 0.2, '⅖': 0.4, '⅗': 0.6, '⅘': 0.8,
            '⅙': 1/6, '⅚': 5/6, '⅐': 1/7, '⅛': 0.125, '⅜': 0.375,
            '⅝': 0.625, '⅞': 0.875,
        }
        for k, v in frac_map.items():
            if k in f:
                return v
        # Handle parametric entries like 'x', '2x', 'z', etc.
        if any(c in f for c in 'xyz'):
            return None  # Parametric
        return float(f)
    
    result = []
    for p in parts:
        val = parse_frac(p)
        if val is None:
            return None  # Parametric offset
        result.append(val)
    return tuple(result)


# N=1 configurations - includes all 14 Bravais lattices
N1_CONFIGS = {
    'arity0': [
        # Cubic
        LatticeConfig('N1-cP', 'Cubic', [(0,0,0)], 'a', 'Primitive', bravais_type='cubic_P'),
        LatticeConfig('N1-cI', 'Cubic', [(0,0,0)], 'a', 'BCC', bravais_type='cubic_I'),
        LatticeConfig('N1-cF', 'Cubic', [(0,0,0)], 'a', 'FCC', bravais_type='cubic_F'),
        # Tetragonal
        LatticeConfig('N1-tP', 'Tetragonal', [(0,0,0)], 'a, c/a', 'Primitive', bravais_type='tetragonal_P'),
        LatticeConfig('N1-tI', 'Tetragonal', [(0,0,0)], 'a, c/a', 'Body-centered', bravais_type='tetragonal_I'),
        # Orthorhombic
        LatticeConfig('N1-oP', 'Orthorhombic', [(0,0,0)], 'a, b/a, c/a', 'Primitive', bravais_type='orthorhombic_P'),
        LatticeConfig('N1-oI', 'Orthorhombic', [(0,0,0)], 'a, b/a, c/a', 'Body-centered', bravais_type='orthorhombic_I'),
        LatticeConfig('N1-oF', 'Orthorhombic', [(0,0,0)], 'a, b/a, c/a', 'Face-centered', bravais_type='orthorhombic_F'),
        LatticeConfig('N1-oC', 'Orthorhombic', [(0,0,0)], 'a, b/a, c/a', 'C-centered', bravais_type='orthorhombic_C'),
        # Hexagonal
        LatticeConfig('N1-hP', 'Hexagonal', [(0,0,0)], 'a, c/a', 'Primitive', bravais_type='hexagonal_P'),
        LatticeConfig('N1-hR', 'Hexagonal', [(0,0,0)], 'a, c/a', 'HCP', bravais_type='hexagonal_H'),
        # Rhombohedral
        LatticeConfig('N1-rP', 'Rhombohedral', [(0,0,0)], 'a, α', 'Primitive', bravais_type='rhombohedral_P'),
        # Monoclinic
        LatticeConfig('N1-mP', 'Monoclinic', [(0,0,0)], 'a, b/a, c/a, β', 'Primitive', bravais_type='monoclinic_P'),
        LatticeConfig('N1-mC', 'Monoclinic', [(0,0,0)], 'a, b/a, c/a, β', 'C-centered', bravais_type='monoclinic_C'),
    ],
    'arity1': []
}

# N=2 configurations
N2_CONFIGS = {
    'arity0': [
        # Cubic Primitive
        LatticeConfig('N2-cP1', 'Cubic', [(0,0,0), (0.5,0.5,0.5)], 'a', 'Body center (CsCl)', bravais_type='cubic_P'),
        LatticeConfig('N2-cP2', 'Cubic', [(0,0,0), (0.5,0.5,0)], 'a', 'Face center', bravais_type='cubic_P'),
        LatticeConfig('N2-cP3', 'Cubic', [(0,0,0), (0.5,0,0)], 'a', 'Edge center', bravais_type='cubic_P'),
        LatticeConfig('N2-cP4', 'Cubic', [(0,0,0), (0.25,0.25,0.25)], 'a', 'Quarter diagonal', bravais_type='cubic_P'),
        # Cubic BCC
        LatticeConfig('N2-cI1', 'Cubic', [(0,0,0), (0.5,0.5,0)], 'a', 'Face center', bravais_type='cubic_I'),
        LatticeConfig('N2-cI2', 'Cubic', [(0,0,0), (0.25,0.25,0.25)], 'a', 'Quarter diagonal', bravais_type='cubic_I'),
        # Cubic FCC
        LatticeConfig('N2-cF1', 'Cubic', [(0,0,0), (0.5,0.5,0.5)], 'a', 'Body center', bravais_type='cubic_F'),
        LatticeConfig('N2-cF2', 'Cubic', [(0,0,0), (0.25,0.25,0.25)], 'a', 'Quarter diagonal', bravais_type='cubic_F'),
        # Tetragonal Primitive
        LatticeConfig('N2-tP1', 'Tetragonal', [(0,0,0), (0,0,0.5)], 'a, c/a', 'c-stack', bravais_type='tetragonal_P'),
        LatticeConfig('N2-tP2', 'Tetragonal', [(0,0,0), (0.5,0.5,0)], 'a, c/a', 'In-plane center', bravais_type='tetragonal_P'),
        LatticeConfig('N2-tP3', 'Tetragonal', [(0,0,0), (0.5,0.5,0.5)], 'a, c/a', 'Body center', bravais_type='tetragonal_P'),
        LatticeConfig('N2-tP4', 'Tetragonal', [(0,0,0), (0,0.5,0)], 'a, c/a', 'Edge', bravais_type='tetragonal_P'),
        LatticeConfig('N2-tP5', 'Tetragonal', [(0,0,0), (0,0.5,0.5)], 'a, c/a', 'Edge + c/2', bravais_type='tetragonal_P'),
        # Tetragonal Body-centered
        LatticeConfig('N2-tI1', 'Tetragonal', [(0,0,0), (0,0,0.5)], 'a, c/a', 'c-stack', bravais_type='tetragonal_I'),
        LatticeConfig('N2-tI2', 'Tetragonal', [(0,0,0), (0.5,0.5,0)], 'a, c/a', 'In-plane center', bravais_type='tetragonal_I'),
        # Hexagonal Primitive
        LatticeConfig('N2-hP1', 'Hexagonal', [(0,0,0), (0,0,0.5)], 'a, c/a', 'c-stack', bravais_type='hexagonal_P'),
        LatticeConfig('N2-hP2', 'Hexagonal', [(0,0,0), (1/3,2/3,0)], 'a, c/a', 'Triangular', bravais_type='hexagonal_P'),
        LatticeConfig('N2-hP3', 'Hexagonal', [(0,0,0), (1/3,2/3,0.5)], 'a, c/a', 'Triangular + c/2', bravais_type='hexagonal_P'),
        LatticeConfig('N2-hP4', 'Hexagonal', [(0,0,0), (0.5,0,0)], 'a, c/a', 'Edge', bravais_type='hexagonal_P'),
        LatticeConfig('N2-hP5', 'Hexagonal', [(0,0,0), (0.5,0.5,0)], 'a, c/a', '', bravais_type='hexagonal_P'),
        # Hexagonal HCP
        LatticeConfig('N2-hR1', 'Hexagonal', [(0,0,0), (0,0,0.5)], 'a, c/a', 'c-stack', bravais_type='hexagonal_H'),
        LatticeConfig('N2-hR2', 'Hexagonal', [(0,0,0), (1/3,2/3,0)], 'a, c/a', 'Triangular', bravais_type='hexagonal_H'),
        # Orthorhombic Primitive
        LatticeConfig('N2-oP1', 'Orthorhombic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a', 'c-stack', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP2', 'Orthorhombic', [(0,0,0), (0,0.5,0)], 'a, b/a, c/a', 'b-stack', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP3', 'Orthorhombic', [(0,0,0), (0.5,0,0)], 'a, b/a, c/a', 'a-stack', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP4', 'Orthorhombic', [(0,0,0), (0.5,0.5,0.5)], 'a, b/a, c/a', 'Body center', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP5', 'Orthorhombic', [(0,0,0), (0.5,0.5,0)], 'a, b/a, c/a', 'ab-face', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP6', 'Orthorhombic', [(0,0,0), (0.5,0,0.5)], 'a, b/a, c/a', 'ac-face', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP7', 'Orthorhombic', [(0,0,0), (0,0.5,0.5)], 'a, b/a, c/a', 'bc-face', bravais_type='orthorhombic_P'),
        # Orthorhombic Body-centered
        LatticeConfig('N2-oI1', 'Orthorhombic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a', 'c-stack', bravais_type='orthorhombic_I'),
        LatticeConfig('N2-oI2', 'Orthorhombic', [(0,0,0), (0.5,0.5,0)], 'a, b/a, c/a', 'ab-face', bravais_type='orthorhombic_I'),
        # Orthorhombic Face-centered
        LatticeConfig('N2-oF1', 'Orthorhombic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a', 'c-stack', bravais_type='orthorhombic_F'),
        LatticeConfig('N2-oF2', 'Orthorhombic', [(0,0,0), (0.5,0.5,0.5)], 'a, b/a, c/a', 'Body center', bravais_type='orthorhombic_F'),
        # Orthorhombic C-centered
        LatticeConfig('N2-oC1', 'Orthorhombic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a', 'c-stack', bravais_type='orthorhombic_C'),
        LatticeConfig('N2-oC2', 'Orthorhombic', [(0,0,0), (0.5,0.5,0.5)], 'a, b/a, c/a', 'Body center', bravais_type='orthorhombic_C'),
        # Rhombohedral
        LatticeConfig('N2-rP1', 'Rhombohedral', [(0,0,0), (0,0,0.5)], 'a, α', 'c-stack', bravais_type='rhombohedral_P'),
        LatticeConfig('N2-rP2', 'Rhombohedral', [(0,0,0), (1/3,2/3,1/3)], 'a, α', 'R-offset', bravais_type='rhombohedral_P'),
        # Monoclinic Primitive
        LatticeConfig('N2-mP1', 'Monoclinic', [(0,0,0), (0,0.5,0)], 'a, b/a, c/a, β', 'b-stack', bravais_type='monoclinic_P'),
        LatticeConfig('N2-mP2', 'Monoclinic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a, β', 'c-stack', bravais_type='monoclinic_P'),
        LatticeConfig('N2-mP3', 'Monoclinic', [(0,0,0), (0.5,0,0)], 'a, b/a, c/a, β', 'a-stack', bravais_type='monoclinic_P'),
        LatticeConfig('N2-mP4', 'Monoclinic', [(0,0,0), (0.5,0.5,0.5)], 'a, b/a, c/a, β', 'Body center', bravais_type='monoclinic_P'),
        LatticeConfig('N2-mP5', 'Monoclinic', [(0,0,0), (0.5,0,0.5)], 'a, b/a, c/a, β', 'ac-face', bravais_type='monoclinic_P'),
        # Monoclinic C-centered
        LatticeConfig('N2-mC1', 'Monoclinic', [(0,0,0), (0,0,0.5)], 'a, b/a, c/a, β', 'c-stack', bravais_type='monoclinic_C'),
        LatticeConfig('N2-mC2', 'Monoclinic', [(0,0,0), (0.5,0.5,0.5)], 'a, b/a, c/a, β', 'Body center', bravais_type='monoclinic_C'),
    ],
    'arity1': [
        LatticeConfig('N2-cP5', 'Cubic', None, 'a', '', 'x ∈ (0, ½]', bravais_type='cubic_P'),
        LatticeConfig('N2-tP6', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ½]', bravais_type='tetragonal_P'),
        LatticeConfig('N2-tP7', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ½]', bravais_type='tetragonal_P'),
        LatticeConfig('N2-hP6', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ½]', bravais_type='hexagonal_P'),
        LatticeConfig('N2-hP7', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ½]', bravais_type='hexagonal_P'),
        LatticeConfig('N2-oP8', 'Orthorhombic', None, 'a, b/a, c/a', '', 'z ∈ (0, ½]', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP9', 'Orthorhombic', None, 'a, b/a, c/a', '', 'y ∈ (0, ½]', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-oP10', 'Orthorhombic', None, 'a, b/a, c/a', '', 'x ∈ (0, ½]', bravais_type='orthorhombic_P'),
        LatticeConfig('N2-rP3', 'Rhombohedral', None, 'a, α', '', 'z ∈ (0, ½]', bravais_type='rhombohedral_P'),
        LatticeConfig('N2-mP6', 'Monoclinic', None, 'a, b/a, c/a, β', '', 'y ∈ (0, ½]', bravais_type='monoclinic_P'),
    ]
}

# N=3 configurations
N3_CONFIGS = {
    'arity0': [
        LatticeConfig('N3-C1', 'Cubic', [(0,0,0), (1/3,1/3,1/3), (2/3,2/3,2/3)], 'a', '[111] thirds'),
        LatticeConfig('N3-C2', 'Cubic', [(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0)], 'a', 'Body + face'),
        LatticeConfig('N3-C3', 'Cubic', [(0,0,0), (0.5,0.5,0), (0.5,0,0.5)], 'a', 'Two faces'),
        LatticeConfig('N3-C4', 'Cubic', [(0,0,0), (0.5,0,0), (0,0.5,0)], 'a', 'Two edges'),
        LatticeConfig('N3-C5', 'Cubic', [(0,0,0), (0.25,0.25,0.25), (0.5,0.5,0.5)], 'a', 'Quarter + body'),
        LatticeConfig('N3-T1', 'Tetragonal', [(0,0,0), (0,0,1/3), (0,0,2/3)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N3-T2', 'Tetragonal', [(0,0,0), (0,0,0.5), (0.5,0.5,0)], 'a, c/a', 'c/2 + in-plane'),
        LatticeConfig('N3-T3', 'Tetragonal', [(0,0,0), (0,0,0.5), (0.5,0.5,0.5)], 'a, c/a', 'c/2 + body'),
        LatticeConfig('N3-T4', 'Tetragonal', [(0,0,0), (0.5,0.5,0), (0.5,0.5,0.5)], 'a, c/a', 'In-plane stack'),
        LatticeConfig('N3-T5', 'Tetragonal', [(0,0,0), (0,0,0.25), (0,0,0.5)], 'a, c/a', 'Unequal c-stack'),
        LatticeConfig('N3-T6', 'Tetragonal', [(0,0,0), (0,0.5,0), (0.5,0,0)], 'a, c/a', 'Two edges'),
        LatticeConfig('N3-H1', 'Hexagonal', [(0,0,0), (0,0,1/3), (0,0,2/3)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N3-H2', 'Hexagonal', [(0,0,0), (1/3,2/3,0), (2/3,1/3,0)], 'a, c/a', 'In-plane triangular'),
        LatticeConfig('N3-H3', 'Hexagonal', [(0,0,0), (0,0,0.5), (1/3,2/3,0)], 'a, c/a', 'Mixed'),
        LatticeConfig('N3-H4', 'Hexagonal', [(0,0,0), (1/3,2/3,1/3), (2/3,1/3,2/3)], 'a, c/a', 'Helical'),
        LatticeConfig('N3-H5', 'Hexagonal', [(0,0,0), (0.5,0,0), (0.25,0.5,0)], 'a, c/a', 'In-plane'),
        LatticeConfig('N3-H6', 'Hexagonal', [(0,0,0), (0,0,0.5), (1/3,2/3,0.5)], 'a, c/a', 'c/2 + triangular'),
        LatticeConfig('N3-O1', 'Orthorhombic', [(0,0,0), (0,0,1/3), (0,0,2/3)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N3-O2', 'Orthorhombic', [(0,0,0), (0,1/3,0), (0,2/3,0)], 'a, b/a, c/a', 'Equal b-stack'),
        LatticeConfig('N3-O3', 'Orthorhombic', [(0,0,0), (1/3,0,0), (2/3,0,0)], 'a, b/a, c/a', 'Equal a-stack'),
        LatticeConfig('N3-O4', 'Orthorhombic', [(0,0,0), (0.5,0,0), (0,0.5,0)], 'a, b/a, c/a', 'Two edges'),
        LatticeConfig('N3-O5', 'Orthorhombic', [(0,0,0), (0,0,0.5), (0.5,0.5,0)], 'a, b/a, c/a', 'c/2 + ab-face'),
        LatticeConfig('N3-O6', 'Orthorhombic', [(0,0,0), (0.5,0.5,0), (0.5,0.5,0.5)], 'a, b/a, c/a', 'ab-face stack'),
        LatticeConfig('N3-R1', 'Rhombohedral', [(0,0,0), (0,0,1/3), (0,0,2/3)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N3-R2', 'Rhombohedral', [(0,0,0), (1/3,2/3,1/3), (2/3,1/3,2/3)], 'a, α', 'R-helical'),
        LatticeConfig('N3-M1', 'Monoclinic', [(0,0,0), (0,1/3,0), (0,2/3,0)], 'a, b/a, c/a, β', 'Equal b-stack'),
        LatticeConfig('N3-M2', 'Monoclinic', [(0,0,0), (0,0.5,0), (0,0,0.5)], 'a, b/a, c/a, β', 'b/2 + c/2'),
        LatticeConfig('N3-M3', 'Monoclinic', [(0,0,0), (0,0.5,0), (0.5,0,0)], 'a, b/a, c/a, β', 'b/2 + a/2'),
        LatticeConfig('N3-M4', 'Monoclinic', [(0,0,0), (0,0.5,0), (0.5,0.5,0.5)], 'a, b/a, c/a, β', 'b/2 + body'),
    ],
    'arity1': [
        LatticeConfig('N3-C6', 'Cubic', None, 'a', '', 'x ∈ (0, ⅓]'),
        LatticeConfig('N3-T7', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ⅓]'),
        LatticeConfig('N3-T8', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ½]'),
        LatticeConfig('N3-H7', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ⅓]'),
        LatticeConfig('N3-H8', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ⅓]'),
        LatticeConfig('N3-O7', 'Orthorhombic', None, 'a, b/a, c/a', '', 'z ∈ (0, ⅓]'),
        LatticeConfig('N3-O8', 'Orthorhombic', None, 'a, b/a, c/a', '', 'y ∈ (0, ⅓]'),
        LatticeConfig('N3-R3', 'Rhombohedral', None, 'a, α', '', 'z ∈ (0, ⅓]'),
        LatticeConfig('N3-M5', 'Monoclinic', None, 'a, b/a, c/a, β', '', 'y ∈ (0, ⅓]'),
    ]
}

# N=4 configurations (FCC, etc.)
N4_CONFIGS = {
    'arity0': [
        LatticeConfig('N4-C1', 'Cubic', [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)], 'a', 'FCC motif'),
        LatticeConfig('N4-C2', 'Cubic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5)], 'a', 'Three edges'),
        LatticeConfig('N4-C3', 'Cubic', [(0,0,0), (0.25,0.25,0.25), (0.5,0.5,0.5), (0.75,0.75,0.75)], 'a', '[111] quarters'),
        LatticeConfig('N4-C4', 'Cubic', [(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0), (0.5,0,0.5)], 'a', 'Body + 2 faces'),
        LatticeConfig('N4-C5', 'Cubic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0)], 'a', 'ab-plane grid'),
        LatticeConfig('N4-T1', 'Tetragonal', [(0,0,0), (0,0,0.25), (0,0,0.5), (0,0,0.75)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N4-T2', 'Tetragonal', [(0,0,0), (0,0,0.5), (0.5,0.5,0), (0.5,0.5,0.5)], 'a, c/a', 'Two parallel stacks'),
        LatticeConfig('N4-T3', 'Tetragonal', [(0,0,0), (0,0,0.5), (0,0.5,0), (0,0.5,0.5)], 'a, c/a', 'c + edge stack'),
        LatticeConfig('N4-T4', 'Tetragonal', [(0,0,0), (0.5,0.5,0), (0,0.5,0.25), (0.5,0,0.75)], 'a, c/a', 'Complex'),
        LatticeConfig('N4-T5', 'Tetragonal', [(0,0,0), (0,0,1/3), (0,0,2/3), (0.5,0.5,0)], 'a, c/a', 'Triple c + in-plane'),
        LatticeConfig('N4-T6', 'Tetragonal', [(0,0,0), (0,0.5,0), (0.5,0,0), (0.5,0.5,0)], 'a, c/a', 'In-plane grid'),
        LatticeConfig('N4-H1', 'Hexagonal', [(0,0,0), (0,0,0.25), (0,0,0.5), (0,0,0.75)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N4-H2', 'Hexagonal', [(0,0,0), (0,0,0.5), (1/3,2/3,0), (1/3,2/3,0.5)], 'a, c/a', 'Origin + tri stacks'),
        LatticeConfig('N4-H3', 'Hexagonal', [(0,0,0), (1/3,2/3,0), (2/3,1/3,0), (0,0,0.5)], 'a, c/a', 'Triangular + c/2'),
        LatticeConfig('N4-H4', 'Hexagonal', [(0,0,0), (1/3,2/3,0.25), (2/3,1/3,0.5), (0,0,0.75)], 'a, c/a', 'Helical'),
        LatticeConfig('N4-H5', 'Hexagonal', [(0,0,0), (0,0,0.5), (1/3,2/3,0.25), (2/3,1/3,0.75)], 'a, c/a', 'Mixed helical'),
        LatticeConfig('N4-O1', 'Orthorhombic', [(0,0,0), (0,0,0.25), (0,0,0.5), (0,0,0.75)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N4-O2', 'Orthorhombic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0)], 'a, b/a, c/a', 'ab-plane grid'),
        LatticeConfig('N4-O3', 'Orthorhombic', [(0,0,0), (0.5,0,0), (0,0,0.5), (0.5,0,0.5)], 'a, b/a, c/a', 'ac-plane grid'),
        LatticeConfig('N4-O4', 'Orthorhombic', [(0,0,0), (0,0.5,0), (0,0,0.5), (0,0.5,0.5)], 'a, b/a, c/a', 'bc-plane grid'),
        LatticeConfig('N4-O5', 'Orthorhombic', [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)], 'a, b/a, c/a', 'Three face centers'),
        LatticeConfig('N4-O6', 'Orthorhombic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5)], 'a, b/a, c/a', 'Three edges'),
        LatticeConfig('N4-R1', 'Rhombohedral', [(0,0,0), (0,0,0.25), (0,0,0.5), (0,0,0.75)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N4-R2', 'Rhombohedral', [(0,0,0), (0,0,0.5), (1/3,2/3,1/3), (2/3,1/3,2/3)], 'a, α', 'Mixed'),
        LatticeConfig('N4-M1', 'Monoclinic', [(0,0,0), (0,0.25,0), (0,0.5,0), (0,0.75,0)], 'a, b/a, c/a, β', 'Equal b-stack'),
        LatticeConfig('N4-M2', 'Monoclinic', [(0,0,0), (0,0.5,0), (0,0,0.5), (0,0.5,0.5)], 'a, b/a, c/a, β', 'bc-plane grid'),
        LatticeConfig('N4-M3', 'Monoclinic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0.5,0.5,0)], 'a, b/a, c/a, β', 'ab-plane grid'),
        LatticeConfig('N4-M4', 'Monoclinic', [(0,0,0), (0.5,0,0), (0,0,0.5), (0.5,0,0.5)], 'a, b/a, c/a, β', 'ac-plane grid'),
    ],
    'arity1': [
        LatticeConfig('N4-C6', 'Cubic', None, 'a', '', 'x ∈ (0, ¼]'),
        LatticeConfig('N4-T7', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ¼]'),
        LatticeConfig('N4-T8', 'Tetragonal', None, 'a, c/a', '', 'z ∈ (0, ½]'),
        LatticeConfig('N4-H6', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ¼]'),
        LatticeConfig('N4-H7', 'Hexagonal', None, 'a, c/a', '', 'z ∈ (0, ½]'),
        LatticeConfig('N4-O7', 'Orthorhombic', None, 'a, b/a, c/a', '', 'z ∈ (0, ¼]'),
        LatticeConfig('N4-R3', 'Rhombohedral', None, 'a, α', '', 'z ∈ (0, ¼]'),
        LatticeConfig('N4-M5', 'Monoclinic', None, 'a, b/a, c/a, β', '', 'y ∈ (0, ¼]'),
    ]
}

# N=5 through N=8 (condensed)
N5_CONFIGS = {
    'arity0': [
        LatticeConfig('N5-C1', 'Cubic', [(0,0,0), (0.2,0.2,0.2), (0.4,0.4,0.4), (0.6,0.6,0.6), (0.8,0.8,0.8)], 'a', '[111] fifths'),
        LatticeConfig('N5-C2', 'Cubic', [(0,0,0), (0.5,0.5,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)], 'a', 'Body + 3 faces'),
        LatticeConfig('N5-T1', 'Tetragonal', [(0,0,0), (0,0,0.2), (0,0,0.4), (0,0,0.6), (0,0,0.8)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N5-H1', 'Hexagonal', [(0,0,0), (0,0,0.2), (0,0,0.4), (0,0,0.6), (0,0,0.8)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N5-O1', 'Orthorhombic', [(0,0,0), (0,0,0.2), (0,0,0.4), (0,0,0.6), (0,0,0.8)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N5-R1', 'Rhombohedral', [(0,0,0), (0,0,0.2), (0,0,0.4), (0,0,0.6), (0,0,0.8)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N5-M1', 'Monoclinic', [(0,0,0), (0,0.2,0), (0,0.4,0), (0,0.6,0), (0,0.8,0)], 'a, b/a, c/a, β', 'Equal b-stack'),
    ],
    'arity1': []
}

N6_CONFIGS = {
    'arity0': [
        LatticeConfig('N6-C1', 'Cubic', [(0,0,0), (1/6,1/6,1/6), (2/6,2/6,2/6), (3/6,3/6,3/6), (4/6,4/6,4/6), (5/6,5/6,5/6)], 'a', '[111] sixths'),
        LatticeConfig('N6-T1', 'Tetragonal', [(0,0,0), (0,0,1/6), (0,0,2/6), (0,0,3/6), (0,0,4/6), (0,0,5/6)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N6-H1', 'Hexagonal', [(0,0,0), (0,0,1/6), (0,0,2/6), (0,0,3/6), (0,0,4/6), (0,0,5/6)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N6-H2', 'Hexagonal', [(0,0,0), (1/3,2/3,0), (2/3,1/3,0), (0,0,0.5), (1/3,2/3,0.5), (2/3,1/3,0.5)], 'a, c/a', 'Two triangular layers'),
        LatticeConfig('N6-O1', 'Orthorhombic', [(0,0,0), (0,0,1/6), (0,0,2/6), (0,0,3/6), (0,0,4/6), (0,0,5/6)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N6-R1', 'Rhombohedral', [(0,0,0), (0,0,1/6), (0,0,2/6), (0,0,3/6), (0,0,4/6), (0,0,5/6)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N6-M1', 'Monoclinic', [(0,0,0), (0,1/6,0), (0,2/6,0), (0,3/6,0), (0,4/6,0), (0,5/6,0)], 'a, b/a, c/a, β', 'Equal b-stack'),
    ],
    'arity1': []
}

N7_CONFIGS = {
    'arity0': [
        LatticeConfig('N7-C1', 'Cubic', [(i/7,i/7,i/7) for i in range(7)], 'a', '[111] sevenths'),
        LatticeConfig('N7-T1', 'Tetragonal', [(0,0,i/7) for i in range(7)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N7-H1', 'Hexagonal', [(0,0,i/7) for i in range(7)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N7-O1', 'Orthorhombic', [(0,0,i/7) for i in range(7)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N7-R1', 'Rhombohedral', [(0,0,i/7) for i in range(7)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N7-M1', 'Monoclinic', [(0,i/7,0) for i in range(7)], 'a, b/a, c/a, β', 'Equal b-stack'),
    ],
    'arity1': []
}

N8_CONFIGS = {
    'arity0': [
        LatticeConfig('N8-C1', 'Cubic', [(0,0,0), (0.5,0,0), (0,0.5,0), (0,0,0.5), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5), (0.5,0.5,0.5)], 'a', 'Complete cubic set'),
        LatticeConfig('N8-C2', 'Cubic', [(i/8,i/8,i/8) for i in range(8)], 'a', '[111] eighths'),
        LatticeConfig('N8-T1', 'Tetragonal', [(0,0,i/8) for i in range(8)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N8-H1', 'Hexagonal', [(0,0,i/8) for i in range(8)], 'a, c/a', 'Equal c-stack'),
        LatticeConfig('N8-O1', 'Orthorhombic', [(0,0,i/8) for i in range(8)], 'a, b/a, c/a', 'Equal c-stack'),
        LatticeConfig('N8-R1', 'Rhombohedral', [(0,0,i/8) for i in range(8)], 'a, α', 'Equal c-stack'),
        LatticeConfig('N8-M1', 'Monoclinic', [(0,i/8,0) for i in range(8)], 'a, b/a, c/a, β', 'Equal b-stack'),
    ],
    'arity1': []
}

# Complete catalogue
LATTICE_CONFIGS = {
    1: N1_CONFIGS,
    2: N2_CONFIGS,
    3: N3_CONFIGS,
    4: N4_CONFIGS,
    5: N5_CONFIGS,
    6: N6_CONFIGS,
    7: N7_CONFIGS,
    8: N8_CONFIGS,
}


def get_configs_for_n(n: int, lattice_filter: str = 'all', arity_filter: str = 'all') -> dict:
    """Get filtered configurations for a given N."""
    if n not in LATTICE_CONFIGS:
        return {'arity0': [], 'arity1': []}
    
    configs = LATTICE_CONFIGS[n]
    arity0 = configs['arity0']
    arity1 = configs['arity1']
    
    if lattice_filter != 'all':
        arity0 = [c for c in arity0 if c.lattice == lattice_filter]
        arity1 = [c for c in arity1 if c.lattice == lattice_filter]
    
    if arity_filter == '0':
        arity1 = []
    elif arity_filter == '1':
        arity0 = []
    
    return {'arity0': arity0, 'arity1': arity1}


def get_all_lattice_types() -> list[str]:
    """Get list of all lattice types."""
    return ['Cubic', 'Tetragonal', 'Hexagonal', 'Orthorhombic', 'Rhombohedral', 'Monoclinic']


def get_all_bravais_types() -> dict[str, list[str]]:
    """Get dict of lattice type -> list of Bravais centering types."""
    return {
        'Cubic': ['cubic_P', 'cubic_I', 'cubic_F'],
        'Tetragonal': ['tetragonal_P', 'tetragonal_I'],
        'Orthorhombic': ['orthorhombic_P', 'orthorhombic_I', 'orthorhombic_F', 'orthorhombic_C'],
        'Hexagonal': ['hexagonal_P', 'hexagonal_H'],
        'Rhombohedral': ['rhombohedral_P'],
        'Monoclinic': ['monoclinic_P', 'monoclinic_C'],
    }


BRAVAIS_LABELS = {
    'cubic_P': 'cP (Primitive)',
    'cubic_I': 'cI (BCC)',
    'cubic_F': 'cF (FCC)',
    'tetragonal_P': 'tP (Primitive)',
    'tetragonal_I': 'tI (Body-centered)',
    'orthorhombic_P': 'oP (Primitive)',
    'orthorhombic_I': 'oI (Body-centered)',
    'orthorhombic_F': 'oF (Face-centered)',
    'orthorhombic_C': 'oC (C-centered)',
    'hexagonal_P': 'hP (Primitive)',
    'hexagonal_H': 'hR (HCP)',
    'rhombohedral_P': 'rP (Primitive)',
    'monoclinic_P': 'mP (Primitive)',
    'monoclinic_C': 'mC (C-centered)',
}


LATTICE_COLORS = {
    'Cubic': '#FEE2E2',        # red-100
    'Tetragonal': '#FFEDD5',   # orange-100
    'Hexagonal': '#FEF9C3',    # yellow-100
    'Orthorhombic': '#DCFCE7', # green-100
    'Rhombohedral': '#DBEAFE', # blue-100
    'Monoclinic': '#F3E8FF',   # purple-100
}
