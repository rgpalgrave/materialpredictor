# Crystal Coordination Calculator & Lattice Explorer

A Streamlit application for crystallographic analysis that calculates:
- **Stoichiometry** from metal ratios and charges
- **Anion coordination number** from charge balance and metal CNs
- **Minimum scale factors** to achieve target intersection multiplicities
- **Coordination environment regularity** for assessing polyhedron geometry
- **Madelung energy** for electrostatic stability analysis

## Physical Model

The app uses a sphere intersection model where each metal cation is surrounded by a coordination sphere of radius:

```
r = s × (r_cation + r_anion)
```

Where:
- `r_cation` and `r_anion` are Shannon ionic radii in Ångströms
- `s` is a scale factor (typically close to 1 for real structures)

Anion positions are found at the intersection points where multiple spheres meet. The **multiplicity** of an intersection is the number of spheres that meet there, which corresponds to the coordination number of the anion.

## Features

### 1. Streamlined Workflow
Click **"Calculate Stoichiometry & CN"** to run the complete analysis chain:
1. Calculate stoichiometry and target coordination number
2. Find minimum scale factors for all lattice configurations
3. Calculate stoichiometries for each successful configuration
4. Match results: exact matches (✓), half-filling matches (½), or no match
5. Analyze coordination polyhedron regularity for exact matches
6. Generate 3D previews for exact matches

Results are displayed in a unified section showing:
- Summary metrics (total configs, exact matches, half-filling matches)
- Detailed cards for each exact match with regularity scores and 3D preview
- Tables for half-filling and other configurations

### 2. Composition Calculator
- Input multiple metal cations with their symbols, charges, ratios, and coordination numbers
- Shannon ionic radii database (1976) with auto-lookup
- Automatic charge balancing and stoichiometry calculation
- Anion CN derived from metal-anion coordination consistency

### 3. Lattice Configuration Catalogue
- Complete catalogue of metal sublattice configurations for N=1 through N=8
- Covers all 6 Bravais lattice systems (Cubic, Tetragonal, Hexagonal, Orthorhombic, Rhombohedral, Monoclinic)
- Both fixed (arity-0) and parametric (arity-1) configurations
- Filterable by lattice type and arity

### 4. Madelung Energy Calculation (NEW)
- Calculate approximate electrostatic (Madelung) energy
- Uses direct Coulomb summation over a supercell
- Input experimental lattice parameters for accurate results
- Returns Madelung constant, energy per formula unit, and nearest-neighbor distances
- Reference values for common structure types (rocksalt: 1.748, fluorite: 2.519, etc.)

### 5. Advanced Manual Controls
For users who want more control, an expandable "Advanced" section provides:
- Manual scale factor calculation
- c/a ratio optimization for tetragonal/hexagonal/orthorhombic lattices
- Stoichiometry-based c/a scanning

### 6. Coordination Environment Analysis
- Analyzes the regularity of coordination polyhedra around each metal type
- Uses periodic boundary conditions to find nearest intersection sites
- Calculates distance metrics: mean, std deviation, coefficient of variation
- Calculates angular metrics: comparison to ideal polyhedra (tetrahedron, octahedron, cube, etc.)
- Provides regularity scores (0-1) for distance uniformity and angular regularity
- Uses each metal's specific coordination number (not a single global value)

### 7. Integrated Half-Filling Analysis
For structures where only half the anion sites are occupied (zinc blende from fluorite, wurtzite, anti-fluorite, etc.):
- Automatically detected as "Half-Filling Matches" in the workflow
- Optimization algorithm finds which sites to remove for maximum coordination regularity
- Shows regularity scores before and after half-filling optimization
- 3D preview distinguishes kept sites (red) from removed sites (gray)
- Per-metal coordination details showing achieved CN and regularity
- Typical regularity improvement: 10-30% from unoptimized half-filling

**Example**: CaF₂ (fluorite) → CaF (half-filled)
- Original: 8 tetrahedral F sites per Ca
- Half-filled: 4 tetrahedral F sites, optimally selected
- Regularity improves from ~0.57 to ~0.70

## Local Development

### Installation

```bash
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run app.py
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Deployment to Google Cloud Run

### Option 1: GitHub Actions (Recommended)

1. **Set up GitHub Secrets:**
   - `GCP_PROJECT_ID`: Your Google Cloud project ID
   - `WIF_PROVIDER`: Workload Identity Federation provider
   - `WIF_SERVICE_ACCOUNT`: Service account for deployment

2. **Enable required APIs in GCP:**
   ```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     artifactregistry.googleapis.com
   ```

3. **Create Artifact Registry repository:**
   ```bash
   gcloud artifacts repositories create cloud-run-source-deploy \
     --repository-format=docker \
     --location=us-central1
   ```

4. **Push to main branch** - deployment happens automatically

### Option 2: Manual Cloud Build

```bash
# Submit build
gcloud builds submit --config=cloudbuild.yaml

# Or use gcloud run deploy directly
gcloud run deploy coordination-calculator \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

### Option 3: Docker Build & Push

```bash
# Build locally
docker build -t coordination-calculator .

# Tag for Artifact Registry
docker tag coordination-calculator \
  us-central1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy/coordination-calculator

# Push
docker push us-central1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy/coordination-calculator

# Deploy
gcloud run deploy coordination-calculator \
  --image us-central1-docker.pkg.dev/PROJECT_ID/cloud-run-source-deploy/coordination-calculator \
  --region us-central1 \
  --allow-unauthenticated
```

## Project Structure

```
coordination_app/
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions CI/CD
├── .streamlit/
│   └── config.toml           # Streamlit configuration
├── tests/
│   ├── __init__.py
│   └── test_modules.py       # Unit tests
├── app.py                    # Main Streamlit application
├── ionic_radii.py            # Shannon ionic radii database
├── lattice_configs.py        # Lattice configuration catalogue
├── interstitial_engine.py    # Sphere intersection engine
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
├── .dockerignore             # Docker ignore rules
├── .gitignore                # Git ignore rules
├── cloudbuild.yaml           # Cloud Build configuration
├── service.yaml              # Cloud Run service spec
└── README.md                 # This file
```

## Scientific Background

### Sphere Intersection Method
The algorithm places coordination spheres on each metal position with radius r = α × s × a, where:
- α = alpha ratio (typically 0.5)
- s = scale factor (dimensionless)
- a = lattice parameter

An N-fold intersection occurs where N or more sphere surfaces meet at a point. The minimum scale factor s* is the smallest s value that achieves intersections of order ≥ N.

### Shannon Ionic Radii
Radii data from Shannon, R.D. (1976) "Revised effective ionic radii and systematic studies of interatomic distances in halides and chalcogenides" Acta Cryst. A32, 751-767.

## Example Results

For FCC motif (N4-C1) with α=0.5:
| Target CN | s* |
|-----------|------|
| 2 | 0.7071 |
| 4 | 0.8658 |
| 6 | 1.0000 |
| 8 | 1.3742 |
| 12 | 1.4140 |

## License

MIT License
