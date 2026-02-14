# Medical Surface Reconstruction: MC vs IMC

**Reimplementation of Improved Marching Cubes for Brain Tumor 3D Reconstruction**

*Based on: "3D reconstruction of brain tumors from 2D MRI scans: An improved marching cube algorithm" (Mittal et al., 2024)*

## Overview

Brain tumors represent a serious pathology with over 320,000 new cases worldwide annually. Obtaining reliable estimation of tumor volume and geometry is essential for staging, surgical planning, and treatment follow-up. While MRI provides 2D slices sampling a 3D volume, mentally visualizing complex tumor shapes remains challenging even for experienced clinicians.

This project provides a faithful and more detailled reimplementation of the Improved Marching Cubes (IMC) algorithm for 3D tumor surface reconstruction from medical imaging segmentation masks. The implementation follows the methodology described by Mittal et al. (2024) and systematically evaluates different mesh tightening configurations.

### Implemented Algorithms

1. **MC (Marching Cubes)** - Reference implementation using scikit-image's Lewiner variant
2. **IMC0 (Pure IMC)** - Paper-faithful implementation with:
   - Midpoint edge positioning (§4.4.1) instead of linear interpolation
   - Global edge vertex caching (§4.4.2) for topology preservation
   - Paul Bourke's canonical lookup tables
3. **IMCdef (Default Tightening)** - Moderate mesh simplification (~3.5× reduction)
4. **IMCext (Extreme Tightening)** - Aggressive simplification (~20× reduction)

## Key Features

### Surface Reconstruction
- **Paper-faithful IMC implementation** with exact midpoint positioning (no scalar interpolation)
- **Mesh tightening via edge collapse** (§4.4.3) with 2/3 rule:
  - Maximum Angle Criterion (MAC): normal angle ≤ threshold
  - Maximum Distance Criterion (MDC, Eq. 30): custom distance metric
  - Boundary preservation: non-boundary vertices only
- **Multiple tightening configurations**: moderate (IMCdef) and aggressive (IMCext)
- **Topology-aware**: Union-Find for consistent vertex clustering

### Evaluation Metrics
- **Geometric measures**: vertex/face counts, surface area, volume (mm³)
- **Surface distances**: Symmetric Hausdorff Distance (HD) and Mean Surface Distance (MSD)
- **Robust point sampling**: Custom Poisson-disk sampling (grid-based, no deprecated dependencies)
- **Physical units**: All measurements in millimeters, preserving voxel spacing

### Technical Highlights
- Compatible with **Paul Bourke's canonical Marching Cubes tables**
- **Global edge caching** for vertex sharing between adjacent cubes
- **Comparative visualization** with PyVista (2×2 grid layout)
- **Medical imaging support** via NiBabel (NIfTI format)

## Requirements

```bash
pip install numpy nibabel scikit-image scipy pyvista trimesh
```

### Dependencies

- `numpy` - Array operations
- `nibabel` - NIfTI medical image I/O
- `scikit-image` - Reference Marching Cubes (Lewiner)
- `scipy` - KD-tree for distance computation
- `pyvista` - 3D visualization
- `trimesh` - Mesh utilities (area, volume, components)

## Usage

### Quick Start

1. **Edit file paths** in `surface_reconstruction.py`:

```python
#specify your file paths here (modify these):
brain_nii_path = ROOT / "UCSF-PDGM-0009_nifti" / "UCSF-PDGM-0009_brain_parenchyma_segmentation.nii.gz"
tumor_nii_path = ROOT / "UCSF-PDGM-0009_nifti" / "UCSF-PDGM-0009_tumor_segmentation.nii.gz"
```

2. **Run the script**:

```bash
python surface_reconstruction.py
```

**Automatic Fallback**: If the specified files don't exist, the script automatically searches for the first available valid subject in your dataset. This means you can run the script immediately without modification - it will find and process any available data.

### What the Script Does

- Generates MC and IMC surfaces with default/extreme tightening
- Computes statistics (vertices, faces, area, volume) 
- Calculates distance metrics (Hausdorff Distance, Mean Surface Distance)
- Displays 2×2 comparative visualization

### Programmatic Usage

For custom processing pipelines:

```python
from pathlib import Path
from surface_reconstruction import (
    mc_surface_skimage,
    imc_surface,
    tighten_mesh_IMC,
    surface_distances,
    load_binary_zooms
)

# Load binary segmentation mask
binary_vol, spacing = load_binary_zooms(Path("brain_mask.nii.gz"))

# Generate surfaces
V_mc, F_mc = mc_surface_skimage(binary_vol, spacing)
V_imc, F_imc = imc_surface(binary_vol, spacing)

# Optimize with tightening
V_opt, F_opt = tighten_mesh_IMC(
    V_imc, F_imc,
    MAC_deg=20.0,           # Maximum normal angle (degrees)
    MDC_mm=1.2 * min(spacing),  # Maximum edge distance (mm)
    max_passes=2
)

# Compare meshes
distances = surface_distances(V_mc, F_mc, V_opt, F_opt)
print(f"Hausdorff Distance: {distances['hd']:.3f} mm")
print(f"Mean Surface Distance: {distances['msd']:.3f} mm")
```

## Dataset Structure

Expected UCSF 3D Brain MRI dataset structure:

```
data/3d-brain-mri_repo/extracted/
├── subject_001_nifti/
│   ├── *brain*parenchyma*segmentation*.nii.gz
│   ├── *tumor*segmentation*.nii.gz
│   └── ...
├── subject_002_nifti/
│   └── ...
```

The code automatically searches for:
- Brain masks: `*brain*parenchyma*.nii*`
- Tumor masks: `*tumor*segmentation*.nii*`, `*enhancing*.nii*`, etc.

## Configuration

Edit constants at the top of `surface_reconstruction.py`:

```python
#surface distance sampling
N_SAMPLES = 50_000        #points for Hausdorff/MSD computation
SAMPLE_METHOD = "poisson" #"poisson" | "area" | "even"
POISSON_RELAX = 0.90      #radius relaxation factor (<1.0)
RANDOM_SEED = 42          #reproducibility

#tightening parameters (in __main__)
#moderate configuration (IMCdef):
MAC_def = 20.0            #max angle: 20° (normal angle threshold)
MDC_def = 1.2 * voxel_size  #max distance: 1.2 voxels (Eq. 30)
max_passes = 2            #optimization iterations

#aggressive configuration (IMCext):
MAC_ext = 45.0            #max angle: 45°
MDC_ext = 2.0 * voxel_size  #max distance: 2 voxels
max_passes = 6
```

### Tightening Parameters Guide

**MAC (Maximum Angle Criterion)**:
- Controls smoothness: lower = preserve fine details, higher = more smoothing
- Recommended: 15-20° (conservative), 30-45° (aggressive)

**MDC (Maximum Distance Criterion, Eq. 30)**:
- Physical distance threshold in mm: `d = 0.5 * (||vi - vj||₁ + ||vi - vj||∞)`
- Recommended: 1.0-1.5× voxel size (conservative), 2-3× (aggressive)

**max_passes**:
- Number of iterative edge collapse passes
- Recommended: 2-3 (moderate), 5-8 (aggressive)

## Typical Results

### Mesh Complexity Reduction (UCSF-PDGM Dataset, 30 subjects average)

**Brain Parenchyma:**
| Method | Vertices | Faces | Area Reduction |
|--------|----------|-------|----------------|
| MC (reference) | ~800K | ~1.6M | baseline |
| IMC0 | 0.98× | 0.93× | 0.96× |
| IMCdef | 0.29× | 0.28× | 0.75× |
| IMCext | 0.05× | 0.05× | 0.54× |

**Tumor:**
| Method | Vertices | Faces | Area Reduction |
|--------|----------|-------|----------------|
| MC (reference) | ~11K | ~22K | baseline |
| IMC0 | 1.00× | 1.00× | 1.00× |
| IMCdef | 0.29× | 0.29× | 0.91× |
| IMCext | 0.03× | 0.03× | 0.81× |

### Surface Distance to MC Reference

**Brain:**
- IMC0: HD = 6.79 mm, MSD = 1.22 mm (~1 voxel)
- IMCdef: HD = 11.27 mm, MSD = 1.22 mm
- IMCext: HD = 17.71 mm, MSD = 1.34 mm

**Tumor:**
- IMC0: HD = 0.63 mm, MSD = 0.18 mm
- IMCdef: HD = 1.48 mm, MSD = 0.21 mm
- IMCext: HD = 2.98 mm, MSD = 0.51 mm

### Key Observations

1. **IMC0 vs MC**: Near-identical geometry (MSD ≈ 1 voxel), slight complexity reduction (~2-7%)
2. **IMCdef**: Excellent compromise — 3.5× reduction with MSD unchanged (~1.2 mm for brain)
3. **IMCext**: Extreme simplification (20-40× reduction) with moderate geometric impact
4. **Topology**: Watertightness and component count stable across variants

*See Section 3.2 of the full report for detailed analysis.*

## Algorithm Details

### Marching Cubes Conventions

- **Cube indexing**: 8 corners → 256 configurations (0-255)
- **Edge numbering**: 12 edges per cube (Paul Bourke convention)
- **Lookup tables**: `edgeTable` (256,) and `triTable` (256, 16)

### IMC Implementation

**Vertex positioning (§4.4.1)**:
- Midpoint of edge endpoints (no scalar field interpolation)
- World coordinates: `(x + m[i]) * spacing[i]`

**Vertex sharing (§4.4.2)**:
- Global edge cache: `((x0,y0,z0), (x1,y1,z1)) → vertex_index`
- Ordered tuples ensure adjacent cubes reuse vertices

### Mesh Tightening (§4.4.3)

**2/3 Rule**: Edge collapse occurs if ≥2 criteria satisfied:
1. **Normal constraint**: `angle(n_i, n_j) ≤ MAC_deg`
2. **Distance constraint**: `d(v_i, v_j) ≤ MDC_mm` (Eq. 30)
3. **Topology constraint**: Both vertices non-boundary

**Distance metric (Eq. 30)**:
```python
d = 0.5 * (||v_i - v_j||_1 + ||v_i - v_j||_∞)
```

**Greedy strategy**: Process edges shortest-first to minimize distortion

### Distance Metrics

**Symmetric Hausdorff Distance**:
```
HD(A, B) = max(max_a min_b d(a,b), max_b min_a d(b,a))
```

**Mean Surface Distance**:
```
MSD(A, B) = mean([d(a, B) for a in A] + [d(b, A) for b in B])
```

Computed via:
- Poisson-disk surface sampling (50k points/mesh)
- KD-tree nearest-neighbor queries

## Output

### Statistics

For each mesh (MC, IMC0, IMCdef, IMCext):
- Vertex/face counts
- Surface area (mm²)
- Volume (mm³, if watertight)
- Number of connected components

### Distance Metrics

Compared against MC reference:
- Hausdorff Distance (HD, mm)
- Mean Surface Distance (MSD, mm)

### Visualization

2×2 PyVista window:
- **Top-left**: MC (reference)
- **Top-right**: IMC0 (pure, no tightening)
- **Bottom-left**: IMCdef (default tightening)
- **Bottom-right**: IMCext (extreme tightening)

Gray = brain, Red = tumor

## Performance

### Computation Time (UCSF-PDGM, ~256³ volume)

| Algorithm | Brain | Tumor | Notes |
|-----------|-------|-------|-------|
| MC (Lewiner) | 0.54s | 0.11s | Optimized C implementation (scikit-image) |
| IMC0 (Python) | 19.60s | 0.37s | Pure Python implementation |

**Runtime Notes**:
- The Python IMC is slower than the optimized C implementation of MC
- This reflects implementation differences, not algorithmic inefficiency
- A multithreaded C++ version would be comparable to MC
- Tightening adds ~0.5-2s depending on passes and constraint strictness

### Memory Usage

- Typical brain mesh (IMC0): ~800K vertices → ~25 MB (float32)
- After IMCdef tightening: ~240K vertices → ~7.5 MB (3.3× reduction)
- After IMCext tightening: ~40K vertices → ~1.2 MB (20× reduction)

### Scalability

The grid-based Poisson sampling scales well:
- 50K points from 800K vertices: ~2-3 seconds
- KD-tree distance computation (50K × 50K): ~1-2 seconds
- Total distance metrics: ~5-8 seconds per comparison

## Limitations and Future Work

### Current Limitations

1. **Runtime Performance**: Python implementation is slower than optimized C (MC in scikit-image)
   - Solution: Migrate core loops to C++/Cython or use Numba JIT compilation

2. **Topology Sensitivity**: Number of components varies with mask noise
   - Solution: Pre-smooth masks or remove small components (<volume threshold)

3. **Single-threaded**: No parallelization of cube processing
   - Solution: Parallelize cube iteration with shared edge cache (thread-safe)

4. **Memory for Large Volumes**: Global vertex cache grows with mesh size
   - Solution: Spatial hashing or octree-based cache eviction

### Future Directions

**Algorithm Enhancements:**
- Adaptive tightening: vary MAC/MDC based on local curvature
- Quadric Error Metrics (QEM) for edge collapse priority
- Dual Marching Cubes for better sharp feature preservation
- GPU acceleration of cube processing and tightening

**Application Extensions:**
- Multi-label reconstruction (tumor core, edema, necrosis)
- Integration with segmentation networks (end-to-end pipeline)
- Temporal tracking: surface evolution across multiple MRI sessions
- Export to clinical formats (DICOM-RT, STL for 3D printing)

**Evaluation:**
- Benchmark on larger datasets (BraTS, UCSF-PDGM full cohort)
- Clinical validation: surgeon feedback on surface quality
- Comparison with implicit surface methods (level sets, neural surface reconstruction)

## Implementation Notes

### Poisson-Disk Sampling

Custom grid-based implementation (no deprecated `trimesh.sample.sample_surface_even`):
1. Oversample surface (6× target)
2. Grid-based dart throwing with 3×3×3 neighborhood checks
3. Top-up with area-weighted sampling if needed

### Union-Find Clustering

Efficient vertex merging with:
- Path compression in `find()`
- Union by rank
- O(α(n)) amortized complexity

### Boundary Detection

Boundary vertices identified by edges with `count == 1` (appear in single face).

## References

### Primary Reference

**Mittal R, Malik V, Singla G, Kaur A, Singh M, Mittal A.** (2024).  
*3D reconstruction of brain tumors from 2D MRI scans: An improved marching cube algorithm.*  
Biomedical Signal Processing and Control, Volume 91, 105901.  
https://doi.org/10.1016/j.bspc.2023.105901

### Datasets

- **UCSF-PDGM**: Pre-operative diffuse glioma MRI cohort (87 subjects)  
  Hugging Face: https://huggingface.co/datasets/determined-ai/3d-brain-mri  
  TCIA: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/

- **MSD Task01 BrainTumour**: Medical Segmentation Decathlon  
  Antonelli M, et al. (2022). Nature Communications, 13:4128  
  https://medicaldecathlon.com

### Algorithmic Background

- **Marching Cubes**: Lorensen WE, Cline HE (1987). SIGGRAPH  
- **Paul Bourke Tables**: https://paulbourke.net/geometry/polygonise/  
- **Lewiner MC**: Lewiner T, et al. (2003). *Efficient Implementation of Marching Cubes*

### Tools and Libraries

- **nibabel**: NIfTI medical image I/O - https://nipy.org/nibabel/
- **scikit-image**: Image processing (Marching Cubes reference) - https://scikit-image.org/
- **trimesh**: Mesh processing utilities - https://trimsh.org/
- **PyVista**: 3D visualization - https://docs.pyvista.org/

### Medical Context

- **Global Cancer Statistics**: Ferlay J, et al. (2024). IARC Global Cancer Observatory  
  https://gco.iarc.who.int/today
- **Canadian Cancer Society**: Brain tumor statistics  
  https://cancer.ca/en/cancer-information/cancer-types/brain-and-spinal-cord/statistics

This is an academic reimplementation for educational and research purposes.

**Data Usage**:
- UCSF-PDGM dataset: See TCIA data usage policy
- MSD Task01: CC BY-SA 4.0 license

## Acknowledgments

- **Original paper authors**: Mittal R, Malik V, Singla G, et al.
- **Dataset providers**: UCSF-PDGM team, Medical Segmentation Decathlon
- **Paul Bourke**: Canonical Marching Cubes tables and documentation
- **Open-source tools**: nibabel, scikit-image, trimesh, PyVista communities

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{mittal2024imc,
  title={3D reconstruction of brain tumors from 2D MRI scans: An improved marching cube algorithm},
  author={Mittal, Ruchi and Malik, Varun and Singla, Geetanjali and Kaur, Amandeep and Singh, Manjinder and Mittal, Amit},
  journal={Biomedical Signal Processing and Control},
  volume={91},
  pages={105901},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.bspc.2023.105901}
}
```

## Academic Context

**Course**: GBM6700E - 3D Reconstruction from Medical Images  
**Institution**: Polytechnique Montréal  
**Professor**: Lama Séoud  
**Student**: Yassine Ben Ammar 

This project was developed as part of a graduate-level medical imaging course, focusing on faithful reimplementation and quantitative evaluation of surface reconstruction algorithms for clinical neuroimaging applications.
