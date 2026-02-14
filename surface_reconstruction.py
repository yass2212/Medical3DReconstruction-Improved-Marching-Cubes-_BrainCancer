"""
Medical Surface Reconstruction: Marching Cubes vs Interpolated Marching Cubes
=============================================================================

This module implements and compares three surface reconstruction algorithms for
medical imaging data (brain and tumor segmentation masks):

1. MC (Marching Cubes) - Reference implementation using scikit-image
2. IMC (Interpolated Marching Cubes) - Paper-faithful implementation
3. IMC with tightening - Edge collapse optimization variants

The implementation includes:
- Canonical Marching Cubes tables (Paul Bourke convention)
- Edge caching for vertex sharing (§4.4.2)
- Midpoint positioning (§4.4.1)
- Mesh tightening via edge collapse (§4.4.3)
- Robust surface distance metrics (Hausdorff, MSD)
- Custom Poisson-disk sampling for point clouds

Dataset: UCSF 3D Brain MRI
"""

from __future__ import annotations
from pathlib import Path
import time
import numpy as np
import nibabel as nib
from skimage import measure
from scipy.spatial import cKDTree
import pyvista as pv
import trimesh

#import canonical Marching Cubes lookup tables (Paul Bourke convention)
from mc_tables import edgeTable as _edgeTable, triTable as _triTable

edgeTable = np.asarray(_edgeTable, dtype=np.int32)
triTable = np.asarray(_triTable, dtype=np.int32)
assert edgeTable.shape == (256,), "Edge table must have 256 entries (one per cube config)"
assert triTable.shape == (256, 16), "Triangle table must be 256x16"

#configuration: Surface Distance Sampling

N_SAMPLES = 50_000  #number of points for Hausdorff/MSD computation
SAMPLE_METHOD = "poisson"  #options: "poisson" | "area" | "even"
POISSON_RELAX = 0.90  #relax factor (<1.0) for Poisson radius
RANDOM_SEED = 42  #reproducibility

#reduce trimesh verbosity
trimesh.util.attach_to_log(level="ERROR")


#dataset Utilities (UCSF Brain MRI)

def _is_nifti(p: Path) -> bool:
    """Check if path is a NIfTI file (.nii or .nii.gz)"""
    s = p.name.lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")


def find_subject_dirs(root: Path) -> list[Path]:
    """Find all subject directories (ending with '_nifti')"""
    return sorted([p for p in root.glob("*_nifti") if p.is_dir()])


def find_brain_mask(sd: Path) -> Path | None:
    """
    Locate brain parenchyma segmentation mask in subject directory.
    
    Args:
        sd: Subject directory path
        
    Returns:
        Path to brain mask or None if not found
    """
    patterns = [
        "*brain*parenchyma*segmentation*.nii*",
        "*brain_parenchyma*.nii*",
        "*brain_segmentation*.nii*"
    ]
    for pat in patterns:
        candidates = [p for p in sd.glob(pat) if _is_nifti(p)]
        if candidates:
            return candidates[0]
    return None


def find_tumor_masks(sd: Path) -> list[Path]:
    """
    Find all tumor/lesion segmentation masks in subject directory.
    
    Searches for various tumor annotations: whole tumor, tumor core,
    enhancing, non-enhancing, necrosis, edema, FLAIR abnormality.
    
    Args:
        sd: Subject directory path
        
    Returns:
        List of unique tumor mask paths
    """
    patterns = [
        "*tumor*segmentation*.nii*",
        "*lesion*seg*.nii*",
        "*enhancing*segmentation*.nii*",
        "*non*enhancing*segmentation*.nii*",
        "*nonenhancing*segmentation*.nii*",
        "*necrosis*segmentation*.nii*",
        "*edema*segmentation*.nii*",
        "*flair*abnormality*segmentation*.nii*",
        "*whole_tumor*segmentation*.nii*",
        "*tumor_core*segmentation*.nii*",
    ]
    out, seen = [], set()
    for pat in patterns:
        for p in sd.glob(pat):
            if _is_nifti(p) and p not in seen:
                seen.add(p)
                out.append(p)
    return out


def pick_subject(root: Path) -> Path:
    """
    Select first valid subject with both brain and tumor masks.
    
    Args:
        root: Dataset root directory
        
    Returns:
        Subject directory path
        
    Raises:
        FileNotFoundError: If no valid subject found
    """
    for sd in find_subject_dirs(root):
        if find_brain_mask(sd) is not None and len(find_tumor_masks(sd)) > 0:
            return sd
    raise FileNotFoundError("No subject with brain + tumor masks found in ROOT.")


def load_binary_zooms(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load NIfTI mask and extract voxel spacing.
    
    Args:
        path: Path to NIfTI file
        
    Returns:
        Binary volume (uint8) and voxel spacing (mm) as (x, y, z)
    """
    img = nib.load(str(path))
    arr = img.get_fdata()
    zooms = img.header.get_zooms()[:3]
    binary_vol = (arr > 0).astype(np.uint8)
    return binary_vol, (float(zooms[0]), float(zooms[1]), float(zooms[2]))



#marching Cubes Convention (Paul Bourke)

#cube corner offsets (8 vertices)
corner_offsets = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  #bottom face (z=0)
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   #top face (z=1)
], dtype=np.int32)

#edge definitions (12 edges, each defined by 2 corner indices)
edge_to_corners = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0],  #bottom face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  #top face edges
    [0, 4], [1, 5], [2, 6], [3, 7]   #vertical edges
], dtype=np.int32)


#reference Implementation: Marching Cubes (Lewiner)


def mc_surface_skimage(binary_volume: np.ndarray, 
                       spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate surface mesh using scikit-image's Marching Cubes (Lewiner variant).
    
    This serves as the reference implementation for comparison.
    
    Args:
        binary_volume: Binary segmentation mask (0 or 1)
        spacing: Voxel dimensions in mm (x, y, z)
        
    Returns:
        Vertices (Nx3 float32) and faces (Mx3 int64)
    """
    vol = (binary_volume > 0).astype(np.uint8)
    V, F, _, _ = measure.marching_cubes(
        vol, level=0.5, spacing=spacing, allow_degenerate=False
    )
    return V.astype(np.float32), F.astype(np.int64)



#interpolated Marching Cubes (IMC) - Paper-Faithful Implementation


def _edge_midpoint_world(x: int, y: int, z: int, e: int, 
                         spacing: tuple[float, float, float]) -> np.ndarray:
    """
    Compute world coordinates of edge midpoint (§4.4.1).
    
    Unlike interpolated MC, this uses exact midpoint positioning
    without interpolation from scalar field values.
    
    Args:
        x, y, z: Cube grid coordinates
        e: Edge index (0-11)
        spacing: Voxel dimensions in mm
        
    Returns:
        3D point in world coordinates (mm)
    """
    #get edge endpoints in local cube coordinates
    a, b = edge_to_corners[e]
    v0 = corner_offsets[a].astype(np.float32)
    v1 = corner_offsets[b].astype(np.float32)
    
    #midpoint in local coordinates
    m = 0.5 * (v0 + v1)
    
    #transform to world coordinates
    sx, sy, sz = spacing
    return np.array([
        (x + m[0]) * sx,
        (y + m[1]) * sy,
        (z + m[2]) * sz
    ], dtype=np.float32)


def _edge_global_key(x: int, y: int, z: int, e: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    Generate unique global key for edge vertex caching (§4.4.2).
    
    Ensures same edge shared by adjacent cubes reuses the same vertex.
    Key is ordered pair of grid coordinates of edge endpoints.
    
    Args:
        x, y, z: Cube grid coordinates
        e: Edge index (0-11)
        
    Returns:
        Canonical ordered tuple ((x0,y0,z0), (x1,y1,z1))
    """
    a, b = edge_to_corners[e]
    p0 = (
        x + int(corner_offsets[a, 0]),
        y + int(corner_offsets[a, 1]),
        z + int(corner_offsets[a, 2])
    )
    p1 = (
        x + int(corner_offsets[b, 0]),
        y + int(corner_offsets[b, 1]),
        z + int(corner_offsets[b, 2])
    )
    #return ordered pair for consistency
    return (p0, p1) if p0 <= p1 else (p1, p0)


def _cube_index(vol: np.ndarray, x: int, y: int, z: int) -> int:
    """
    Compute cube configuration index (0-255) based on corner occupancy.
    
    Each of 8 corners contributes a bit to the index.
    
    Args:
        vol: Binary volume
        x, y, z: Cube origin coordinates
        
    Returns:
        Cube index (0-255)
    """
    c = 0
    corners = [
        vol[x,     y,     z    ], vol[x + 1, y,     z    ],
        vol[x + 1, y + 1, z    ], vol[x,     y + 1, z    ],
        vol[x,     y,     z + 1], vol[x + 1, y,     z + 1],
        vol[x + 1, y + 1, z + 1], vol[x,     y + 1, z + 1],
    ]
    for i, val in enumerate(corners):
        if val > 0:
            c |= (1 << i)
    return c


def imc_surface(binary_volume: np.ndarray, 
                spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate surface mesh using Interpolated Marching Cubes (IMC).
    
    Paper-faithful implementation with:
    - Midpoint edge positioning (§4.4.1)
    - Global edge vertex caching (§4.4.2)
    - Canonical lookup tables (Paul Bourke)
    
    Args:
        binary_volume: Binary segmentation mask
        spacing: Voxel dimensions in mm (x, y, z)
        
    Returns:
        Vertices (Nx3 float32) and faces (Mx3 int64)
    """
    vol = (binary_volume > 0).astype(np.uint8)
    nx, ny, nz = vol.shape
    
    #find occupied region to limit processing
    pos = np.argwhere(vol > 0)
    if pos.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int64)
    
    #compute bounding box with 1-voxel margin
    lo = np.maximum(pos.min(axis=0) - 1, 0)
    hi = np.minimum(pos.max(axis=0) + 1, np.array([nx - 1, ny - 1, nz - 1]))
    x0, y0, z0 = map(int, lo)
    x1, y1, z1 = map(int, hi)
    
    #global edge vertex cache: edge_key -> vertex_index
    vertex_cache = {}
    V, F = [], []
    
    #process each cube in bounding box
    for x in range(x0, x1):
        for y in range(y0, y1):
            for z in range(z0, z1):
                #determine cube configuration
                ci = _cube_index(vol, x, y, z)
                
                #skip empty or full cubes
                if ci == 0 or ci == 255:
                    continue
                
                #lookup triangle edges for this configuration
                tt = triTable[ci]
                end = int(np.where(tt == -1)[0][0]) if np.any(tt == -1) else 16
                
                #collect unique edges needed (preserve order)
                needed, seen = [], set()
                i = 0
                while i + 2 < end:
                    a, b, c = int(tt[i]), int(tt[i + 1]), int(tt[i + 2])
                    if a not in seen:
                        seen.add(a)
                        needed.append(a)
                    if b not in seen:
                        seen.add(b)
                        needed.append(b)
                    if c not in seen:
                        seen.add(c)
                        needed.append(c)
                    i += 3
                
                #create/lookup vertices for needed edges
                local = {}
                for e in needed:
                    key = _edge_global_key(x, y, z, e)
                    if key in vertex_cache:
                        #reuse existing vertex
                        idx = vertex_cache[key]
                    else:
                        #create new vertex at edge midpoint
                        p = _edge_midpoint_world(x, y, z, e, spacing)
                        idx = len(V)
                        V.append(p)
                        vertex_cache[key] = idx
                    local[e] = idx
                
                #generate triangles
                i = 0
                while i + 2 < end:
                    a, b, c = int(tt[i]), int(tt[i + 1]), int(tt[i + 2])
                    F.append([local[a], local[b], local[c]])
                    i += 3
    
    #convert to numpy arrays
    V = np.asarray(V, dtype=np.float32) if V else np.zeros((0, 3), np.float32)
    F = np.asarray(F, dtype=np.int64) if F else np.zeros((0, 3), np.int64)
    return V, F



#mesh Tightening (§4.4.3) - Edge Collapse Optimization


def face_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute unnormalized face normals.
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        
    Returns:
        Face normals (Mx3)
    """
    a = V[F[:, 1]] - V[F[:, 0]]
    b = V[F[:, 2]] - V[F[:, 0]]
    return np.cross(a, b)


def vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute area-weighted vertex normals.
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        
    Returns:
        Normalized vertex normals (Nx3)
    """
    nF = face_normals(V, F)
    acc = np.zeros_like(V, dtype=np.float64)
    
    #accumulate face normals to vertices
    for k in range(3):
        np.add.at(acc, F[:, k], nF)
    
    #normalize
    lens = np.linalg.norm(acc, axis=1, keepdims=True) + 1e-12
    return (acc / lens).astype(np.float32)


def edges_from_faces(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract unique edges from faces with occurrence counts.
    
    Args:
        F: Faces (Mx3)
        
    Returns:
        Unique edges (Kx2) and their counts (K,)
    """
    #extract all half-edges
    E = np.vstack([
        np.stack([F[:, 0], F[:, 1]], 1),
        np.stack([F[:, 1], F[:, 2]], 1),
        np.stack([F[:, 2], F[:, 0]], 1),
    ])
    
    #sort edge endpoints for consistency
    E = np.sort(E, axis=1)
    
    #find unique edges and counts
    view = E.view([('a', E.dtype), ('b', E.dtype)])
    uniq, counts = np.unique(view, return_counts=True)
    edges = uniq.view(E.dtype).reshape(-1, 2)
    
    return edges, counts


def boundary_vertices(F: np.ndarray, nV: int) -> np.ndarray:
    """
    Identify boundary (border) vertices.
    
    Boundary edges appear in only one face (count == 1).
    
    Args:
        F: Faces (Mx3)
        nV: Number of vertices
        
    Returns:
        Boolean array (nV,) marking boundary vertices
    """
    edges, counts = edges_from_faces(F)
    boundary_edges = edges[counts == 1]
    
    is_boundary = np.zeros(nV, dtype=bool)
    if len(boundary_edges):
        is_boundary[boundary_edges[:, 0]] = True
        is_boundary[boundary_edges[:, 1]] = True
    
    return is_boundary


def eq30_distance(vi: np.ndarray, vj: np.ndarray) -> np.ndarray:
    """
    Compute edge distance metric (Equation 30 from paper).
    
    Combines L1 and L∞ norms: d = 0.5 * (||v_i - v_j||_1 + ||v_i - v_j||_∞)
    
    Args:
        vi, vj: Vertex positions (Nx3)
        
    Returns:
        Distance values (N,)
    """
    d = np.abs(vi - vj)
    return 0.5 * (d.sum(axis=-1) + d.max(axis=-1))


class UnionFind:
    """
    Union-Find (Disjoint Set Union) data structure for vertex clustering.
    
    Supports efficient merging and root finding with path compression
    and union by rank.
    """
    
    def __init__(self, n: int):
        """
        Initialize n disjoint sets.
        
        Args:
            n: Number of elements
        """
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros(n, dtype=np.int16)
    
    def find(self, x: int) -> int:
        """
        Find root of element x with path compression.
        
        Args:
            x: Element index
            
        Returns:
            Root index
        """
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)  #path compression
        return self.parent[x]
    
    def union(self, a: int, b: int) -> int:
        """
        Merge sets containing a and b.
        
        Args:
            a, b: Element indices
            
        Returns:
            Root of merged set
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        
        #union by rank
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        
        return ra


def tighten_mesh_IMC(V: np.ndarray, F: np.ndarray, 
                     MAC_deg: float, MDC_mm: float, 
                     max_passes: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Tighten mesh via edge collapse with 2/3 rule (§4.4.3).
    
    Edges are collapsed if they satisfy at least 2 of 3 conditions:
    1. Normal angle ≤ MAC_deg (Maximum Angle Criterion)
    2. Distance ≤ MDC_mm (Maximum Distance Criterion, Eq. 30)
    3. Both endpoints are non-boundary vertices
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        MAC_deg: Maximum normal angle threshold (degrees)
        MDC_mm: Maximum edge distance threshold (mm)
        max_passes: Maximum optimization iterations
        
    Returns:
        Optimized vertices and faces
    """
    if len(V) == 0 or len(F) == 0:
        return V, F
    
    V = V.copy()
    F = F.copy()
    
    for pass_num in range(max_passes):
        #compute vertex normals
        VN = vertex_normals(V, F)
        
        #identify boundary vertices
        is_boundary = boundary_vertices(F, len(V))
        
        #extract edges
        edges, _ = edges_from_faces(F)
        if len(edges) == 0:
            break
        
        vi, vj = V[edges[:, 0]], V[edges[:, 1]]
        
        #criterion 1: Normal angle constraint
        dot = np.einsum('ij,ij->i', VN[edges[:, 0]], VN[edges[:, 1]])
        dot = np.clip(dot, -1, 1)
        angle_deg = np.degrees(np.arccos(dot))
        c_normal = (angle_deg <= MAC_deg)
        
        #criterion 2: Distance constraint (Eq. 30)
        dist = eq30_distance(vi, vj)
        c_dist = (dist <= MDC_mm)
        
        #criterion 3: Non-boundary constraint
        c_non_boundary = (~is_boundary[edges[:, 0]]) & (~is_boundary[edges[:, 1]])
        
        #2/3 rule: at least 2 criteria must be satisfied
        satisfied = (c_normal.astype(int) + c_dist.astype(int) + c_non_boundary.astype(int)) >= 2
        
        #greedy collapse: process shortest edges first
        order = np.argsort(dist)
        
        #initialize Union-Find for vertex clustering
        uf = UnionFind(len(V))
        new_pos = V.copy()
        merged = np.zeros(len(V), dtype=bool)
        merged_any = False
        
        #process edges in order
        for k in order:
            if not satisfied[k]:
                continue
            
            a, b = int(edges[k, 0]), int(edges[k, 1])
            ra, rb = uf.find(a), uf.find(b)
            
            #skip if already merged or either endpoint already processed
            if ra == rb or merged[ra] or merged[rb]:
                continue
            
            #collapse edge: midpoint of cluster representatives
            midpoint = 0.5 * (new_pos[ra] + new_pos[rb])
            root = uf.union(ra, rb)
            new_pos[root] = midpoint
            merged[root] = True
            merged_any = True
        
        #if no edges collapsed, stop
        if not merged_any:
            break
        
        #remap faces to cluster roots
        map_idx = np.arange(len(V), dtype=np.int64)
        for i in range(len(V)):
            map_idx[i] = uf.find(i)
        
        F = map_idx[F]
        
        #remove degenerate faces (collapsed to line or point)
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
        F = F[keep]
        
        #remove duplicate faces
        F_sorted = np.sort(F, axis=1)
        view = F_sorted.view([('a', F_sorted.dtype), ('b', F_sorted.dtype), ('c', F_sorted.dtype)])
        _, idx = np.unique(view, return_index=True)
        F = F[idx]
        
        #remove unused vertices and compact indexing
        used = np.zeros(len(V), dtype=bool)
        used[F.reshape(-1)] = True
        old2new = -np.ones(len(V), dtype=np.int64)
        old2new[np.where(used)[0]] = np.arange(used.sum(), dtype=np.int64)
        
        V = new_pos[used]
        F = old2new[F]
    
    return V.astype(np.float32), F.astype(np.int64)



#surface Distance Metrics (Hausdorff and Mean Surface Distance)


def build_trimesh(V: np.ndarray, F: np.ndarray) -> trimesh.Trimesh | None:
    """
    Build trimesh object from vertices and faces.
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        
    Returns:
        Trimesh object or None if empty
    """
    if len(V) == 0 or len(F) == 0:
        return None
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def _poisson_radius_for(mesh_area: float, target_n: int, relax: float = 0.9) -> float:
    """
    Compute Poisson disk radius for target point density.
    
    Assumes density ≈ 1/(π r²) and applies relaxation factor.
    
    Args:
        mesh_area: Surface area (mm²)
        target_n: Target number of points
        relax: Radius reduction factor (<1.0 admits more points)
        
    Returns:
        Poisson disk radius (mm)
    """
    r = np.sqrt(mesh_area / (target_n * np.pi))
    return float(r * relax)


def _poisson_grid_select(points: np.ndarray, radius: float, 
                         target_n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Select Poisson-disk distributed subset via 3D grid-based dart throwing.
    
    Uses voxel grid for efficient neighbor queries (27-cell neighborhood).
    
    Args:
        points: Input point cloud (Nx3)
        radius: Minimum inter-point distance
        target_n: Target number of points to select
        rng: Random number generator
        
    Returns:
        Indices of selected points
    """
    if points.shape[0] == 0 or target_n <= 0:
        return np.empty((0,), dtype=np.int64)
    
    #shuffle to avoid sequential bias
    idx_all = np.arange(points.shape[0], dtype=np.int64)
    rng.shuffle(idx_all)
    P = points[idx_all]
    
    #grid cell size = radius
    cell = radius
    inv = 1.0 / cell
    
    #grid: key=(ix,iy,iz) -> list of selected point indices
    grid = {}
    selected_local = []
    
    #3x3x3 neighborhood offsets
    neighbors = [
        (dx, dy, dz) 
        for dx in (-1, 0, 1) 
        for dy in (-1, 0, 1) 
        for dz in (-1, 0, 1)
    ]
    
    #process points in shuffled order
    for i, p in enumerate(P):
        #determine grid cell
        key = tuple(np.floor(p * inv).astype(np.int64))
        
        #check if too close to any point in neighboring cells
        ok = True
        for dx, dy, dz in neighbors:
            neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
            if neighbor_key in grid:
                for j in grid[neighbor_key]:
                    if np.linalg.norm(P[i] - P[j]) < radius:
                        ok = False
                        break
            if not ok:
                break
        
        #accept point if sufficiently far from others
        if ok:
            grid.setdefault(key, []).append(i)
            selected_local.append(i)
            if len(selected_local) >= target_n:
                break
    
    #map back to original indices
    sel_idx_global = idx_all[np.array(selected_local, dtype=np.int64)]
    return sel_idx_global


def sample_surface_points(mesh: trimesh.Trimesh, count: int,
                          method: str = "poisson",
                          relax: float = POISSON_RELAX,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Sample exactly 'count' points on mesh surface.
    
    Args:
        mesh: Input mesh
        count: Target number of points
        method: Sampling method ("poisson" | "area" | "even")
        relax: Poisson radius relaxation factor
        rng: Random number generator
        
    Returns:
        Point cloud (count x 3)
    """
    assert method in {"poisson", "area", "even"}
    
    if mesh is None or mesh.area <= 0 or len(mesh.faces) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    #area-weighted random sampling
    if method == "area":
        pts, _ = trimesh.sample.sample_surface(mesh, count)
        return pts.astype(np.float32)
    
    #poisson-disk sampling (grid-based)
    if method in {"poisson", "even"}:
        #compute target radius
        radius = _poisson_radius_for(mesh.area, count, relax=relax)
        
        #oversample for filtering (6x margin)
        M = int(count * 6)
        pts, _ = trimesh.sample.sample_surface(mesh, M)
        
        #filter to Poisson distribution
        sel_idx = _poisson_grid_select(pts, radius, count, rng)
        sel_pts = pts[sel_idx]
        
        #top-up if insufficient points
        if sel_pts.shape[0] < count:
            extra_need = count - sel_pts.shape[0]
            extra, _ = trimesh.sample.sample_surface(mesh, extra_need)
            sel_pts = np.vstack([sel_pts, extra])
        
        #truncate if too many (rare)
        if sel_pts.shape[0] > count:
            pick = rng.choice(sel_pts.shape[0], size=count, replace=False)
            sel_pts = sel_pts[pick]
        
        return sel_pts.astype(np.float32)
    
    #fallback
    pts, _ = trimesh.sample.sample_surface(mesh, count)
    return pts.astype(np.float32)


def surface_distances(VA: np.ndarray, FA: np.ndarray,
                      VB: np.ndarray, FB: np.ndarray,
                      n_samples: int = N_SAMPLES,
                      method: str = SAMPLE_METHOD,
                      relax: float = POISSON_RELAX) -> dict[str, float]:
    """
    Compute symmetric Hausdorff and Mean Surface Distance between two meshes.
    
    Args:
        VA, FA: Mesh A vertices and faces
        VB, FB: Mesh B vertices and faces
        n_samples: Number of points to sample per mesh
        method: Sampling method
        relax: Poisson relaxation factor
        
    Returns:
        Dictionary with 'hd' (Hausdorff) and 'msd' (Mean Surface Distance) in mm
    """
    A = build_trimesh(VA, FA)
    B = build_trimesh(VB, FB)
    
    if A is None or B is None:
        return {"hd": float("nan"), "msd": float("nan")}
    
    #sample point clouds
    rng = np.random.default_rng(RANDOM_SEED)
    PA = sample_surface_points(A, n_samples, method=method, relax=relax, rng=rng)
    PB = sample_surface_points(B, n_samples, method=method, relax=relax, rng=rng)
    
    #compute nearest-neighbor distances (A→B and B→A)
    kB = cKDTree(PB)
    dAB, _ = kB.query(PA, k=1, workers=-1)
    
    kA = cKDTree(PA)
    dBA, _ = kA.query(PB, k=1, workers=-1)
    
    #hausdorff: max of all distances
    hd = float(max(dAB.max(initial=0.0), dBA.max(initial=0.0)))
    
    #mean Surface Distance: average of all distances
    msd = float(np.concatenate([dAB, dBA]).mean())
    
    return {"hd": hd, "msd": msd}



#visualization and Statistics


def poly_from_numpy(V: np.ndarray, F: np.ndarray) -> pv.PolyData:
    """
    Convert numpy mesh to PyVista PolyData.
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        
    Returns:
        PyVista PolyData object
    """
    if len(V) == 0 or len(F) == 0:
        return pv.PolyData()
    
    #pyVista face format: [3, v0, v1, v2, 3, v3, v4, v5, ...]
    faces = np.hstack([np.full((F.shape[0], 1), 3), F]).astype(np.int64)
    return pv.PolyData(V, faces)


def mesh_stats(V: np.ndarray, F: np.ndarray, name: str = "mesh") -> dict:
    """
    Compute mesh statistics.
    
    Args:
        V: Vertices (Nx3)
        F: Faces (Mx3)
        name: Mesh identifier
        
    Returns:
        Dictionary with stats: verts, faces, area, volume, watertight, components
    """
    if len(V) == 0 or len(F) == 0:
        return {
            "name": name,
            "verts": 0,
            "faces": 0,
            "area": 0.0,
            "volume": 0.0,
            "watertight": False,
            "components": 0
        }
    
    tm = build_trimesh(V, F)
    components = tm.split(only_watertight=False)
    
    return {
        "name": name,
        "verts": len(V),
        "faces": len(F),
        "area": float(tm.area),
        "volume": float(tm.volume) if tm.is_watertight else 0.0,
        "watertight": bool(tm.is_watertight),
        "components": len(components)
    }


def pretty(n: int) -> str:
    """Format integer with spaces as thousand separators."""
    return f"{n:,}".replace(",", " ")


def dump_block(title: str, ref: tuple, imc0: tuple, imc_def: tuple, imc_ext: tuple):
    """
    Print comparison statistics block for all algorithm variants.
    
    Args:
        title: Block title (e.g., "Brain" or "Tumor")
        ref: ((V_mc, F_mc), time_mc)
        imc0: ((V_imc0, F_imc0), time_imc0)
        imc_def: ((V_def, F_def), None)
        imc_ext: ((V_ext, F_ext), None)
    """
    print(f"\n## {title}")
    
    #print mesh statistics
    for name, (V, F), t in [
        ("MC", ref[0], ref[1]),
        ("IMC0", imc0[0], imc0[1]),
        ("IMCdef", imc_def[0], None),
        ("IMCext", imc_ext[0], None),
    ]:
        st = mesh_stats(V, F, name)
        print(f"- {name:6s}: verts={pretty(st['verts'])}, faces={pretty(st['faces'])}, "
              f"area={st['area']:.3f} mm², volume={st['volume']:.3f} mm³, "
              f"watertight={st['watertight']}, comps={st['components']}")
    
    #compute distances relative to MC reference
    V_mc, F_mc = ref[0]
    V0, F0 = imc0[0]
    Vd, Fd = imc_def[0]
    Vx, Fx = imc_ext[0]
    
    d0 = surface_distances(V_mc, F_mc, V0, F0)
    d1 = surface_distances(V_mc, F_mc, Vd, Fd)
    d2 = surface_distances(V_mc, F_mc, Vx, Fx)
    
    print(f"  Dist(MC→IMC0)  : HD={d0['hd']:.3f} mm, MSD={d0['msd']:.3f} mm")
    print(f"  Dist(MC→IMCdef): HD={d1['hd']:.3f} mm, MSD={d1['msd']:.3f} mm")
    print(f"  Dist(MC→IMCext): HD={d2['hd']:.3f} mm, MSD={d2['msd']:.3f} mm")



#main Execution


if __name__ == "__main__":
    
    #USER CONFIGURATION : Modify paths here
  
    ROOT = Path("data/3d-brain-mri_repo/extracted")  #dataset root directory
    
    #specify your file paths here (modify these):
    brain_nii_path = ROOT / "UCSF-PDGM-0009_nifti" / "UCSF-PDGM-0009_brain_parenchyma_segmentation.nii.gz"
    tumor_nii_path = ROOT / "UCSF-PDGM-0009_nifti" / "UCSF-PDGM-0009_tumor_segmentation.nii.gz"
    
  
    #automatic fallback: if files don't exist, pick first available subject
    
    if not brain_nii_path.exists() or not tumor_nii_path.exists():
        print("Specified files not found. Searching for any valid subject...")
        try:
            sd = pick_subject(ROOT)
            print(f"Using subject: {sd.name}")
            brain_nii_path = find_brain_mask(sd)
            tumor_nii_paths = find_tumor_masks(sd)
            tumor_nii_path = tumor_nii_paths[0] if tumor_nii_paths else None
            
            if brain_nii_path is None or tumor_nii_path is None:
                raise FileNotFoundError("No valid brain/tumor masks found")
        except Exception as e:
            print(f"Error: {e}")
            print("\nPlease check:")
            print(f"  1. ROOT directory exists: {ROOT}")
            print(f"  2. Directory contains *_nifti folders")
            print(f"  3. Folders contain brain_parenchyma and tumor segmentation files")
            exit(1)
    
 
    #load data

    print(f"\nBrain mask: {brain_nii_path.name}")
    print(f"Tumor mask: {tumor_nii_path.name}")
    
    #load binary volumes and spacing
    brain_bin, brain_sp = load_binary_zooms(brain_nii_path)
    tumor_bin, tumor_sp = load_binary_zooms(tumor_nii_path)
    
    print(f"Brain shape: {brain_bin.shape}, spacing: {brain_sp} mm")
    print(f"Tumor shape: {tumor_bin.shape}, spacing: {tumor_sp} mm")

    #generate surfaces with all algorithms
    
    #Marching Cubes (reference) 
    print("\n[1/5] Running Marching Cubes (reference)...")
    t0 = time.perf_counter()
    bV_mc, bF_mc = mc_surface_skimage(brain_bin, brain_sp)
    t1 = time.perf_counter()
    tV_mc, tF_mc = mc_surface_skimage(tumor_bin, tumor_sp)
    t2 = time.perf_counter()
    
    #IMC (pure, no tightening)
    print("[2/5] Running IMC (pure)...")
    t3 = time.perf_counter()
    bV_imc0, bF_imc0 = imc_surface(brain_bin, brain_sp)
    t4 = time.perf_counter()
    tV_imc0, tF_imc0 = imc_surface(tumor_bin, tumor_sp)
    t5 = time.perf_counter()
    
    #IMC with default tightening
    print("[3/5] Running IMC with default tightening...")
    MAC_def = 20.0  #maximum normal angle (degrees)
    MDC_def_b = 1.2 * min(brain_sp)  #~1.2 voxel diagonal for brain
    MDC_def_t = 1.2 * min(tumor_sp)  #~1.2 voxel diagonal for tumor
    
    bV_imc, bF_imc = tighten_mesh_IMC(bV_imc0, bF_imc0, MAC_def, MDC_def_b, max_passes=2)
    tV_imc, tF_imc = tighten_mesh_IMC(tV_imc0, tF_imc0, MAC_def, MDC_def_t, max_passes=2)
    
    #IMC with extreme tightening
    print("[4/5] Running IMC with extreme tightening...")
    MAC_ext = 45.0  #more permissive angle
    MDC_ext_b = 2.0 * min(brain_sp)  #2x voxel diagonal for brain
    MDC_ext_t = 2.0 * min(tumor_sp)  #2x voxel diagonal for tumor
    
    bV_imcX, bF_imcX = tighten_mesh_IMC(bV_imc0, bF_imc0, MAC_ext, MDC_ext_b, max_passes=6)
    tV_imcX, tF_imcX = tighten_mesh_IMC(tV_imc0, tF_imc0, MAC_ext, MDC_ext_t, max_passes=6)

    #print results
    
    print("TIMING RESULTS")
    print(f"MC brain  : {t1 - t0:.2f}s | IMC0 brain : {t4 - t3:.2f}s")
    print(f"MC tumor  : {t2 - t1:.2f}s | IMC0 tumor : {t5 - t4:.2f}s")
    
    print("STATISTICS & DISTANCES")
    
    dump_block(
        "Brain",
        ((bV_mc, bF_mc), t1 - t0),
        ((bV_imc0, bF_imc0), t4 - t3),
        ((bV_imc, bF_imc), None),
        ((bV_imcX, bF_imcX), None)
    )
    
    dump_block(
        "Tumor",
        ((tV_mc, tF_mc), t2 - t1),
        ((tV_imc0, tF_imc0), t5 - t4),
        ((tV_imc, tF_imc), None),
        ((tV_imcX, tF_imcX), None)
    )
    
    #visualization (2x2 grid)
    
    print("\n[5/5] Rendering visualization...")
    
    try:
        pv.set_jupyter_backend("trame")
    except Exception:
        pass
    
    def show_panel(title: str, brain: tuple, tumor: tuple, 
                   pl: pv.Plotter, r: int, c: int):
        """Add mesh panel to plotter grid."""
        pl.subplot(r, c)
        pl.add_text(title, font_size=12)
        pl.add_mesh(poly_from_numpy(*brain), color="lightgray", opacity=0.3) #keep opacity to 0.3 for better visibility of tumor
        pl.add_mesh(poly_from_numpy(*tumor), color="red", opacity=1.0)
        pl.add_axes()
        pl.show_grid()
    
    #create 2x2 plotter
    pl = pv.Plotter(shape=(2, 2), window_size=(1400, 1000))
    
    #top row: MC vs IMC0
    show_panel("MC", (bV_mc, bF_mc), (tV_mc, tF_mc), pl, 0, 0)
    show_panel("IMC0", (bV_imc0, bF_imc0), (tV_imc0, tF_imc0), pl, 0, 1)
    
    #bottom row: IMC default vs extreme
    show_panel("IMCdef", (bV_imc, bF_imc), (tV_imc, tF_imc), pl, 1, 0)
    show_panel("IMCext", (bV_imcX, bF_imcX), (tV_imcX, tF_imcX), pl, 1, 1)
    
    #link camera views and display
    pl.link_views()
    pl.show()
    
    print("Finished :) ")