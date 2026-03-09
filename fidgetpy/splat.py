"""
fidgetpy.splat — SDF → Gaussian Splatting converter.

Converts any fidgetpy SDF expression into a Gaussians object or a Gaussian
Splatting .ply file suitable for viewing in SuperSplat, SIBR, or Gaussian
Opacity Fields viewers.

Scale estimation uses the SDF Hessian to compute principal curvatures
analytically, giving each Gaussian a correctly-shaped covariance ellipsoid:
thin along the surface normal, wide along tangent directions with widths
inversely proportional to principal curvatures.
"""

import numpy as np
import time


# ──────────────────────────────────────────────────────────────────────────────
#  SDF evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _eval_sdf_at(expr, pts):
    """
    Evaluate a fidgetpy SDF expression at an (N, 3) array of xyz points.
    Returns a float64 array of shape (N,).

    Automatically handles expressions that reference fewer than three
    coordinates by retrying with a reduced variable mapping.
    """
    import fidgetpy as fp
    pts32 = pts.astype(np.float32)
    vx = fp.var('x')
    vy = fp.var('y')
    vz = fp.var('z')
    candidates = [
        ([vx, vy, vz], pts32),
        ([vx, vy],     pts32[:, :2]),
        ([vx],         pts32[:, :1]),
    ]
    for vars_, data in candidates:
        try:
            result = fp.eval(expr, data, variables=vars_)
            return np.asarray(result, dtype=np.float64)
        except (ValueError, TypeError):
            continue
    raise RuntimeError(
        "Could not evaluate SDF expression: check that your expression uses "
        "fp.x(), fp.y(), and/or fp.z() (or fp.var('x') etc.)."
    )


def _auto_domain(expr, padding=0.15, verbose=False):
    """
    Estimate the sampling domain by meshing at low resolution and reading
    the vertex bounding box.  Falls back to (-2, 2) if meshing produces nothing.

    Tries scale=1 first, then scale=2, then scale=4 to catch larger shapes.
    """
    import fidgetpy as fp
    for scale in (1.0, 2.0, 4.0):
        try:
            m = fp.mesh(expr, depth=3, scale=scale, numpy=True)
            verts = m.vertices
            if len(verts) == 0:
                continue
            lo = float(verts.min()) - padding
            hi = float(verts.max()) + padding
            return (lo, hi)
        except Exception as e:
            if verbose:
                print(f"  _auto_domain: mesh at scale={scale} failed: {e}")
    if verbose:
        print("  _auto_domain: falling back to (-2, 2)")
    return (-2.0, 2.0)


def _compute_gradient(expr, pts):
    """
    ∇SDF at each point via GradSliceEval (exact forward-mode AD, single pass).
    Returns shape (N, 3). Also returns the SDF value as the second element if
    called as (grad, val) = _compute_gradient_with_val(expr, pts).
    """
    import fidgetpy as fp
    result = fp.eval_grad(expr, np.asarray(pts, dtype=np.float32))
    return np.asarray(result[:, 1:], dtype=np.float64)


def _compute_gradient_and_val(expr, pts):
    """
    Returns (sdf_values (N,), gradient (N, 3)) in one GradSliceEval pass.
    Replaces separate _eval_sdf_at + _compute_gradient calls.
    """
    import fidgetpy as fp
    result = fp.eval_grad(expr, np.asarray(pts, dtype=np.float32))
    return (np.asarray(result[:, 0], dtype=np.float64),
            np.asarray(result[:, 1:], dtype=np.float64))


def _compute_hessian(expr, pts, h=1.5e-3):
    """
    Full 3×3 Hessian of SDF at each point via finite differences.
    Uses a 13-tap stencil. Returns shape (N, 3, 3).
    """
    N = len(pts)
    H = np.zeros((N, 3, 3))
    d0 = _eval_sdf_at(expr, pts)

    for i in range(3):
        ei = np.zeros(3); ei[i] = h
        H[:, i, i] = (_eval_sdf_at(expr, pts + ei) - 2 * d0 + _eval_sdf_at(expr, pts - ei)) / (h * h)

    for i in range(3):
        for j in range(i + 1, 3):
            ei = np.zeros(3); ei[i] = h
            ej = np.zeros(3); ej[j] = h
            val = (
                _eval_sdf_at(expr, pts + ei + ej)
                - _eval_sdf_at(expr, pts + ei - ej)
                - _eval_sdf_at(expr, pts - ei + ej)
                + _eval_sdf_at(expr, pts - ei - ej)
            ) / (4 * h * h)
            H[:, i, j] = val
            H[:, j, i] = val

    return H


# ──────────────────────────────────────────────────────────────────────────────
#  Surface geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _principal_curvatures(normals, hessians):
    """
    Extract principal curvatures κ₁, κ₂ and principal directions t₁, t₂
    from the surface-projected Hessian (shape operator).

    Returns:
        kappa1, kappa2 : (N,)   principal curvatures (|κ₁| ≤ |κ₂|)
        t1, t2         : (N,3)  principal tangent directions (unit vectors)
    """
    N = len(normals)
    kappa1 = np.zeros(N)
    kappa2 = np.zeros(N)
    t1 = np.zeros((N, 3))
    t2 = np.zeros((N, 3))

    for i in range(N):
        n = normals[i]
        H = hessians[i]

        ref = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
        u = np.cross(n, ref); u /= np.linalg.norm(u) + 1e-12
        v = np.cross(n, u)

        Hu = H @ u; Hv = H @ v
        M = np.array([[u @ Hu, u @ Hv], [v @ Hu, v @ Hv]])

        eigvals, eigvecs = np.linalg.eigh(M)
        order = np.argsort(np.abs(eigvals))
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        kappa1[i] = eigvals[0]
        kappa2[i] = eigvals[1]
        t1[i] = eigvecs[0, 0] * u + eigvecs[1, 0] * v
        t2[i] = eigvecs[0, 1] * u + eigvecs[1, 1] * v

    return kappa1, kappa2, t1, t2


def _rotation_matrix_to_quaternion(R):
    """
    Convert (N,3,3) rotation matrices → (N,4) quaternions [x,y,z,w].
    Uses Shepperd's method for numerical stability.
    """
    N = len(R)
    q = np.zeros((N, 4))
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    s = np.sqrt(np.maximum(trace + 1.0, 1e-10)) * 2
    mask = trace > 0
    q[mask, 3] = 0.25 * s[mask]
    q[mask, 0] = (R[mask, 2, 1] - R[mask, 1, 2]) / s[mask]
    q[mask, 1] = (R[mask, 0, 2] - R[mask, 2, 0]) / s[mask]
    q[mask, 2] = (R[mask, 1, 0] - R[mask, 0, 1]) / s[mask]

    m0 = (~mask) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = np.sqrt(np.maximum(1.0 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2], 1e-10)) * 2
    q[m0, 3] = (R[m0, 2, 1] - R[m0, 1, 2]) / s2[m0]
    q[m0, 0] = 0.25 * s2[m0]
    q[m0, 1] = (R[m0, 0, 1] + R[m0, 1, 0]) / s2[m0]
    q[m0, 2] = (R[m0, 0, 2] + R[m0, 2, 0]) / s2[m0]

    m1 = (~mask) & (~m0) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = np.sqrt(np.maximum(1.0 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2], 1e-10)) * 2
    q[m1, 3] = (R[m1, 0, 2] - R[m1, 2, 0]) / s3[m1]
    q[m1, 0] = (R[m1, 0, 1] + R[m1, 1, 0]) / s3[m1]
    q[m1, 1] = 0.25 * s3[m1]
    q[m1, 2] = (R[m1, 1, 2] + R[m1, 2, 1]) / s3[m1]

    m2 = (~mask) & (~m0) & (~m1)
    s4 = np.sqrt(np.maximum(1.0 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2], 1e-10)) * 2
    q[m2, 3] = (R[m2, 1, 0] - R[m2, 0, 1]) / s4[m2]
    q[m2, 0] = (R[m2, 0, 2] + R[m2, 2, 0]) / s4[m2]
    q[m2, 1] = (R[m2, 1, 2] + R[m2, 2, 1]) / s4[m2]
    q[m2, 2] = 0.25 * s4[m2]

    norms = np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12
    q /= norms
    q[q[:, 3] < 0] *= -1
    return q


# ──────────────────────────────────────────────────────────────────────────────
#  Spherical Harmonics helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sh_coeffs_count(degree):
    return (degree + 1) ** 2


def _color_to_sh_dc(color_rgb):
    """Convert linear RGB → degree-0 SH coefficient (DC term)."""
    C0 = 0.28209479177387814
    color_clamped = np.clip(color_rgb, 0.001, 0.999)
    return np.log(color_clamped / (1 - color_clamped)) / C0


def _build_sh_features(color_rgb, normals, degree):
    """Build full SH feature array of shape (N, (deg+1)²×3)."""
    N = len(color_rgb)
    n_coeffs = _sh_coeffs_count(degree)
    sh = np.zeros((N, n_coeffs, 3))

    sh[:, 0, :] = _color_to_sh_dc(color_rgb)

    if degree >= 1:
        scale = 0.1
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        sh[:, 1, :] = ny[:, None] * scale   # Y_{1,-1}
        sh[:, 2, :] = nz[:, None] * scale   # Y_{1, 0}
        sh[:, 3, :] = nx[:, None] * scale   # Y_{1, 1}

    if degree >= 2:
        scale2 = 0.03
        nx, ny, nz = normals[:, 0], normals[:, 1], normals[:, 2]
        sh[:, 4, :] = (nx * ny)[:, None] * scale2
        sh[:, 5, :] = (ny * nz)[:, None] * scale2
        sh[:, 6, :] = (3 * nz**2 - 1)[:, None] * scale2 / 2
        sh[:, 7, :] = (nx * nz)[:, None] * scale2
        sh[:, 8, :] = (nx**2 - ny**2)[:, None] * scale2 / 2

    # degree >= 3: band-3 stays zero (no photometric data from SDF alone)

    return sh.reshape(N, n_coeffs * 3)


# ──────────────────────────────────────────────────────────────────────────────
#  PLY writer
# ──────────────────────────────────────────────────────────────────────────────

def _write_ply(path, gaussians, verbose=True):
    """
    Write Gaussian Splatting .ply in the standard 3DGS binary format.
    Fields: x y z  nx ny nz  f_dc_0..2  f_rest_0..M  opacity
            scale_0..2  rot_0..3 (wxyz convention)
    """
    N = gaussians['positions'].shape[0]
    n_sh_rest = gaussians['sh_rest'].shape[1] if gaussians['sh_rest'].size else 0

    props = []
    props += [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    props += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    props += [('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')]
    props += [(f'f_rest_{i}', 'f4') for i in range(n_sh_rest)]
    props += [('opacity', 'f4')]
    props += [('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4')]
    props += [('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4')]

    dtype = np.dtype([(name, fmt) for name, fmt in props])
    arr = np.zeros(N, dtype=dtype)

    pos = gaussians['positions']
    arr['x'] = pos[:, 0]; arr['y'] = pos[:, 1]; arr['z'] = pos[:, 2]

    nrm = gaussians['normals']
    arr['nx'] = nrm[:, 0]; arr['ny'] = nrm[:, 1]; arr['nz'] = nrm[:, 2]

    sh_dc = gaussians['sh_dc']
    arr['f_dc_0'] = sh_dc[:, 0]; arr['f_dc_1'] = sh_dc[:, 1]; arr['f_dc_2'] = sh_dc[:, 2]

    for i in range(n_sh_rest):
        arr[f'f_rest_{i}'] = gaussians['sh_rest'][:, i]

    arr['opacity'] = gaussians['opacity']

    sc = gaussians['scales']
    arr['scale_0'] = sc[:, 0]; arr['scale_1'] = sc[:, 1]; arr['scale_2'] = sc[:, 2]

    # 3DGS quaternion convention: rot_0=w, rot_1=x, rot_2=y, rot_3=z
    q = gaussians['quaternions']
    arr['rot_0'] = q[:, 3]; arr['rot_1'] = q[:, 0]
    arr['rot_2'] = q[:, 1]; arr['rot_3'] = q[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
    )
    for name, _ in props:
        header += f"property float {name}\n"
    header += "end_header\n"

    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(arr.tobytes())

    if verbose:
        print(f"  Wrote {N:,} Gaussians → {path}  ({arr.nbytes / 1e6:.2f} MB)")


# ──────────────────────────────────────────────────────────────────────────────
#  Gaussians — inspectable result object
# ──────────────────────────────────────────────────────────────────────────────

class Gaussians:
    """
    A collection of Gaussian splats computed from an SDF.

    Attributes:
        positions   (N, 3) float64 — surface point positions.
        normals     (N, 3) float64 — surface normals (unit vectors).
        colors      (N, 3) float64 — linear RGB color in [0, 1].
        scales      (N, 3) float64 — Gaussian radii (tangent1, tangent2, normal).
        quaternions (N, 4) float64 — rotation quaternions [x, y, z, w].
        count       int             — number of Gaussians.

    Methods:
        save(path, verbose=True) — write a 3DGS-compatible .ply file.
    """

    _C0 = 0.28209479177387814  # degree-0 SH normalisation constant

    def __init__(self, gaussians_dict):
        self.positions   = gaussians_dict['positions']
        self.normals     = gaussians_dict['normals']
        self.quaternions = gaussians_dict['quaternions']

        # Expose linear-RGB colors (inverse of the log-odds SH encoding)
        sh_dc = gaussians_dict['sh_dc']
        self.colors = 1.0 / (1.0 + np.exp(-sh_dc * self._C0))

        # Expose actual scale values (exp of the stored log-scale)
        self.scales = np.exp(gaussians_dict['scales'])

        # Keep raw data for saving
        self._raw = gaussians_dict

    @property
    def count(self):
        return len(self.positions)

    def save(self, path, colors=None, verbose=True):
        """
        Write a 3DGS-compatible binary PLY file.

        Args:
            path:    Output file path.
            colors:  (N, 3) float array in [0, 1] to override the stored RGB
                     colors, or None to use the colors set at splat time.
            verbose: Print a summary line. Default True.

        Returns:
            str: The path that was written.

        Example::

            import fidgetpy as fp, numpy as np
            g = fp.splat(shape)
            grad = fp.eval_grad(shape, g.positions.astype(np.float32))
            brightness = np.clip((grad[:, 1:] * [0.6, 0.8, 0.0]).sum(axis=1), 0, 1)
            g.save("out.ply", colors=np.stack([brightness]*3, axis=1))
        """
        import math as _math
        raw = self._raw
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)
            if colors.shape != (self.count, 3):
                raise ValueError(
                    f"colors must have shape ({self.count}, 3), got {colors.shape}"
                )
            # Recover SH degree from stored sh_rest shape
            n_rest = raw['sh_rest'].shape[1] if raw['sh_rest'].size else 0
            sh_degree = round(_math.sqrt(n_rest // 3 + 1)) - 1
            sh_all = _build_sh_features(colors, self.normals, sh_degree)
            raw = dict(raw)
            raw['sh_dc'] = sh_all[:, :3]
            raw['sh_rest'] = sh_all[:, 3:] if sh_degree > 0 else np.zeros((self.count, 0))
        _write_ply(path, raw, verbose=verbose)
        return path

    def __repr__(self):
        return f"Gaussians(count={self.count:,})"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_color(color, pts, N):
    """
    Resolve the ``color`` argument to an (N, 3) float64 array.

    Accepted forms:
        None                     → white
        callable                 → called as color(pts) → ndarray(N, 3)
        (r, g, b) tuple/list     → each channel is a float or a fidgetpy expression
    """
    if color is None:
        return np.ones((N, 3), dtype=np.float64)

    if callable(color):
        result = np.clip(color(pts), 0.0, 1.0).astype(np.float64)
        if result.shape != (N, 3):
            raise ValueError(
                f"color callable must return shape (N, 3), got {result.shape}"
            )
        return result

    # (r, g, b) tuple — each element is a float or a fidgetpy expression
    if len(color) != 3:
        raise ValueError(
            f"color tuple must have 3 elements (r, g, b), got {len(color)}"
        )
    out = np.zeros((N, 3), dtype=np.float64)
    for i, c in enumerate(color):
        if isinstance(c, (int, float)):
            out[:, i] = np.clip(float(c), 0.0, 1.0)
        elif callable(c):
            # _CallableExpr or any callable returning (N,) array
            out[:, i] = np.clip(c(pts), 0.0, 1.0)
        else:
            # Assume fidgetpy expression — evaluate at surface points
            out[:, i] = np.clip(_eval_sdf_at(c, pts), 0.0, 1.0)
    return out


def splat(
    expr,
    output_file=None,
    color=None,
    size=96,
    domain=None,
    surface_thresh=0.02,
    max_gaussians=None,
    sh_degree=0,
    h_hess=1.5e-3,
    scale_min=5e-5,
    scale_max=0.02,
    opacity_logit=4.0,
    verbose=True,
):
    """
    Convert a fidgetpy SDF expression to a Gaussian Splatting representation.

    The SDF is sampled on a regular grid, surface candidates are extracted and
    refined via Newton projection, then each surface point becomes a Gaussian
    whose shape is derived from the local principal curvatures.

    When output_file is None, returns a ``Gaussians`` object you can inspect
    (positions, colors, normals, scales, quaternions) and save later.
    When output_file is provided, writes the .ply immediately and returns the path.

    Args:
        expr:           A fidgetpy SDF expression using fp.x(), fp.y(), fp.z().
        output_file:    Path to the output .ply file, or None to return a
                        Gaussians object. Default: None.
        color:          Surface color.  Accepted forms:

                        ``None`` (default)
                            All Gaussians are white.

                        ``(r, g, b)`` tuple
                            Each channel is either a ``float`` in [0, 1] or a
                            fidgetpy expression that is evaluated at every
                            surface point::

                                # Solid red
                                color=(1.0, 0.0, 0.0)

                                # X-axis gradient from black to green
                                color=(0.0, fp.x() * 0.5 + 0.5, 0.0)

                        callable ``(pts: ndarray(N,3)) -> ndarray(N,3)``
                            Arbitrary per-point color function returning linear
                            RGB in [0, 1].

        size:           Sampling grid resolution (size³ points). Default: 96.
        domain:         (min, max) range for all three axes. If None (default),
                        the bounds are estimated automatically from a low-res mesh.
        surface_thresh: SDF value threshold for surface detection. Default: 0.02.
        max_gaussians:  Cap on number of output Gaussians (random subsample).
                        None means no cap. Default: None.
        sh_degree:      Spherical harmonics degree 0–3. Default: 0 (color only).
        h_hess:         Finite-difference step for Hessian computation.
        scale_min:      Minimum Gaussian scale (log-space clamping). Default: 5e-5.
        scale_max:      Maximum Gaussian scale. Default: 0.02.
        opacity_logit:  Opacity stored as logit (inverse sigmoid). Default: 4.0.
        verbose:        Print progress information. Default: True.

    Returns:
        Gaussians: when output_file is None — inspectable object with .save().
        str:       when output_file is set — path to the written .ply file.

    Examples:
        import fidgetpy as fp
        import fidgetpy.shape as fps

        # Return Gaussians object for inspection
        g = fp.splat(fps.sphere(1.0))
        print(g.count, g.positions.shape)  # inspect
        g.save("sphere.ply")               # write later

        # Write directly
        fp.splat(fps.sphere(1.0), output_file="sphere.ply")

        # Solid color
        fp.splat(fps.sphere(1.0), color=(0.2, 0.6, 1.0), output_file="blue.ply")
    """
    t_total = time.time()

    # Resolve domain (auto-detect from low-res mesh if not given)
    if domain is None:
        if verbose:
            print("  Auto-detecting domain from low-res mesh...")
        domain = _auto_domain(expr, verbose=verbose)
        if verbose:
            print(f"  Domain: {domain[0]:.3f} → {domain[1]:.3f}")

    if verbose:
        print("=" * 60)
        print("  SDF → Gaussian Splatting Converter (fidgetpy)")
        print(f"  Grid: {size}³  |  SH degree: {sh_degree}")
        print(f"  Domain: {domain}  |  Surface thresh: {surface_thresh}")
        print("=" * 60)

    # ── Step 1: Sample SDF on a regular grid ──────────────────────────────────
    if verbose:
        print("\n[1/5] Sampling SDF grid...")
    lo, hi = domain
    voxel_size = (hi - lo) / size
    coords = np.linspace(lo, hi, size)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing='ij')
    pts_all = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=-1).astype(np.float64)

    t0 = time.time()
    d_all = _eval_sdf_at(expr, pts_all)
    if verbose:
        print(f"  Evaluated {len(pts_all):,} voxels in {time.time() - t0:.2f}s")

    # ── Step 2: Extract and refine surface candidates ─────────────────────────
    if verbose:
        print("\n[2/5] Extracting zero-level set...")
    mask = (d_all < voxel_size * 0.5) & (d_all > -surface_thresh)
    pts_surf = pts_all[mask]
    if verbose:
        print(f"  Found {len(pts_surf):,} surface voxels")

    if len(pts_surf) == 0:
        raise RuntimeError(
            "No surface found — try increasing surface_thresh or size, "
            "or check that your SDF fits within the domain."
        )

    # Newton refinement: project candidates onto the true zero-level set
    if verbose:
        print("  Projecting candidates onto zero-level set (Newton refinement)...")
    pts_start = pts_surf.copy()
    for _ in range(5):
        d_cur, grad_cur = _compute_gradient_and_val(expr, pts_surf)
        grad_mag = np.linalg.norm(grad_cur, axis=-1, keepdims=True)
        n_cur = grad_cur / (grad_mag + 1e-12)
        pts_surf = pts_surf - d_cur[:, np.newaxis] * n_cur

    d_final = _eval_sdf_at(expr, pts_surf)

    # Discard floaters: candidates that drifted far from their grid origin during
    # projection landed on a different surface (common with smooth-blend SDFs where
    # |∇SDF| ≠ 1 in blend zones).  Keep only those within 3 voxels of their start.
    displacement = np.linalg.norm(pts_surf - pts_start, axis=-1)
    keep = displacement < voxel_size * 3.0
    pts_surf = pts_surf[keep]
    d_final = d_final[keep]

    if verbose:
        n_floaters = (~keep).sum()
        print(f"  After projection: max |SDF| = {np.abs(d_final).max():.2e}"
              + (f"  |  removed {n_floaters:,} floaters" if n_floaters else ""))

    # Optional subsample
    if max_gaussians is not None and len(pts_surf) > max_gaussians:
        idx = np.random.choice(len(pts_surf), max_gaussians, replace=False)
        pts_surf = pts_surf[idx]
        if verbose:
            print(f"  Subsampled to {len(pts_surf):,} Gaussians")

    N = len(pts_surf)

    # ── Step 3: Normals and colors ────────────────────────────────────────────
    if verbose:
        print("\n[3/5] Computing normals and surface colors...")
    t0 = time.time()
    grad = _compute_gradient(expr, pts_surf)
    grad_mag = np.linalg.norm(grad, axis=-1, keepdims=True)
    normals = grad / (grad_mag + 1e-12)

    colors = _resolve_color(color, pts_surf, N)

    if verbose:
        print(f"  Done in {time.time() - t0:.2f}s  |  avg |∇SDF| = {grad_mag.mean():.4f}")

    # ── Step 4: Hessian → principal curvatures → scales ──────────────────────
    if verbose:
        print("\n[4/5] Computing Hessians and principal curvatures...")
        print("  (Expensive step — uses Hessian shape operator)")
    t0 = time.time()
    hessians = _compute_hessian(expr, pts_surf, h=h_hess)
    kappa1, kappa2, t1, t2 = _principal_curvatures(normals, hessians)

    if verbose:
        print(f"  Hessians computed in {time.time() - t0:.2f}s")
        print(f"  κ₁ range: [{kappa1.min():.3f}, {kappa1.max():.3f}]")
        print(f"  κ₂ range: [{kappa2.min():.3f}, {kappa2.max():.3f}]")

    # Scales from curvature: flatter regions get wider Gaussians
    laplacian = hessians[:, 0, 0] + hessians[:, 1, 1] + hessians[:, 2, 2]
    feature_scale = np.clip(1.0 / (np.abs(laplacian) + 0.5), scale_min, scale_max)

    s_t1 = np.minimum(np.clip(1.0 / (np.abs(kappa1) + 0.3), scale_min, scale_max), feature_scale)
    s_t2 = np.minimum(np.clip(1.0 / (np.abs(kappa2) + 0.3), scale_min, scale_max), feature_scale)
    s_n = np.full(N, voxel_size * 0.5)  # thin along normal; must stay non-zero for silhouette coverage

    # 3DGS stores log(scale)
    log_scales = np.stack([
        np.log(s_t1 + 1e-10),
        np.log(s_t2 + 1e-10),
        np.log(s_n + 1e-10),
    ], axis=-1)

    # Rotation: columns are [t1, t2, normal]
    R_mats = np.stack([t1, t2, normals], axis=-1)
    quats = _rotation_matrix_to_quaternion(R_mats)

    # ── Step 5: SH features, opacity, and write ───────────────────────────────
    if verbose:
        print("\n[5/5] Building SH features and writing PLY...")

    n_coeffs = _sh_coeffs_count(sh_degree)
    sh_all = _build_sh_features(colors, normals, sh_degree)
    sh_dc = sh_all[:, :3]
    sh_rest = sh_all[:, 3:] if sh_degree > 0 else np.zeros((N, 0))

    gaussians_dict = {
        'positions':   pts_surf,
        'normals':     normals,
        'sh_dc':       sh_dc,
        'sh_rest':     sh_rest,
        'opacity':     np.full(N, opacity_logit),
        'scales':      log_scales,
        'quaternions': quats,
    }

    result = Gaussians(gaussians_dict)

    if verbose:
        print(f"\n✓ Total time: {time.time() - t_total:.2f}s")
        print(f"  SH degree {sh_degree} → {n_coeffs} coefficients × 3 channels")
        print("=" * 60)

    if output_file is not None:
        result.save(output_file, verbose=verbose)
        return output_file

    return result
