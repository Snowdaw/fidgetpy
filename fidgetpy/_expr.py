"""
fidgetpy.expr — Container: a named collection of SDF expressions and values.

Container is the unified object for bundling a geometry SDF with any number
of named data attributes (colour r/g/b, PBR roughness/metallic, custom
simulation fields, etc.).  Every attribute is either a fidgetpy SDF expression
or a plain float.  There is no special "colour" type — colour is just three
attributes named 'r', 'g', 'b'.

Usage:
    import fidgetpy as fp
    import fidgetpy.shape as fps
    import fidgetpy.math as fpm

    # Default container already has shape, r, g, b slots
    fpc = fp.container()
    fpc.shape = fps.sphere(1.0)
    fpc.r = 0.9
    fpc.g = 0.1
    fpc.b = 0.1

    # Or declare and fill in one go with keyword args
    fpc = fp.container("shape", "r", "g", "b")
    fpc.shape = fps.rounded_box(1, 1, 1, 0.02)
    fpc.r, fpc.g, fpc.b = 0.1, 0.2, 0.9   # blue

    # Proximity paint: blend r/g/b near a dot SDF
    dot = fps.sphere(0.15).translate(0.5, 0.5, 0)
    fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.08)

    # Export
    fp.to_vm(fpc)      # dict {name: vm_string} for all set attributes
    fpc.mesh()         # mesh from 'shape'
    fpc.splat()        # Gaussian splat from 'shape' + r/g/b

    # Iterate
    for ch in fpc:
        with open(ch.name + ".vm", "w") as f:
            f.write(ch.to_vm())
"""


# ──────────────────────────────────────────────────────────────────────────────
#  ChannelEntry — yielded when iterating over a Container
# ──────────────────────────────────────────────────────────────────────────────

class ChannelEntry:
    """
    One named attribute from a Container, yielded by iteration.

    Attributes:
        name:  Attribute name (str).
        value: Current value — float, SDF expression, or None (unset).

    Methods:
        vm():  Return the VM string for this attribute's expression.
    """

    __slots__ = ('name', 'value')

    def __init__(self, name, value):
        self.name  = name
        self.value = value

    def _coerce_to_expr(self):
        """Return an SDF expression for this attribute's value (wraps floats)."""
        if self.value is None:
            raise ValueError(
                f"Attribute '{self.name}' has no value set. "
                f"Assign a value before exporting: fpc.{self.name} = ..."
            )
        if isinstance(self.value, (int, float)):
            from fidgetpy.fidgetpy import x as _x
            return _x() * 0.0 + float(self.value)
        return self.value

    def to_vm(self):
        """Return the VM string for this attribute's SDF expression."""
        from fidgetpy.fidgetpy import to_vm as _to_vm
        return _to_vm(self._coerce_to_expr())

    def to_frep(self):
        """Return the F-Rep string for this attribute's SDF expression."""
        from fidgetpy.fidgetpy import to_frep as _to_frep
        return _to_frep(self._coerce_to_expr())

    def __repr__(self):
        return f"ChannelEntry({self.name!r}, {self.value!r})"


# ──────────────────────────────────────────────────────────────────────────────
#  Container
# ──────────────────────────────────────────────────────────────────────────────

class Container:
    """
    A named collection of SDF expressions and scalar values.

    Each attribute maps a name (str) to one of:
      - a fidgetpy SDF expression  (evaluated per point at render/splat time)
      - a float                    (constant value)
      - None                       (declared but not yet assigned)

    The primary attribute is conventionally named ``'shape'`` and holds the
    geometry SDF.  Additional attributes carry per-point data:
    colour (``'r'``, ``'g'``, ``'b'``), PBR (``'roughness'``, ``'metallic'``),
    custom simulation fields, etc.

    Construction
    ------------
    Declare attribute names up front (all values start as None):

        fpc = fp.container("shape", "r", "g", "b")
        fpc.shape = fps.sphere(1.0)
        fpc.r = 0.9
        fpc.g = 0.1
        fpc.b = 0.1

    Or declare an empty container and add attributes later:

        fpc = fp.container()
        fpc.shape = fps.sphere(1.0)   # auto-creates the 'shape' attribute

    Modification
    ------------
        fpc.add("roughness")          # declare new empty attribute
        fpc.update("r", 0.5)         # update by name (raises if missing)
        fpc.remove("roughness")       # remove attribute

    Proximity paint
    ---------------
    Blend named attributes near a region SDF:

        dot = fps.sphere(0.1).translate(0.5, 0.5, 0)
        fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.05)

    Export
    ------
        fp.to_vm(fpc)    # dict {name: vm_string} for all set attributes
        fpc.mesh()       # mesh from 'shape' attribute
        fpc.splat()      # Gaussian splat using 'shape' + r/g/b

    Iteration
    ---------
        for ch in fpc:
            with open(ch.name + ".vm", "w") as f:
                f.write(ch.to_vm())
    """

    # We override __setattr__ and __getattr__ so that attribute access on the
    # Container reads/writes _attrs.  Use object.__setattr__ for anything
    # prefixed with '_' to avoid infinite recursion.

    def __init__(self, *names):
        """
        Create a Container with declared (empty) attribute slots.

        Args:
            *names: String names for the attribute slots.  All values start
                    as None; assign them with fpc.name = value.

        Example:
            fpc = fp.container("shape", "r", "g", "b")
        """
        object.__setattr__(self, '_attrs', {})
        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"fp.container() only accepts string names, "
                    f"got {type(name).__name__} ({name!r})."
                )
            self._attrs[name] = None

    # ── Attribute access ──────────────────────────────────────────────────────

    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            # Auto-creates the attribute if it doesn't exist yet
            self._attrs[name] = value

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        attrs = object.__getattribute__(self, '_attrs')
        if name in attrs:
            return attrs[name]
        raise AttributeError(
            f"Container has no attribute '{name}'. "
            f"Declare it with fpc.add('{name}') or assign directly: fpc.{name} = value"
        )

    def __delattr__(self, name):
        if name.startswith('_'):
            object.__delattr__(self, name)
        else:
            if name not in self._attrs:
                raise AttributeError(f"Container has no attribute '{name}'")
            del self._attrs[name]

    # ── Add / update / remove ─────────────────────────────────────────────────

    def add(self, *names):
        """
        Declare one or more new empty attribute slots.

        Only string names are accepted.  The slot is created with value None
        until you assign a value.  Existing attributes are left unchanged.

        Args:
            *names: String attribute names.

        Example:
            fpc.add("roughness")
            fpc.roughness = 0.4
        """
        for name in names:
            if not isinstance(name, str):
                raise TypeError(
                    f"add() expects string names, "
                    f"got {type(name).__name__} ({name!r})."
                )
            if name not in self._attrs:
                self._attrs[name] = None

    def update(self, name, value):
        """
        Update the value of an existing attribute.

        Args:
            name:  Attribute name (str).
            value: New value (float or SDF expression).

        Raises:
            KeyError: If the attribute has not been declared.

        Example:
            fpc.update("r", 0.5)
        """
        if name not in self._attrs:
            raise KeyError(
                f"Container has no attribute '{name}'. "
                f"Use fpc.add('{name}') to declare it, "
                f"or assign directly: fpc.{name} = value"
            )
        self._attrs[name] = value

    def remove(self, *names):
        """
        Remove one or more attributes from this Container.

        Args:
            *names: String attribute names to remove.

        Raises:
            KeyError: If any attribute does not exist.

        Example:
            fpc.remove("roughness", "metallic")
        """
        for name in names:
            if name not in self._attrs:
                raise KeyError(f"Container has no attribute '{name}'")
            del self._attrs[name]

    # ── Paint (proximity blend) ───────────────────────────────────────────────

    def paint(self, region_sdf, width=0.05, **attr_vals):
        """
        Proximity-blend named attribute values near or inside a region SDF.

        For each keyword argument, blends the existing attribute value toward
        the new value using a smooth ramp based on the SDF distance of the region.

        Blend weight:
            t = clamp(1 - region_sdf / width, 0, 1)
            - t = 1  inside / at the surface of region_sdf
            - t = 0  at distance ≥ width from the surface

        Result per attribute:
            new_val = old_val * (1 - t) + target_val * t

        Args:
            region_sdf: SDF expression (or Container with 'shape') defining
                        the paint region.
            width:      Transition band width in world units (default 0.05).
            **attr_vals: Attribute names and target values to blend toward.

        Returns:
            self (for chaining).

        Example:
            dot = fps.sphere(0.1).translate(0.5, 0.5, 0)
            fpc.paint(dot, r=0.1, g=0.9, b=0.1, width=0.05)
        """
        import fidgetpy.math as fpm

        # Unwrap region SDF if it's a Container
        if isinstance(region_sdf, Container):
            region = region_sdf._attrs.get('shape')
            if region is None:
                raise ValueError(
                    "paint() region Container has no 'shape' attribute"
                )
        else:
            region = region_sdf

        t = fpm.clamp(1.0 - region / width, 0.0, 1.0)

        for name, new_val in attr_vals.items():
            old_val = self._attrs.get(name, 1.0)
            if old_val is None:
                old_val = 1.0  # treat unset colour attributes as white
            self._attrs[name] = old_val * (1.0 - t) + new_val * t

        return self

    # ── Iteration ─────────────────────────────────────────────────────────────

    def __iter__(self):
        """Yield a ChannelEntry for each attribute (in insertion order)."""
        return (ChannelEntry(name, val) for name, val in self._attrs.items())

    def __len__(self):
        return len(self._attrs)

    def __contains__(self, name):
        return name in self._attrs

    # ── VM export ─────────────────────────────────────────────────────────────

    def to_vm(self):
        """
        Export all set attributes to VM format.

        Returns:
            dict {name: vm_string} for every attribute with a non-None value.

        Example:
            vms = fpc.to_vm()
            for name, vm_str in vms.items():
                with open(f"{name}.vm", "w") as f:
                    f.write(vm_str)
        """
        result = {}
        for entry in self:
            if entry.value is not None:
                result[entry.name] = entry.to_vm()
        return result

    def to_frep(self):
        """
        Export all set attributes to F-Rep format.

        Returns:
            dict {name: frep_string} for every attribute with a non-None value.

        Example:
            freps = fpc.to_frep()
            for name, frep_str in freps.items():
                with open(f"{name}.frep", "w") as f:
                    f.write(frep_str)
        """
        result = {}
        for entry in self:
            if entry.value is not None:
                result[entry.name] = entry.to_frep()
        return result

    def eval(self, points, variables=None, **kwargs):
        """
        Evaluate all set attributes at a set of points.

        Equivalent to ``fp.eval(self, points, ...)``.

        Args:
            points:    (N, 3) array of xyz coordinates.
            variables: Optional list of variable expressions. When None,
                       x/y/z are used automatically for expression attributes.
            **kwargs:  Passed to the underlying Rust eval function.

        Returns:
            dict {name: (N,) array} for every set attribute.

        Example:
            pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
            results = fpc.eval(pts)
            # {'shape': array([...]), 'r': array([...]), ...}
        """
        from fidgetpy import eval as _eval
        return _eval(self, points, variables=variables, **kwargs)

    # ── Geometry / rendering operations ───────────────────────────────────────

    def mesh(self, output_file=None, verbose=True, **kwargs):
        """
        Generate a mesh from the 'shape' attribute.

        When output_file is None, returns a Mesh object (vertices + triangles).
        When output_file is provided, writes a colored PLY using 'r', 'g', 'b'
        attributes for per-vertex colors and returns the path.

        Args:
            output_file: Path for the output .ply file, or None to return PyMesh.
            verbose:     Print progress when writing a file. Default True.
            **kwargs:    Passed to fp.mesh() (e.g. depth=6, numpy=True).

        Returns:
            PyMesh: when output_file is None.
            str:    path to the written .ply file when output_file is set.

        Raises:
            ValueError: If 'shape' attribute is not set.

        Examples:
            m = fpc.mesh(depth=5)                       # returns PyMesh
            fpc.mesh(output_file="sphere.ply", depth=6) # writes colored PLY
        """
        from fidgetpy import mesh as _mesh
        return _mesh(self, output_file=output_file, verbose=verbose, **kwargs)

    def splat(self, output_file=None, **kwargs):
        """
        Generate a Gaussian Splatting representation from this Container.

        Uses the 'shape' attribute for geometry.
        Uses 'r', 'g', 'b' attributes for colour (defaults to 1.0 = white).

        When output_file is None, returns a Gaussians object for inspection.
        When output_file is provided, writes a .ply file and returns the path.

        Args:
            output_file: Path for the output .ply file, or None to return
                         a Gaussians object. Default: None.
            **kwargs:    Passed to the splat function (size, domain, etc.).

        Returns:
            Gaussians: when output_file is None.
            str:       path to the written .ply file when output_file is set.

        Raises:
            ValueError: If 'shape' attribute is not set.
        """
        shape = self._attrs.get('shape')
        if shape is None:
            raise ValueError(
                "Container needs a 'shape' attribute for splat(). "
                "Set it with: fpc.shape = some_sdf"
            )
        r = self._attrs.get('r')
        g = self._attrs.get('g')
        b = self._attrs.get('b')
        r = 1.0 if r is None else r
        g = 1.0 if g is None else g
        b = 1.0 if b is None else b
        from fidgetpy.splat import splat as _splat_fn
        return _splat_fn(shape, output_file=output_file, color=(r, g, b), **kwargs)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        if not self._attrs:
            return "Container (empty)"
        lines = []
        for name, val in self._attrs.items():
            if val is None:
                lines.append(f"{name}: (unset)")
            else:
                lines.append(f"{name}: {val}")
        return "\n".join(lines)
