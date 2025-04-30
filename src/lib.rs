use pyo3::prelude::*;
use pyo3::types::PyList; // Import PyList here

mod fp_var;
mod fp_expr;
mod fp_utils;
mod fp_eval;
mod fp_mesh;
mod fp_stl;
mod fp_context;


use fp_var::PyVar;
use fp_expr::PyExpr;
use crate::fp_mesh::PyMesh;

// Core coordinate functions
#[pyfunction]
fn x() -> PyExpr {
    PyExpr::x()
}

#[pyfunction]
fn y() -> PyExpr {
    PyExpr::y()
}

#[pyfunction]
fn z() -> PyExpr {
    PyExpr::z()
}

#[pyfunction(name = "var")]
fn create_var(name: &str) -> PyResult<PyExpr> {
    match name {
        "x" | "X" => Ok(PyExpr::x()),
        "y" | "Y" => Ok(PyExpr::y()),
        "z" | "Z" => Ok(PyExpr::z()),
        _ => {
            // Create a custom variable
            let var = PyVar::new(Some(name.to_string()));
            Ok(PyExpr::from_var(&var))
        }
    }
}

// Removed constant function to allow seamless use of numbers

// Direct evaluation function
#[pyfunction]
#[pyo3(signature = (expr, values, variables=None, backend=None))]
fn eval(
    py: Python,
    expr: crate::fp_eval::ExprOrFloat,
    values: crate::fp_eval::ArrayOrList,
    variables: Option<&Bound<'_, PyList>>,
    backend: Option<String>,
) -> PyResult<PyObject> {
    // Call the implementation function which now handles variable list parsing
    crate::fp_eval::eval_impl(py, expr, values, variables, backend)
}

/// Mesh an SDF expression into a triangle mesh.
///
/// Args:
///     expr: The SDF expression to mesh.
///     center: Optional center point as [x, y, z]. Defaults to [0, 0, 0].
///     scale: Scale factor for the result (larger values make the shape bigger). Defaults to 1.0.
///            Note: The meshing process uses internal bounds of approximately [-1, 1] in each dimension.
///            If your shape is scaled too large, it may not be meshed correctly.
///     depth: Octree depth for meshing (higher values give more detail). Defaults to 4.
///     threads: Whether to use multithreading. Defaults to true.
///     variables: Optional list of custom variables in the expression.
///     variable_values: Optional list of values for the custom variables.
///     numpy: Whether to return numpy arrays (True) or lists (False). Defaults to False.
///     bounds_min: Optional minimum bounds for meshing as [x, y, z]. If provided, bounds_max must also be provided.
///                 This is an alternative to using center and scale, and provides more intuitive control.
///     bounds_max: Optional maximum bounds for meshing as [x, y, z]. If provided, bounds_min must also be provided.
///
/// Returns:
///     A PyMesh object containing vertices and triangles.
#[pyfunction]
#[pyo3(signature = (expr, center=None, scale=1.0, depth=4, threads=true, variables=None, variable_values=None, numpy=false, bounds_min=None, bounds_max=None))]
fn mesh(
    py: Python,
    expr: &PyExpr,
    center: Option<&Bound<'_, PyList>>,
    scale: f64,
    depth: u8,
    threads: bool,
    variables: Option<&Bound<'_, PyList>>,
    variable_values: Option<&Bound<'_, PyList>>,
    numpy: bool,
    bounds_min: Option<&Bound<'_, PyList>>,
    bounds_max: Option<&Bound<'_, PyList>>,
) -> PyResult<Py<crate::fp_mesh::PyMesh>> {
    // Call the implementation function with the updated signature
    crate::fp_mesh::mesh_impl(
        py, expr, center, scale, depth, threads, variables, variable_values, numpy,
        bounds_min, bounds_max
    )
}

#[pyfunction]
fn save_stl(py: Python, mesh: &PyMesh, filepath: String) -> PyResult<()> {
    crate::fp_stl::save_stl(py, mesh, filepath)
}

// Import/Export functions
#[pyfunction]
fn from_vm(text: String) -> PyResult<PyExpr> {
    let ctx = fp_context::PySDFContext::new();
    ctx.from_vm(text)
}

#[pyfunction]
fn to_vm(expr: &PyExpr) -> PyResult<String> {
    let mut ctx = fp_context::PySDFContext::new();
    ctx.to_vm(expr)
}

#[pyfunction]
fn from_frep(text: String) -> PyResult<PyExpr> {
    let ctx = fp_context::PySDFContext::new();
    ctx.from_frep(text)
}

#[pyfunction]
fn to_frep(expr: &PyExpr) -> PyResult<String> {
    let ctx = fp_context::PySDFContext::new();
    ctx.to_frep(expr)
}

#[pymodule]
fn fidgetpy(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // 1. Core coordinate and variable functions
    m.add_function(wrap_pyfunction!(x, m)?)?;
    m.add_function(wrap_pyfunction!(y, m)?)?;
    m.add_function(wrap_pyfunction!(z, m)?)?;
    m.add_function(wrap_pyfunction!(create_var, m)?)?;
    // Removed constant function registration
    
    // 2. Evaluation and meshing functions
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(mesh, m)?)?;
    m.add_function(wrap_pyfunction!(save_stl, m)?)?;

    // 3. Import/Export functions
    m.add_function(wrap_pyfunction!(from_vm, m)?)?;
    m.add_function(wrap_pyfunction!(to_vm, m)?)?;
    m.add_function(wrap_pyfunction!(from_frep, m)?)?;
    m.add_function(wrap_pyfunction!(to_frep, m)?)?;
    
    // 5. Add the mesh class
    m.add_class::<crate::fp_mesh::PyMesh>()?;
    
    Ok(())
}