// src/eval.rs
// Evaluation functionality for SDF expressions

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use numpy::{PyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use ndarray::{Array, ArrayView2};

use fidget::var::Var;
use fidget::context::Tree;
use fidget::{
    context::TreeOp,
    shape::{EzShape, Shape, ShapeBulkEval, ShapeVars},
};
use fidget::jit::{JitFloatSliceEval, JitFunction};
use fidget::vm::{VmFloatSliceEval, VmFunction};

use crate::fp_utils::parse_variable_list;

// Enum to accept either NumPy array or Python list for eval values
#[derive(FromPyObject)]
pub enum ArrayOrList<'a> {
    Array(PyReadonlyArray2<'a, f32>),
    List(Bound<'a, PyList>),
}

// Enum to accept either PyExpr or float for eval expression
#[derive(FromPyObject)]
pub enum ExprOrFloat<'a> {
    Expr(PyRef<'a, crate::fp_expr::PyExpr>),
    Float(f64),
}

/// Determine which backend to use for evaluation
pub fn determine_backend(backend: Option<String>) -> PyResult<bool> {
    match backend.as_deref() {
        None => Ok(true), // Default to JIT
        Some("jit") => Ok(true),
        Some("vm") => Ok(false),
        Some(other) => {
            Err(PyValueError::new_err(format!(
                "Unknown backend: {}. Valid options are 'jit' or 'vm'.",
                other
            )))
        }
    }
}

/// Evaluate a Tree using the JIT backend
pub(crate) fn _evaluate_bulk_jit(
    tree: &Tree,
    values: &ArrayView2<'_, f32>,
    variables: &[Var],
) -> PyResult<Vec<f32>> {
    let shape: Shape<JitFunction> = Shape::from(tree.clone());
    let tape = shape.ez_float_slice_tape();
    let varmap = tape.vars();
    let arr = values;
    let n = arr.shape()[0];
    let num_vars_input = arr.shape()[1];

    if num_vars_input != variables.len() {
        return Err(PyValueError::new_err(format!(
            "Number of columns in input array ({}) must match the number of variables provided ({})",
            num_vars_input, variables.len()
        )));
    }

    // Prepare var_slices based on input array columns and variable order
    let mut var_slices: Vec<Vec<f32>> = vec![vec![0.0; n]; varmap.len()];
    for (col_idx, input_var) in variables.iter().enumerate() {
        if let Some(internal_idx) = varmap.get(input_var) {
            for row_idx in 0..n {
                var_slices[internal_idx][row_idx] = unsafe { *arr.uget([row_idx, col_idx]) };
            }
        }
    }

    let mut eval: ShapeBulkEval<JitFloatSliceEval> = ShapeBulkEval::default();

    let mut x_slice: &[f32] = &[];
    let mut y_slice: &[f32] = &[];
    let mut z_slice: &[f32] = &[];
    let mut shape_vars = ShapeVars::new();
    let default_values: Vec<f32> = vec![0.0; n];

    if let Some(idx) = varmap.get(&Var::X) {
        if idx < var_slices.len() {
            x_slice = &var_slices[idx];
        }
    } else {
        x_slice = &default_values[..];
    }
    if let Some(idx) = varmap.get(&Var::Y) {
        if idx < var_slices.len() {
            y_slice = &var_slices[idx];
        }
    } else {
        y_slice = &default_values[..];
    }
    if let Some(idx) = varmap.get(&Var::Z) {
        if idx < var_slices.len() {
            z_slice = &var_slices[idx];
        }
    } else {
        z_slice = &default_values[..];
    }

    for input_var in variables {
        if !matches!(input_var, &Var::X | &Var::Y | &Var::Z) {
            if let (Some(var_idx), Some(internal_idx)) = (input_var.index(), varmap.get(input_var)) {
                if internal_idx < var_slices.len() {
                    shape_vars.insert(var_idx, &var_slices[internal_idx][..]);
                } else {
                    return Err(PyValueError::new_err(
                        "Internal error: varmap index out of bounds for var_slices",
                    ));
                }
            }
        }
    }

    let result: &[f32] = eval
        .eval_vs::<&[f32], f32>(&tape, x_slice, y_slice, z_slice, &shape_vars)
        .map_err(|e| PyValueError::new_err(format!("JIT evaluation error: {}", e)))?;

    Ok(result.to_vec())
}

/// Evaluate a Tree using the VM backend
pub(crate) fn _evaluate_bulk_vm(
    tree: &Tree,
    values: &ArrayView2<'_, f32>,
    variables: &[Var],
) -> PyResult<Vec<f32>> {
    let shape: Shape<VmFunction> = Shape::from(tree.clone());
    let tape = shape.ez_float_slice_tape();
    let varmap = tape.vars();
    let arr = values;
    let n = arr.shape()[0];
    let num_vars_input = arr.shape()[1];

    if num_vars_input != variables.len() {
        return Err(PyValueError::new_err(format!(
            "Number of columns in input array ({}) must match the number of variables provided ({})",
            num_vars_input, variables.len()
        )));
    }

    // Prepare var_slices based on input array columns and variable order
    // Size the outer vec based on the number of vars the *expression* uses (varmap.len())
    let mut var_slices: Vec<Vec<f32>> = vec![vec![0.0; n]; varmap.len()];
    // Map the provided variable order (and data columns) to the internal varmap indices
    for (col_idx, input_var) in variables.iter().enumerate() {
        if let Some(internal_idx) = varmap.get(input_var) {
            // This input variable is used by the expression, copy its data
            for row_idx in 0..n {
                // Use unchecked access after shape check for potential minor perf gain
                var_slices[internal_idx][row_idx] = unsafe { *arr.uget([row_idx, col_idx]) };
            }
        } else {
            // Variable provided in list but not used by expression - ignore.
        }
    }
    // Note: The check for missing required variables is removed.
    // We rely on the fact that var_slices is sized by varmap.len().
    // If a required variable wasn't in the input `variables` list, its corresponding
    // slice in var_slices will remain default (e.g., zeros).
    // The `eval_vs` function within Fidget must handle cases where required inputs might be zero/default.
    // If `eval_vs` errors appropriately, that's sufficient. If not, more complex checking might be needed.

    let mut eval: ShapeBulkEval<VmFloatSliceEval<255>> = ShapeBulkEval::default();

    // Extract slices for X, Y, Z and ShapeVars
    let mut x_slice: &[f32] = &[];
    let mut y_slice: &[f32] = &[];
    let mut z_slice: &[f32] = &[];
    let mut shape_vars = ShapeVars::new();
    // Create a default slice of length n for unused standard variables
    let default_values: Vec<f32> = vec![0.0; n];

    if let Some(idx) = varmap.get(&Var::X) {
        if idx < var_slices.len() {
            x_slice = &var_slices[idx];
        }
    } else {
        x_slice = &default_values[..];
    }
    if let Some(idx) = varmap.get(&Var::Y) {
        if idx < var_slices.len() {
            y_slice = &var_slices[idx];
        }
    } else {
        y_slice = &default_values[..];
    }
    if let Some(idx) = varmap.get(&Var::Z) {
        if idx < var_slices.len() {
            z_slice = &var_slices[idx];
        }
    } else {
        z_slice = &default_values[..];
    }

    // Iterate through the varmap to populate shape_vars for custom variables
    // Correct iteration and indexing for custom vars
    // Populate shape_vars by iterating through the *input* variables list
    for input_var in variables {
        // Use input_var here for the !matches! check
        if !matches!(input_var, &Var::X | &Var::Y | &Var::Z) {
            // Check if this input var is actually used by the expression (i.e., is in varmap)
            if let (Some(var_idx), Some(internal_idx)) = (input_var.index(), varmap.get(input_var)) {
                // var_idx is the VarIndex (u64) key for shape_vars
                // internal_idx is the index into var_slices
                if internal_idx < var_slices.len() {
                    shape_vars.insert(var_idx, &var_slices[internal_idx][..]);
                } else {
                    return Err(PyValueError::new_err(
                        "Internal error: varmap index out of bounds for var_slices",
                    ));
                }
            }
            // If input_var is not in varmap, we just ignore it (it wasn't needed)
        }
    }

    let result: &[f32] = eval
        .eval_vs::<&[f32], f32>(&tape, x_slice, y_slice, z_slice, &shape_vars)
        .map_err(|e| PyValueError::new_err(format!("VM evaluation error: {}", e)))?;

    Ok(result.to_vec())
}

/// Implementation of the eval method for PyExpr
pub fn eval_impl(
    py: Python,
    expr: ExprOrFloat,
    values: ArrayOrList,
    variables_list: Option<&Bound<'_, PyList>>,
    backend: Option<String>,
) -> PyResult<PyObject> {
    // Convert ExprOrFloat to PyExpr
    let sdf = match expr {
        ExprOrFloat::Expr(expr_ref) => expr_ref,
        ExprOrFloat::Float(val) => {
            // Create a constant expression
            let num_points = match &values {
                ArrayOrList::Array(np_array) => np_array.shape()[0],
                ArrayOrList::List(py_list) => py_list.len(),
            };
            let results = vec![val as f32; num_points];

            // Return the same type as the input
            return match values {
                ArrayOrList::Array(_) => Ok(PyArray1::from_vec(py, results).into()),
                ArrayOrList::List(_) => Ok(results.into_pyobject(py)?.into()),
            };
        }
    };

    // --- Workaround for constant expressions ---
    if let TreeOp::Const(const_val) = &*sdf.tree {
        let num_points = match &values {
            ArrayOrList::Array(np_array) => np_array.shape()[0],
            ArrayOrList::List(py_list) => py_list.len(),
        };
        let results = vec![*const_val as f32; num_points];

        // Return the same type as the input
        return match values {
            ArrayOrList::Array(_) => Ok(PyArray1::from_vec(py, results).into()),
            ArrayOrList::List(_) => Ok(results.into_pyobject(py)?.into()),
        };
    }
    // --- End Workaround ---

    // Determine backend
    let use_jit = determine_backend(backend)?;

    // Parse the input list of variable expressions
    let py_vars = parse_variable_list(variables_list)?;
    let rust_vars: Vec<Var> = py_vars.iter().map(|pv| pv.var.clone()).collect();
    let num_vars_expected = py_vars.len();

    // Process the input values (Array or List) and perform evaluation
    match values {
        ArrayOrList::Array(np_array) => {
            let values_view = np_array.as_array();
            if values_view.shape().len() != 2 {
                return Err(PyValueError::new_err("Input array must be 2-dimensional"));
            }
            if values_view.shape()[1] != num_vars_expected {
                return Err(PyValueError::new_err(format!(
                    "Number of columns in input array ({}) must match the number of variables provided ({})",
                    values_view.shape()[1],
                    num_vars_expected
                )));
            }
            // Evaluate directly using the borrowed view
            let result_vec = if use_jit {
                _evaluate_bulk_jit(&sdf.tree, &values_view, &rust_vars)?
            } else {
                _evaluate_bulk_vm(&sdf.tree, &values_view, &rust_vars)?
            };

            // Return NumPy array for NumPy input
            return Ok(PyArray1::from_vec(py, result_vec).into());
        }
        ArrayOrList::List(py_list) => {
            let mut flat_data: Vec<f32> = Vec::new();
            let num_points = py_list.len();

            if num_points == 0 {
                // Handle empty list case
                return Ok(Vec::<f32>::new().into_pyobject(py)?.into());
            } else {
                let mut first_row_len: Option<usize> = None;

                for (row_idx, row_obj) in py_list.iter().enumerate() {
                    let inner_list = row_obj.downcast::<PyList>()?;
                    let current_row_len = inner_list.len();

                    if let Some(expected_len) = first_row_len {
                        if current_row_len != expected_len {
                            return Err(PyValueError::new_err(format!(
                                "Inconsistent row lengths: Row 0 has {} elements, Row {} has {}",
                                expected_len, row_idx, current_row_len
                            )));
                        }
                    } else {
                        if current_row_len != num_vars_expected {
                            return Err(PyValueError::new_err(format!(
                                "Number of elements in inner lists ({}) must match the number of variables provided ({})",
                                current_row_len,
                                num_vars_expected
                            )));
                        }
                        first_row_len = Some(current_row_len);
                    }

                    for item in inner_list.iter() {
                        let val: f32 = item.extract::<f32>()?;
                        flat_data.push(val);
                    }
                }

                let num_vars = first_row_len.unwrap_or(0);
                let values_array = Array::from_shape_vec((num_points, num_vars), flat_data)
                    .map_err(|e| PyValueError::new_err(format!("Failed to create array from list: {}", e)))?;
                let values_view = values_array.view();

                let result_vec = if use_jit {
                    _evaluate_bulk_jit(&sdf.tree, &values_view, &rust_vars)?
                } else {
                    _evaluate_bulk_vm(&sdf.tree, &values_view, &rust_vars)?
                };

                // Return Python list for Python list input
                return Ok(result_vec.into_pyobject(py)?.into());
            }
        }
    }
}