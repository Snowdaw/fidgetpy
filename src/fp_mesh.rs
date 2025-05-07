// src/fp_mesh.rs
// Meshing functionality for Expressions

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyList;
use ndarray::{Array2};
use numpy::{IntoPyArray};
use fidget::Error as FidgetError;
use fidget::context::{Context, Tree, TreeOp};
use fidget::context::{BinaryOpcode, UnaryOpcode};
use fidget::mesh::{Octree, Settings as MeshSettings, Mesh};
use fidget::vm::VmShape;
use fidget::render::{ThreadPool, View3};
use fidget::var::Var;
use nalgebra::Vector3;
use std::collections::HashMap;

use crate::fp_expr::PyExpr;
use crate::fp_utils::{check_for_custom_vars, parse_variable_list};
use crate::fp_utils::get_required_var_names_from_vm;
use crate::fp_context::PySDFContext; // Needed for to_vm
use std::collections::HashSet;


/// Python class representing a mesh
#[pyclass(name = "Mesh")]
pub struct PyMesh {
    #[pyo3(get)]
    pub vertices: PyObject,
    #[pyo3(get)]
    pub triangles: PyObject,
}

#[pymethods]
impl PyMesh {
    #[new]
    fn new(vertices: PyObject, triangles: PyObject) -> Self {
        PyMesh { vertices, triangles }
    }

    fn __repr__(&self, _py: Python) -> PyResult<String> {
        Ok(format!("Mesh(vertices=..., triangles=...)"))
    }
}

/// Implementation of the mesh method for PyExpr
pub fn mesh_impl(
    py: Python,
    sdf: &PyExpr,
    center: Option<&Bound<'_, PyList>>,
    scale: f64,
    depth: u8,
    threads: bool,
    variables: Option<&Bound<'_, PyList>>,
    variable_values: Option<&Bound<'_, PyList>>,
    use_numpy: bool,
    bounds_min: Option<&Bound<'_, PyList>>,
    bounds_max: Option<&Bound<'_, PyList>>,
) -> PyResult<Py<PyMesh>> {
    // Add a safety check for depth to prevent integer overflow
    // The Octree builder calculates 8^depth, which can overflow for large depth values
    const MAX_SAFE_DEPTH: u8 = 64;
    let safe_depth = if depth > MAX_SAFE_DEPTH {
        eprintln!("Warning: Capping depth from {} to {}", depth, MAX_SAFE_DEPTH);
        MAX_SAFE_DEPTH
    } else {
        depth
    };
    // 1. Check for custom variables and handle substitution if needed
    let tree = if check_for_custom_vars(&*sdf.tree) {
        // If we have custom variables but no substitution values, return an error
        if variables.is_none() || variable_values.is_none() {
            return Err(PyValueError::new_err(
                "Expression contains custom variables. Either provide variables and variable_values or use an expression with only x, y, z.",
            ));
        }
        
        // Parse the variables and values
        let py_vars = parse_variable_list(variables)?;
// --- Validate Variable Mapping using VM (before substitution) ---
        // Generate VM string from the original expression
        let mut ctx_vm = PySDFContext::new();
        let vm_str = ctx_vm.to_vm(sdf)?; // Use the original PyExpr

        // Get required variable names from VM string
        let mut required_var_names = get_required_var_names_from_vm(&vm_str);

        // Get provided variable names (use name if available, otherwise format Var)
        // Note: We use py_vars here which is defined just above
        let mut provided_var_names: HashSet<String> = py_vars.iter().map(|pv| {
            pv.name.clone().unwrap_or_else(|| format!("{:?}", pv.var)) // Fallback to Var debug format if no name
        }).collect();

        // Ignore x, y, z for meshing validation
        required_var_names.remove("x");
        required_var_names.remove("y");
        required_var_names.remove("z");
        provided_var_names.remove("x");
        provided_var_names.remove("y");
        provided_var_names.remove("z");

        // Check for missing variables (excluding x, y, z)
        let missing_vars: Vec<String> = required_var_names
            .difference(&provided_var_names)
            .cloned()
            .collect();

        // Check for extra variables (excluding x, y, z)
        let extra_vars: Vec<String> = provided_var_names
            .difference(&required_var_names)
            .cloned()
            .collect();

        // If there are either missing or unused variables, generate a combined error message
        if !missing_vars.is_empty() || !extra_vars.is_empty() {
            let mut error_message = String::from("Missing or unused variable(s) found in mapping:");
            
            if !missing_vars.is_empty() {
                error_message.push_str(&format!("\nMissing: {}", missing_vars.join(", ")));
            }
            
            if !extra_vars.is_empty() {
                error_message.push_str(&format!("\nUnused: {}", extra_vars.join(", ")));
            }
            
            return Err(PyValueError::new_err(error_message));
        }
        // --- End Validation ---

        let rust_vars: Vec<Var> = py_vars.iter().map(|pv| pv.var.clone()).collect();
        
        // Extract values from the variable_values list
        let values_list = variable_values.unwrap();
        if values_list.len() != rust_vars.len() {
            return Err(PyValueError::new_err(format!(
                "Number of variable values ({}) must match the number of variables provided ({})",
                values_list.len(), rust_vars.len()
            )));
        }
        
        // Create a map of variables to their constant values
        let mut var_values = HashMap::new();
        
        // Add values for custom variables
        for (i, var) in rust_vars.iter().enumerate() {
            if matches!(var, Var::V(_)) {
                // Extract the value for this variable
                let value: f64 = values_list.get_item(i)?.extract()?;
                var_values.insert(*var, value);
            }
        }
        
        // Create a new tree with constants substituted for variables
        // We'll use a recursive function to traverse the tree
        // Recursive function to substitute variables by rebuilding the tree structure
        fn substitute_vars(tree_op: &TreeOp, var_values: &HashMap<Var, f64>) -> Tree {
            match tree_op {
                TreeOp::Input(var) => {
                    // If this is a custom variable with a value in our map, replace it with a constant
                    if let Some(value) = var_values.get(var) {
                        Tree::constant(*value)
                    } else {
                        // Otherwise, keep the original standard variable (x, y, z)
                        match var {
                            Var::X => Tree::x(),
                            Var::Y => Tree::y(),
                            Var::Z => Tree::z(),
                            // If it's a custom var without a value, return 0.0
                            // This case shouldn't ideally happen if check_for_custom_vars is accurate
                            // and values are provided, but provides a fallback.
                            _ => Tree::constant(0.0),
                        }
                    }
                },
                TreeOp::Const(val) => Tree::constant(*val),
                TreeOp::Unary(op, arg_op_arc) => {
                    // Recursively substitute in the argument
                    let new_arg_tree = substitute_vars(arg_op_arc, var_values);
                    // Apply the unary operation using Tree methods
                    match op {
                        UnaryOpcode::Square => new_arg_tree.square(),
                        UnaryOpcode::Sqrt => new_arg_tree.sqrt(),
                        UnaryOpcode::Neg => new_arg_tree.neg(),
                        UnaryOpcode::Recip => new_arg_tree.recip(),
                        UnaryOpcode::Abs => new_arg_tree.abs(),
                        UnaryOpcode::Sin => new_arg_tree.sin(),
                        UnaryOpcode::Cos => new_arg_tree.cos(),
                        UnaryOpcode::Tan => new_arg_tree.tan(),
                        UnaryOpcode::Asin => new_arg_tree.asin(),
                        UnaryOpcode::Acos => new_arg_tree.acos(),
                        UnaryOpcode::Atan => new_arg_tree.atan(),
                        UnaryOpcode::Exp => new_arg_tree.exp(),
                        UnaryOpcode::Ln => new_arg_tree.ln(),
                        UnaryOpcode::Floor => new_arg_tree.floor(),
                        UnaryOpcode::Ceil => new_arg_tree.ceil(),
                        UnaryOpcode::Round => new_arg_tree.round(),
                        UnaryOpcode::Not => new_arg_tree.not(),
                    }
                },
                TreeOp::Binary(op, lhs_op_arc, rhs_op_arc) => {
                    // Recursively substitute in arguments
                    let new_lhs_tree = substitute_vars(lhs_op_arc, var_values);
                    let new_rhs_tree = substitute_vars(rhs_op_arc, var_values);
                    // Apply the binary operation using Tree methods
                    match op {
                        BinaryOpcode::Add => new_lhs_tree + new_rhs_tree,
                        BinaryOpcode::Sub => new_lhs_tree - new_rhs_tree,
                        BinaryOpcode::Mul => new_lhs_tree * new_rhs_tree,
                        BinaryOpcode::Div => new_lhs_tree / new_rhs_tree,
                        BinaryOpcode::Min => new_lhs_tree.min(new_rhs_tree),
                        BinaryOpcode::Max => new_lhs_tree.max(new_rhs_tree),
                        BinaryOpcode::Mod => new_lhs_tree.modulo(new_rhs_tree),
                        BinaryOpcode::Atan => new_lhs_tree.atan2(new_rhs_tree), // Fidget uses Atan for atan2
                        BinaryOpcode::Compare => new_lhs_tree.compare(new_rhs_tree),
                        BinaryOpcode::And => new_lhs_tree.and(new_rhs_tree),
                        BinaryOpcode::Or => new_lhs_tree.or(new_rhs_tree),
                    }
                },
                TreeOp::RemapAxes { target: target_op_arc, x: x_op_arc, y: y_op_arc, z: z_op_arc } => {
                    let new_target = substitute_vars(target_op_arc, var_values);
                    let new_x = substitute_vars(x_op_arc, var_values);
                    let new_y = substitute_vars(y_op_arc, var_values);
                    let new_z = substitute_vars(z_op_arc, var_values);
                    new_target.remap_xyz(new_x, new_y, new_z)
                },
                TreeOp::RemapAffine { target: target_op_arc, mat } => {
                    let new_target = substitute_vars(target_op_arc, var_values);
                    new_target.remap_affine(mat.clone()) // Clone the matrix
                },
            }
        }

        // Apply the substitution by traversing the original TreeOp structure
        substitute_vars(&*sdf.tree, &var_values)
    } else {
        // No custom variables, use the original tree
        sdf.tree.clone()
    };

    // 2. Create Fidget context and shape
    let mut ctx = Context::new();
    let node = ctx.import(&tree);
    let initial_shape = VmShape::new(&ctx, node)
        .map_err(|e: FidgetError| PyValueError::new_err(format!("Failed to create Fidget shape: {}", e)))?;

    // 3. Determine transformation and bounding box based on inputs
    
    // We have two options for specifying the meshing region:
    // 4. Create MeshSettings with a view based on the input parameters
    let thread_pool = if threads { Some(&ThreadPool::Global) } else { None };
    
    // Determine the View3 based on inputs

    // Add safeguard: Check if both bounds and center are provided
    if bounds_min.is_some() && bounds_max.is_some() && center.is_some() {
        // Raise an error because providing both bounds and center/scale is ambiguous
        // and the code prioritizes bounds, silently ignoring center/scale.
        // This forces the user to be explicit about their intended meshing region.
        return Err(PyValueError::new_err(
            "Ambiguous meshing region: Cannot provide both 'bounds_min'/'bounds_max' and 'center'. \
             Please use either bounds OR center/scale to define the region. \
             If bounds are provided, they take precedence and 'center'/'scale' are ignored."
        ));
    }
    // Note: We don't explicitly check for `scale` in the condition above because it always has a
    // value passed from Python. The existing logic correctly ignores `scale` when bounds are used.
    // The error message clarifies that both center and scale are ignored if bounds are present.

    let view = if bounds_min.is_some() && bounds_max.is_some() {
        // Option 1: Using bounds directly
        let bounds_min_list = bounds_min.unwrap();
        let bounds_max_list = bounds_max.unwrap();
        
        if bounds_min_list.len() != 3 || bounds_max_list.len() != 3 {
            return Err(PyValueError::new_err(
                "bounds_min and bounds_max must each be a list with exactly 3 elements"
            ));
        }
        
        let min_x = bounds_min_list.get_item(0)?.extract::<f64>()? as f32;
        let min_y = bounds_min_list.get_item(1)?.extract::<f64>()? as f32;
        let min_z = bounds_min_list.get_item(2)?.extract::<f64>()? as f32;
        
        let max_x = bounds_max_list.get_item(0)?.extract::<f64>()? as f32;
        let max_y = bounds_max_list.get_item(1)?.extract::<f64>()? as f32;
        let max_z = bounds_max_list.get_item(2)?.extract::<f64>()? as f32;
        
        if min_x >= max_x || min_y >= max_y || min_z >= max_z {
            return Err(PyValueError::new_err(
                "bounds_max must be greater than bounds_min in all dimensions"
            ));
        }
        
        // Calculate center and scale for View3
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;
        let center_z = (min_z + max_z) / 2.0;
        let center_vec = Vector3::new(center_x, center_y, center_z);
        
        let width = max_x - min_x;
        let height = max_y - min_y;
        let depth_dim = max_z - min_z; // Renamed from 'depth' to avoid conflict
        let max_dim = width.max(height).max(depth_dim);
        
        // Scale maps the [-1, 1] world cube (size 2) to the max dimension of the bounds
        let view_scale = max_dim / 2.0;
        
        View3::from_center_and_scale(center_vec, view_scale)
        
    } else {
        // Option 2: Legacy approach with center and scale
        
        // Create default center [0.0, 0.0, 0.0] if None is provided
        let center_vec = if let Some(center_list) = center {
            if center_list.len() != 3 {
                return Err(PyValueError::new_err(format!(
                    "Center must be a list with exactly 3 elements, got {} elements",
                    center_list.len()
                )));
            }
            let x = center_list.get_item(0)?.extract::<f64>()? as f32;
            let y = center_list.get_item(1)?.extract::<f64>()? as f32;
            let z = center_list.get_item(2)?.extract::<f64>()? as f32;
            Vector3::new(x, y, z)
        } else {
            Vector3::new(0.0, 0.0, 0.0)
        };
    
        let scale_f32 = scale as f32;
        if scale_f32 <= 0.0 {
            return Err(PyValueError::new_err("Scale must be a positive value"));
        }
        
        // Use the provided center and scale directly for the view
        View3::from_center_and_scale(center_vec, scale_f32)
    };
    
    
    // Create meshing settings with our custom view
    let settings = MeshSettings {
        depth: safe_depth,
        threads: thread_pool,
        view,
    };

    // 6. Build Octree and Mesh using the original shape and settings
    let mesh_result: Mesh = py.allow_threads(move || {
        // Pass the original initial_shape here
        let octree = Octree::build(&initial_shape, settings);
        // walk_dual uses the view from settings internally
        octree.walk_dual(settings)
    });

    // 7. Convert mesh data to ndarray::Array2 or Python lists
    let num_vertices = mesh_result.vertices.len();
    let num_triangles = mesh_result.triangles.len();

    // 8. Create and return Python Mesh object based on input type
    let (vertices, triangles): (PyObject, PyObject) = if !use_numpy {
        // Convert to Python lists - create them directly as nested Python lists
        let mut vertices_data = Vec::with_capacity(num_vertices);
        for i in 0..num_vertices {
            vertices_data.push(vec![
                mesh_result.vertices[i][0],
                mesh_result.vertices[i][1],
                mesh_result.vertices[i][2]
            ]);
        }
        
        let mut triangles_data = Vec::with_capacity(num_triangles);
        for i in 0..num_triangles {
            triangles_data.push(vec![
                mesh_result.triangles[i][0],
                mesh_result.triangles[i][1],
                mesh_result.triangles[i][2]
            ]);
        }
        
        // Convert the nested Rust vectors to Python lists
        (vertices_data.into_pyobject(py)?.into(), triangles_data.into_pyobject(py)?.into())
    } else {
        // Use numpy arrays
        let num_vertices = mesh_result.vertices.len();
        let vertices_array = Array2::from_shape_fn((num_vertices, 3), |(i, j)| {
            mesh_result.vertices[i][j]
        });

        let num_triangles = mesh_result.triangles.len();
        let triangles_array = Array2::from_shape_fn((num_triangles, 3), |(i, j)| {
             mesh_result.triangles[i][j]
        });

        // Create the PyArray bound references using the ndarray
        let vertices_pyarray_bound = vertices_array.into_pyarray(py);
        let triangles_pyarray_bound = triangles_array.into_pyarray(py);

        let vertices_np: PyObject = vertices_pyarray_bound.into();
        let triangles_np: PyObject = triangles_pyarray_bound.into();

        (vertices_np, triangles_np)
    };
    
    // Create the mesh object
    let mesh = PyMesh::new(vertices, triangles);
    Ok(Py::new(py, mesh)?)
}
