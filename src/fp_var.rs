use pyo3::prelude::*;
use pyo3::pyclass;
use fidget::var::Var;
use fidget::context::Tree;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::fp_expr::PyExpr;
use crate::fp_utils::{SdfVarOrFloat, get_tree_and_names_from_other, merge_maps};


/// Represents a variable in an SDF expression.
/// This is a minimal wrapper around Fidget's Var type.
#[pyclass(name = "Var")]
#[derive(Clone)]
pub struct PyVar {
    pub var: Var,
    #[pyo3(get)]
    pub name: Option<String>,
}

#[pymethods]
impl PyVar {
    /// Create a new variable with an optional name
    #[new]
    #[pyo3(signature = (name=None))]
    pub fn new(name: Option<String>) -> Self {
        if let Some(n) = &name {
            match n.to_lowercase().as_str() {
                "x" => return PyVar::x(),
                "y" => return PyVar::y(),
                "z" => return PyVar::z(),
                _ => (),
            }
        }
        PyVar { var: Var::new(), name }
    }

    /// Create the X coordinate variable
    #[staticmethod]
    pub fn x() -> Self {
        PyVar { var: Var::X, name: Some("x".to_string()) }
    }
    
    /// Create the Y coordinate variable
    #[staticmethod]
    pub fn y() -> Self {
        PyVar { var: Var::Y, name: Some("y".to_string()) }
    }
    
    /// Create the Z coordinate variable
    #[staticmethod]
    pub fn z() -> Self {
        PyVar { var: Var::Z, name: Some("z".to_string()) }
    }

    /// Compare variables for equality
    fn __richcmp__(&self, other: &Bound<'_, PyAny>, op: pyo3::basic::CompareOp) -> PyResult<bool> {
        if let Ok(other_var) = other.extract::<PyRef<PyVar>>() {
            match op {
                pyo3::basic::CompareOp::Eq => Ok(self.var == other_var.var),
                pyo3::basic::CompareOp::Ne => Ok(self.var != other_var.var),
                _ => Ok(false),
            }
        } else {
            Ok(false)
        }
    }

    /// Hash function for variables
    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.var.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert this variable to an SDF expression
    /// This is the key method that allows variables to be used in expressions
    pub fn to_expr(&self) -> PyExpr {
        PyExpr::from_var(self)
    }
    
    /// Allow variables to be called directly to convert them to expressions
    /// This enables syntax like: x()(y) instead of x.to_expr()(y)
    fn __call__(&self) -> PyExpr {
        self.to_expr()
    }
    
    // Python operator overloading methods that convert to PyExpr
    
    fn __add__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self_tree + other_tree,
            var_names: merge_maps(&self_names, &other_names),
        }
    }
    
    fn __radd__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: other_tree + self_tree,
            var_names: merge_maps(&other_names, &self_names),
        }
    }
    
    fn __sub__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self_tree - other_tree,
            var_names: merge_maps(&self_names, &other_names),
        }
    }
    
    fn __rsub__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: other_tree - self_tree,
            var_names: merge_maps(&other_names, &self_names),
        }
    }
    
    fn __mul__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self_tree * other_tree,
            var_names: merge_maps(&self_names, &other_names),
        }
    }
    
    fn __rmul__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: other_tree * self_tree,
            var_names: merge_maps(&other_names, &self_names),
        }
    }
    
    fn __truediv__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self_tree / other_tree,
            var_names: merge_maps(&self_names, &other_names),
        }
    }
    
    fn __rtruediv__(&self, other: SdfVarOrFloat) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: other_tree / self_tree,
            var_names: merge_maps(&other_names, &self_names),
        }
    }
    
    fn __neg__(&self) -> PyExpr {
        let self_tree = Tree::from(self.var.clone());
        let mut self_names = HashMap::new();
        if let Some(name) = &self.name {
            self_names.insert(self.var.clone(), name.clone());
        }
        PyExpr {
            tree: self_tree.neg(),
            var_names: self_names
        }
    }

    /// String representation of the variable
    fn __repr__(&self) -> String {
        if let Some(name) = &self.name {
            format!("Var({})", name)
        } else {
            format!("Var({:?})", self.var)
        }
    }
}