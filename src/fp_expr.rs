// src/fp_expr.rs
// Core SDF Expression functionality

use pyo3::prelude::*;
use pyo3::pyclass;
use pyo3::types::PyType;
use fidget::context::Tree;
use fidget::var::Var;
use std::collections::HashMap;

use crate::fp_utils::{SdfVarOrFloat, get_tree_and_names_from_other, merge_maps};

/// Represents an SDF (Signed Distance Field) expression that can be evaluated and manipulated.
#[pyclass(name = "SDF")]
pub struct PyExpr {
    pub tree: Tree,
    pub var_names: HashMap<Var, String>,
}

#[pymethods]
impl PyExpr {
    #[staticmethod]
    /// Creates an SDF expression representing the X coordinate
    pub fn x() -> Self {
        let mut names = HashMap::new();
        names.insert(Var::X, "x".to_string());
        PyExpr { tree: Tree::x(), var_names: names }
    }

    #[staticmethod]
    /// Creates an SDF expression representing the Y coordinate
    pub fn y() -> Self {
        let mut names = HashMap::new();
        names.insert(Var::Y, "y".to_string());
        PyExpr { tree: Tree::y(), var_names: names }
    }

    #[staticmethod]
    /// Creates an SDF expression representing the Z coordinate
    pub fn z() -> Self {
        let mut names = HashMap::new();
        names.insert(Var::Z, "z".to_string());
        PyExpr { tree: Tree::z(), var_names: names }
    }

    #[staticmethod]
    /// Creates an SDF expression representing a constant value
    pub fn constant(val: f64) -> Self {
        PyExpr { tree: Tree::constant(val), var_names: HashMap::new() }
    }

    #[staticmethod]
    /// Creates an SDF expression from a variable
    pub fn from_var(var: &crate::fp_var::PyVar) -> Self {
        let mut names = HashMap::new();
        if let Some(name) = &var.name {
            names.insert(var.var.clone(), name.clone());
        }
        PyExpr { tree: Tree::from(var.var.clone()), var_names: names }
    }
    

    /// Returns the square root of this expression
    pub fn sqrt(&self) -> Self {
        PyExpr { tree: self.tree.sqrt(), var_names: self.var_names.clone() }
    }

    /// Returns the sine of this expression
    pub fn sin(&self) -> Self {
        PyExpr { tree: self.tree.sin(), var_names: self.var_names.clone() }
    }

    /// Returns the cosine of this expression
    pub fn cos(&self) -> Self {
        PyExpr { tree: self.tree.cos(), var_names: self.var_names.clone() }
    }

    /// Returns the absolute value of this expression
    pub fn abs(&self) -> Self {
        PyExpr { tree: self.tree.abs(), var_names: self.var_names.clone() }
    }

    /// Returns the maximum of this expression and another
    pub fn max(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.max(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the minimum of this expression and another
    pub fn min(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.min(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the square of this expression
    pub fn square(&self) -> Self {
        PyExpr { tree: self.tree.square(), var_names: self.var_names.clone() }
    }

    /// Returns the floor of this expression
    pub fn floor(&self) -> Self {
        PyExpr { tree: self.tree.floor(), var_names: self.var_names.clone() }
    }

    /// Returns the ceiling of this expression
    pub fn ceil(&self) -> Self {
        PyExpr { tree: self.tree.ceil(), var_names: self.var_names.clone() }
    }

    /// Returns the rounded value of this expression
    pub fn round(&self) -> Self {
        PyExpr { tree: self.tree.round(), var_names: self.var_names.clone() }
    }

    /// Returns a comparison between this expression and another
    pub fn compare(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.compare(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the modulo of this expression with another
    pub fn modulo(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.modulo(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the logical AND of this expression with another
    /// Fidget's native logical AND operation
    /// This preserves SDF-specific properties and optimizations
    #[pyo3(name = "and_")]
    pub fn and(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.and(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the logical OR of this expression with another
    /// Fidget's native logical OR operation
    /// This preserves SDF-specific properties and optimizations
    #[pyo3(name = "or_")]
    pub fn or(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.or(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Returns the atan2(y, x) of this expression with another
    pub fn atan2(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.atan2(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    pub fn add(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() + other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    pub fn sub(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() - other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    pub fn mul(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() * other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    pub fn div(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() / other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Returns the negation of this expression
    pub fn neg(&self) -> Self {
        PyExpr { tree: self.tree.neg(), var_names: self.var_names.clone() }
    }

    /// Returns the reciprocal (1/x) of this expression
    pub fn recip(&self) -> Self {
        PyExpr { tree: self.tree.recip(), var_names: self.var_names.clone() }
    }

    /// Returns the tangent of this expression
    pub fn tan(&self) -> Self {
        PyExpr { tree: self.tree.tan(), var_names: self.var_names.clone() }
    }

    /// Returns the arcsine of this expression
    pub fn asin(&self) -> Self {
        PyExpr { tree: self.tree.asin(), var_names: self.var_names.clone() }
    }

    /// Returns the arccosine of this expression
    pub fn acos(&self) -> Self {
        PyExpr { tree: self.tree.acos(), var_names: self.var_names.clone() }
    }

    /// Returns the arctangent of this expression
    pub fn atan(&self) -> Self {
        PyExpr { tree: self.tree.atan(), var_names: self.var_names.clone() }
    }

    /// Returns e raised to the power of this expression
    pub fn exp(&self) -> Self {
        PyExpr { tree: self.tree.exp(), var_names: self.var_names.clone() }
    }

    /// Returns the natural logarithm of this expression
    pub fn ln(&self) -> Self {
        PyExpr { tree: self.tree.ln(), var_names: self.var_names.clone() }
    }

    /// Returns the logical NOT of this expression
    #[pyo3(name = "not_")]
    pub fn not(&self) -> Self {
        PyExpr { tree: self.tree.not(), var_names: self.var_names.clone() }
    }
    
    /// Remaps the x, y, z coordinates of this expression
    pub fn remap_xyz(&self, x_expr: SdfVarOrFloat, y_expr: SdfVarOrFloat, z_expr: SdfVarOrFloat) -> Self {
        // Convert parameters to Tree and variable names
        let (x_tree, x_names) = get_tree_and_names_from_other(x_expr);
        let (y_tree, y_names) = get_tree_and_names_from_other(y_expr);
        let (z_tree, z_names) = get_tree_and_names_from_other(z_expr);
        
        let remapped_tree = self.tree.remap_xyz(x_tree, y_tree, z_tree);
        let mut final_names = self.var_names.clone();
        
        // Merge variable names from all expressions
        for (var, name) in &x_names {
            final_names.entry(var.clone()).or_insert_with(|| name.clone());
        }
        for (var, name) in &y_names {
            final_names.entry(var.clone()).or_insert_with(|| name.clone());
        }
        for (var, name) in &z_names {
            final_names.entry(var.clone()).or_insert_with(|| name.clone());
        }
        
        // Ensure x, y, z are in the variable names
        final_names.entry(Var::X).or_insert_with(|| "x".to_string());
        final_names.entry(Var::Y).or_insert_with(|| "y".to_string());
        final_names.entry(Var::Z).or_insert_with(|| "z".to_string());
        
        PyExpr { tree: remapped_tree, var_names: final_names }
    }
    
    /// Applies an affine transformation to this expression
    #[pyo3(signature = (matrix))]
    pub fn remap_affine(&self, matrix: Vec<f64>) -> PyResult<Self> {
        if matrix.len() != 12 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Affine matrix must be a flat array of 12 elements (3x4 matrix)"
            ));
        }
        
        // Create a transformation matrix from the provided values
        let mut transform = nalgebra::Matrix4::<f64>::identity();
        
        // Set the rotation/scaling part (3x3)
        transform[(0, 0)] = matrix[0];
        transform[(0, 1)] = matrix[1];
        transform[(0, 2)] = matrix[2];
        transform[(1, 0)] = matrix[3];
        transform[(1, 1)] = matrix[4];
        transform[(1, 2)] = matrix[5];
        transform[(2, 0)] = matrix[6];
        transform[(2, 1)] = matrix[7];
        transform[(2, 2)] = matrix[8];
        
        // Set the translation part
        transform[(0, 3)] = matrix[9];
        transform[(1, 3)] = matrix[10];
        transform[(2, 3)] = matrix[11];
        
        // Convert to Affine3
        let mat = nalgebra::Affine3::from_matrix_unchecked(transform);
        
        let remapped_tree = self.tree.remap_affine(mat);
        let mut final_names = self.var_names.clone();
        final_names.entry(Var::X).or_insert_with(|| "x".to_string());
        final_names.entry(Var::Y).or_insert_with(|| "y".to_string());
        final_names.entry(Var::Z).or_insert_with(|| "z".to_string());
        
        Ok(PyExpr { tree: remapped_tree, var_names: final_names })
    }

    // --- Python operator overloads ---

    /// Implements addition for Python
    fn __add__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() + other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side addition for Python
    fn __radd__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree + self.tree.clone(), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements subtraction for Python
    fn __sub__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() - other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side subtraction for Python
    fn __rsub__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree - self.tree.clone(), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements multiplication for Python
    fn __mul__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() * other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side multiplication for Python
    fn __rmul__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree * self.tree.clone(), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements division for Python
    fn __truediv__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone() / other_tree, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side division for Python
    fn __rtruediv__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree / self.tree.clone(), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements negation for Python
    fn __neg__(&self) -> Self {
        self.neg()
    }
    
    /// Implements inversion (bitwise NOT) for Python (logical NOT in this context)
    fn __invert__(&self) -> Self {
        self.not()
    }

    /// Implements power operator for Python
    fn __pow__(&self, other: SdfVarOrFloat, _modulo: Option<i64>) -> Self {
        // Special case for integer exponents to use the efficient algorithm
        if let SdfVarOrFloat::Float(n) = other {
            if n.fract() == 0.0 && n.abs() < (i64::MAX as f64) {
                return PyExpr { 
                    tree: self.tree.pow(n as i64), 
                    var_names: self.var_names.clone() 
                };
            }
        }
        
        // General case using e^(y * ln(x)) for all other cases
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let ln_self = self.tree.ln();
        let product = ln_self * other_tree;
        let result = product.exp();
        
        PyExpr { 
            tree: result, 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }


    /// Implements modulo operator for Python
    fn __mod__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: self.tree.clone().modulo(other_tree), 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side modulo operator for Python
    fn __rmod__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree.modulo(self.tree.clone()), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements bitwise AND operator for Python (logical AND in this context)
    fn __and__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        // Use Fidget's native AND behavior for operator overloading
        // This is more appropriate for SDF operations
        PyExpr {
            tree: self.tree.clone().and(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Python-style logical AND (using min)
    /// Returns 1.0 if both operands are non-zero, 0.0 otherwise
    pub fn python_and(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.clone().min(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Implements right-side bitwise AND operator for Python
    fn __rand__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree.and(self.tree.clone()), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements bitwise OR operator for Python (logical OR in this context)
    fn __or__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        // Use Fidget's native OR behavior for operator overloading
        // This is more appropriate for SDF operations
        PyExpr {
            tree: self.tree.clone().or(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Python-style logical OR (using max)
    /// Returns 1.0 if either operand is non-zero, 0.0 otherwise
    pub fn python_or(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr {
            tree: self.tree.clone().max(other_tree),
            var_names: merge_maps(&self.var_names, &other_names)
        }
    }

    /// Implements right-side bitwise OR operator for Python
    fn __ror__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        PyExpr { 
            tree: other_tree.or(self.tree.clone()), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }


    /// Implements floor division operator for Python
    fn __floordiv__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let div_result = self.tree.clone() / other_tree;
        PyExpr { 
            tree: div_result.floor(), 
            var_names: merge_maps(&self.var_names, &other_names) 
        }
    }

    /// Implements right-side floor division operator for Python
    fn __rfloordiv__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let div_result = other_tree / self.tree.clone();
        PyExpr { 
            tree: div_result.floor(), 
            var_names: merge_maps(&other_names, &self.var_names) 
        }
    }

    /// Implements equality operator (==) for Python
    /// Returns 1.0 if expressions are equal, 0.0 otherwise
    fn __eq__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // Compare returns 0.0 when equal, so: 1.0 - abs(compare) will be:
        // - 1.0 if compare == 0 (equal)
        // - 0.0 if compare == 1 or -1 (not equal)
        let one = Tree::constant(1.0);
        let result = one - cmp_result.abs();
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    /// Implements inequality operator (!=) for Python
    /// Returns 1.0 if expressions are not equal, 0.0 otherwise
    fn __ne__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // It's the opposite of equality, so abs(compare) will be:
        // - 0.0 if equal (which we don't want)
        // - 1.0 if not equal (which we want)
        // So abs(compare) gives us exactly what we need
        let result = cmp_result.abs();
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    /// Implements less than operator (<) for Python
    /// Returns 1.0 if self < other, 0.0 otherwise
    fn __lt__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // compare() returns -1.0 when self < other
        // We want 1.0 if result is -1, and 0.0 otherwise
        let zero = Tree::constant(0.0);
        
        // max(0, -compare) returns 1.0 if compare == -1, otherwise 0.0
        let result = zero.max(cmp_result.neg());
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    /// Implements less than or equal operator (<=) for Python
    /// Returns 1.0 if self <= other, 0.0 otherwise
    fn __le__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // compare() returns:
        // -1.0 when self < other  (want 1.0)
        //  0.0 when self == other (want 1.0)
        //  1.0 when self > other  (want 0.0)
        
        // 1.0 if compare <= 0, 0.0 if compare > 0
        let zero = Tree::constant(0.0);
        let one = Tree::constant(1.0);
        
        // This is a simple approach that returns 1.0 for <= and 0.0 for >
        let result = one - zero.max(cmp_result);
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    /// Implements greater than operator (>) for Python
    /// Returns 1.0 if self > other, 0.0 otherwise
    fn __gt__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // compare() returns 1.0 when self > other
        // We want 1.0 if result is 1, and 0.0 otherwise
        let zero = Tree::constant(0.0);
        
        // max(0, compare) returns 1.0 if compare == 1, otherwise 0.0
        let result = zero.max(cmp_result);
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    /// Implements greater than or equal operator (>=) for Python
    /// Returns 1.0 if self >= other, 0.0 otherwise
    fn __ge__(&self, other: SdfVarOrFloat) -> Self {
        let (other_tree, other_names) = get_tree_and_names_from_other(other);
        let cmp_result = self.tree.compare(other_tree);
        
        // compare() returns:
        // -1.0 when self < other  (want 0.0)
        //  0.0 when self == other (want 1.0)
        //  1.0 when self > other  (want 1.0)
        
        // 1.0 if compare >= 0, 0.0 if compare < 0
        let zero = Tree::constant(0.0);
        let one = Tree::constant(1.0);
        
        // This is a simple approach that returns 1.0 for >= and 0.0 for <
        let result = one - zero.max(cmp_result.neg());
        
        PyExpr {
            tree: result,
            var_names: merge_maps(&self.var_names, &other_names),
        }
    }

    fn __repr__(&self) -> String {
        self.f_rep()
    }

    fn __str__(&self) -> String {
        self.f_rep()
    }

    /// Generates an optimized f-rep string representation by first converting to VM format,
    /// applying VM simplifications, and then converting back to f-rep.
    /// This produces a more compact and efficient f-rep expression.
    pub fn f_rep(&self) -> String {
        crate::fp_utils::tree_to_vm_to_frep(&self.tree, &self.var_names)
    }
    
    /// Allow automatic conversion from PyVar in Python
    #[classmethod]
    pub fn __class_getitem__(_cls: &Bound<'_, PyType>, item: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(var) = item.extract::<PyRef<crate::fp_var::PyVar>>() {
            Ok(Self::from_var(&var))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Cannot convert to SDF expression"
            ))
        }
    }
}

/// Implement automatic conversion from PyVar to PyExpr
impl From<&crate::fp_var::PyVar> for PyExpr {
    fn from(var: &crate::fp_var::PyVar) -> Self {
        Self::from_var(var)
    }
}