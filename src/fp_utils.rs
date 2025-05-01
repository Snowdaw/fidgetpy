// src/fp_utils.rs
// Utility functions for SDF expressions

use pyo3::prelude::*;
use pyo3::types::PyList; // Added
use pyo3::exceptions::PyTypeError; // Added
use fidget::context::{Tree, TreeOp, BinaryOpcode, UnaryOpcode};
use fidget::var::Var;
use std::collections::HashMap;

use crate::fp_var::PyVar;
use crate::fp_expr::PyExpr;

/// Enum to accept SDF, Var, or float in operators
#[derive(FromPyObject)]
pub enum SdfVarOrFloat<'a> {
    Sdf(PyRef<'a, PyExpr>),
    Var(PyRef<'a, PyVar>),
    Float(f64),
}

// Implement From<PyVar> for SdfVarOrFloat to allow automatic conversion
impl<'a> From<PyRef<'a, PyVar>> for SdfVarOrFloat<'a> {
    fn from(var: PyRef<'a, PyVar>) -> Self {
        SdfVarOrFloat::Var(var)
    }
}

/// Helper to get Tree and names map from SdfVarOrFloat
pub fn get_tree_and_names_from_other(other: SdfVarOrFloat) -> (Tree, HashMap<Var, String>) {
    match other {
        SdfVarOrFloat::Sdf(sdf_ref) => (sdf_ref.tree.clone(), sdf_ref.var_names.clone()),
        SdfVarOrFloat::Var(var_ref) => {
            let tree = Tree::from(var_ref.var.clone());
            let mut names = HashMap::new();
            if let Some(name) = &var_ref.name {
                names.insert(var_ref.var.clone(), name.clone());
            }
            (tree, names)
        }
        SdfVarOrFloat::Float(val) => (Tree::constant(val), HashMap::new()),
    }
}

/// Helper function to merge two HashMaps.
pub fn merge_maps(map1: &HashMap<Var, String>, map2: &HashMap<Var, String>) -> HashMap<Var, String> {
    let mut merged = map1.clone();
    for (key, value) in map2 {
        merged.insert(key.clone(), value.clone());
    }
    merged
}

/// Check if an expression contains custom (non x,y,z) variables
pub fn check_for_custom_vars(op: &TreeOp) -> bool {
    match op {
        TreeOp::Input(var) => matches!(var, Var::V(_)),
        TreeOp::Const(_) => false,
        TreeOp::Unary(_, arg) => check_for_custom_vars(&*arg),
        TreeOp::Binary(_, lhs, rhs) => check_for_custom_vars(&*lhs) || check_for_custom_vars(&*rhs),
        TreeOp::RemapAxes { target, x, y, z } => {
            check_for_custom_vars(&*target) || 
            check_for_custom_vars(&*x) || 
            check_for_custom_vars(&*y) || 
            check_for_custom_vars(&*z)
        }
        TreeOp::RemapAffine { target, .. } => check_for_custom_vars(&*target),
    }
}

/// Format a tree operation into a human-readable F-Rep string
pub fn format_frep(op: &TreeOp, names: &HashMap<Var, String>) -> String {
    match op {
        TreeOp::Input(var) => {
            if let Some(name) = names.get(var) {
                name.clone()
            } else {
                format!("var({:?})", var)
            }
        }
        TreeOp::Const(val) => format!("{:.3}", val),
        TreeOp::Unary(opcode, arg) => {
            let arg_str = format_frep(&*arg, names);
            let op_name = match opcode {
                UnaryOpcode::Square => "square", 
                UnaryOpcode::Floor => "floor", 
                UnaryOpcode::Ceil => "ceil",
                UnaryOpcode::Round => "round", 
                UnaryOpcode::Sqrt => "sqrt", 
                UnaryOpcode::Neg => "neg",
                UnaryOpcode::Sin => "sin", 
                UnaryOpcode::Cos => "cos", 
                UnaryOpcode::Tan => "tan",
                UnaryOpcode::Asin => "asin", 
                UnaryOpcode::Acos => "acos", 
                UnaryOpcode::Atan => "atan",
                UnaryOpcode::Exp => "exp", 
                UnaryOpcode::Ln => "ln", 
                UnaryOpcode::Not => "not",
                UnaryOpcode::Abs => "abs", 
                UnaryOpcode::Recip => "recip",
            };
            format!("{}({})", op_name, arg_str)
        }
        TreeOp::Binary(opcode, lhs, rhs) => {
            let lhs_str = format_frep(&*lhs, names);
            let rhs_str = format_frep(&*rhs, names);
            let op_name = match opcode {
                BinaryOpcode::Add => "add", 
                BinaryOpcode::Sub => "sub", 
                BinaryOpcode::Mul => "mul",
                BinaryOpcode::Div => "div", 
                BinaryOpcode::Max => "max", 
                BinaryOpcode::Min => "min",
                BinaryOpcode::Compare => "compare", 
                BinaryOpcode::Mod => "mod", 
                BinaryOpcode::And => "and",
                BinaryOpcode::Or => "or", 
                BinaryOpcode::Atan => "atan2",
            };
            format!("{}({}, {})", op_name, lhs_str, rhs_str)
        }
        TreeOp::RemapAxes { target, x, y, z } => {
            let target_str = format_frep(&*target, names);
            let x_str = format_frep(&*x, names);
            let y_str = format_frep(&*y, names);
            let z_str = format_frep(&*z, names);
            format!("remap_xyz({}, x={}, y={}, z={})", target_str, x_str, y_str, z_str)
        }
        TreeOp::RemapAffine { target, .. } => {
            let target_str = format_frep(&*target, names);
            format!("remap_affine({}, mat=[...])", target_str)
        }
    }
}

/// Convert a VM representation to F-Rep format
/// This is a simplified approach that converts registers into variable names
/// and builds a more efficient F-Rep expression
pub fn parse_vm_to_frep(vm_str: &str, var_names: &HashMap<Var, String>) -> String {
    // First pass: collect all register definitions
    let mut register_defs: Vec<(String, String, Vec<String>)> = Vec::new();
    
    // Process each line in the VM format
    for line in vm_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        
        // Parse register assignment
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 || !parts[0].starts_with('_') {
            continue;
        }
        
        let register_name = parts[0].to_string();
        let operation = parts[1].to_string();
        
        // Collect operands
        let operands = parts[2..].iter().map(|s| s.to_string()).collect();
        
        register_defs.push((register_name, operation, operands));
    }
    
    // Second pass: build expressions by substituting registers
    let mut register_map: HashMap<String, String> = HashMap::new();
    
    // First, process all variable definitions to ensure they're available
    for (reg_name, op, _) in register_defs.iter().filter(|(_, op, _)| op.starts_with("var-")) {
        match op.as_str() {
            "var-x" => {
                let var_name = var_names.get(&Var::X).cloned().unwrap_or_else(|| "x".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            "var-y" => {
                let var_name = var_names.get(&Var::Y).cloned().unwrap_or_else(|| "y".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            "var-z" => {
                let var_name = var_names.get(&Var::Z).cloned().unwrap_or_else(|| "z".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            // Handle custom variable names in the format var-name
            op if op.starts_with("var-") => {
                // Extract the variable name from the var-name format
                let var_name = op.strip_prefix("var-").unwrap_or("custom").to_string();
                if var_name != "custom" {
                    // Use the extracted name directly
                    register_map.insert(reg_name.clone(), var_name);
                } else {
                    // Fallback for var-custom
                    register_map.insert(reg_name.clone(), "custom_var".to_string());
                }
            },
            _ => {}
        }
    }
    
    // Process registers in order (they should already be in dependency order)
    for (reg_name, op, operands) in register_defs.iter() {
        // Skip variables as they've already been processed
        if op.starts_with("var-") {
            continue;
        }
        match op.as_str() {
            "const" => {
                if !operands.is_empty() {
                    register_map.insert(reg_name.clone(), operands[0].clone());
                }
            },
            "var-x" => {
                let var_name = var_names.get(&Var::X).cloned().unwrap_or_else(|| "x".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            "var-y" => {
                let var_name = var_names.get(&Var::Y).cloned().unwrap_or_else(|| "y".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            "var-z" => {
                let var_name = var_names.get(&Var::Z).cloned().unwrap_or_else(|| "z".to_string());
                register_map.insert(reg_name.clone(), var_name);
            },
            // Handle custom variable names in the format var-name
            op if op.starts_with("var-") => {
                // Extract the variable name from the var-name format
                let var_name = op.strip_prefix("var-").unwrap_or("custom").to_string();
                if var_name != "custom" {
                    // Use the extracted name directly
                    register_map.insert(reg_name.clone(), var_name);
                } else {
                    // Fallback for var-custom
                    register_map.insert(reg_name.clone(), "custom_var".to_string());
                }
            },
            "add" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("add({}, {})", arg1, arg2));
                }
            },
            "mul" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("mul({}, {})", arg1, arg2));
                }
            },
            "div" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("div({}, {})", arg1, arg2));
                }
            },
            "sub" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("sub({}, {})", arg1, arg2));
                }
            },
            "sqrt" => {
                if operands.len() >= 1 {
                    let arg = get_operand_value(&operands[0], &register_map);
                    register_map.insert(reg_name.clone(), format!("sqrt({})", arg));
                }
            },
            "neg" => {
                if operands.len() >= 1 {
                    let arg = get_operand_value(&operands[0], &register_map);
                    register_map.insert(reg_name.clone(), format!("neg({})", arg));
                }
            },
            "min" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("min({}, {})", arg1, arg2));
                }
            },
            "max" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("max({}, {})", arg1, arg2));
                }
            },
            "and" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("and({}, {})", arg1, arg2));
                }
            },
            "or" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("or({}, {})", arg1, arg2));
                }
            },
            "not" => {
                if operands.len() >= 1 {
                    let arg = get_operand_value(&operands[0], &register_map);
                    register_map.insert(reg_name.clone(), format!("not({})", arg));
                }
            },
            "compare" => {
                if operands.len() >= 2 {
                    let arg1 = get_operand_value(&operands[0], &register_map);
                    let arg2 = get_operand_value(&operands[1], &register_map);
                    register_map.insert(reg_name.clone(), format!("compare({}, {})", arg1, arg2));
                }
            },
            _ => {
                // For other operations we don't recognize, just keep the raw form
                let operation = format!("{}({})", op, operands.join(", "));
                register_map.insert(reg_name.clone(), operation);
            }
        }
    }
    
    // Return the final register's expression or a default value
    if let Some((last_reg, _, _)) = register_defs.last() {
        register_map.get(last_reg)
            .cloned()
            .unwrap_or_else(|| "0.0".to_string())
    } else {
        "0.0".to_string()
    }
}

/// Helper function to get the value of an operand, either from the register map
/// or as a literal if it's not a register reference
fn get_operand_value(operand: &str, register_map: &HashMap<String, String>) -> String {
    if operand.starts_with('_') {
        // It's a register reference
        register_map.get(operand).cloned().unwrap_or_else(|| operand.to_string())
    } else {
        // It's a literal value
        operand.to_string()
    }
}

// Unused function removed

/// Convert a tree to VM format string, parse it, and then convert to F-Rep
pub fn tree_to_vm_to_frep(tree: &Tree, var_names: &HashMap<Var, String>) -> String {
    // Create a PyExpr to use the existing to_vm functionality
    let expr = PyExpr {
        tree: tree.clone(),
        var_names: var_names.clone(),
    };
    
    // Use the existing to_vm functionality from fp_context
    let mut ctx = crate::fp_context::PySDFContext::new();
    let vm_str = match ctx.to_vm(&expr) {
        Ok(s) => s,
        Err(_) => return format_frep(tree, var_names), // Fallback to direct format if VM export fails
    };
    
    // Parse the VM string to create F-Rep
    let mut result = parse_vm_to_frep(&vm_str, var_names);
    
    // Fix any remaining register references in the result
    // This handles cases where the result still contains _5, _7, etc.
    if result.contains('_') {
        // Create a map of variable registers to their names from the VM string
        let mut var_register_map = HashMap::new();
        
        for line in vm_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0].starts_with('_') {
                if parts[1] == "var-x" {
                    var_register_map.insert(parts[0], "x");
                } else if parts[1] == "var-y" {
                    var_register_map.insert(parts[0], "y");
                } else if parts[1] == "var-z" {
                    var_register_map.insert(parts[0], "z");
                } else if parts[1].starts_with("var-") {
                    if let Some(var_name) = parts[1].strip_prefix("var-") {
                        var_register_map.insert(parts[0], var_name);
                    }
                }
            }
        }
        
        // Replace register references in the result
        for (reg, var) in var_register_map {
            result = result.replace(reg, var);
        }
    }
    
    result
}


/// Parses the optional Python list of variable expressions into a Vec<PyVar>.
/// Handles both direct PyVar objects and PyExpr objects representing variables.
pub fn parse_variable_list(variables: Option<&Bound<'_, PyList>>) -> PyResult<Vec<PyVar>> {
    match variables {
        Some(py_list) => {
            let mut vars = Vec::with_capacity(py_list.len());
            for item_result in py_list.iter() {
                // First try to extract PyVar directly
                if let Ok(var) = item_result.extract::<PyRef<PyVar>>() {
                    vars.push(var.clone());
                    continue;
                }
                
                // If not a PyVar, try to extract PyExpr
                if let Ok(expr) = item_result.extract::<PyRef<PyExpr>>() {
                    match &*expr.tree {
                        TreeOp::Input(var) => {
                            // Extract name from the expression's map if available
                            let name = expr.var_names.get(var).cloned();
                            vars.push(PyVar { var: var.clone(), name });
                        }
                        _ => {
                            return Err(PyTypeError::new_err(
                                "Items in the 'variables' list must be either variables (PyVar) or direct variable expressions (e.g., created by fp.var('name'))",
                            ));
                        }
                    }
                } else {
                    return Err(PyTypeError::new_err(
                        "Items in the 'variables' list must be either variables (PyVar) or expressions (PyExpr)",
                    ));
                }
            }
            Ok(vars)
        }
        None => Ok(vec![PyVar::x(), PyVar::y(), PyVar::z()]), // Default to x, y, z
    }
}