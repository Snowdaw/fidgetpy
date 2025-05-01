use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::pyclass;
use std::collections::{HashMap, HashSet};
use numpy::{PyArray1, PyReadonlyArray2};
use fidget::context::{Context, Tree};
use fidget::var::Var;
use std::io::Write;

use crate::fp_var::PyVar;
use crate::fp_expr::PyExpr;
use crate::fp_eval::{determine_backend, _evaluate_bulk_vm, _evaluate_bulk_jit};

#[pyclass(name = "SDFContext")]
pub struct PySDFContext {
    ctx: Context,
    nodes: HashMap<usize, fidget::context::Node>,
    trees: HashMap<usize, Tree>,
    node_vars: HashMap<usize, Var>,
    next_handle: usize,
}

#[pymethods]
impl PySDFContext {
    #[new]
    pub fn new() -> Self {
        PySDFContext {
            ctx: Context::new(),
            nodes: HashMap::new(),
            trees: HashMap::new(),
            node_vars: HashMap::new(),
            next_handle: 0,
        }
    }

    pub fn clear(&mut self) {
        self.ctx = Context::new();
        self.nodes.clear();
        self.trees.clear();
        self.node_vars.clear();
        self.next_handle = 0;
    }

    /// Import an SDFExpr into the context, returning a handle (int)
    pub fn import_expr(&mut self, expr: &PyExpr) -> usize {
        let node = self.ctx.import(&expr.tree);
        let handle = self.next_handle;
        self.nodes.insert(handle, node);
        self.trees.insert(handle, expr.tree.clone());
        self.next_handle += 1;
        handle
    }

    /// Create a variable node from a Var
    pub fn var(&mut self, var: &PyVar) -> usize {
        let node = self.ctx.var(var.var.clone());
        let handle = self.next_handle;
        self.nodes.insert(handle, node);
        self.node_vars.insert(handle, var.var.clone());
        self.next_handle += 1;
        handle
    }

    /// Get the Var for a given node handle (if it is a variable node)
    pub fn get_var(&self, handle: usize) -> PyResult<PyVar> {
        if let Some(var) = self.node_vars.get(&handle) {
            Ok(PyVar { var: var.clone(), name: None })
        } else {
            Err(PyValueError::new_err("Handle does not correspond to a variable node"))
        }
    }

    /// Import a VM format string and create a new SDFExpr.
    ///
    /// Args:
    ///     text: String in VM format
    ///
    /// Returns:
    ///     An SDF expression parsed from the VM format
    #[pyo3(text_signature = "(text)")]
    pub fn from_vm(&self, text: String) -> PyResult<PyExpr> {
        // Create a new context for parsing
        let mut ctx = Context::new();
        
        // Extract variable definitions and build operation map
        let mut var_names: HashMap<Var, String> = HashMap::new();
        let mut node_map: HashMap<String, fidget::context::Node> = HashMap::new();
        
        // First pass: process variable definitions
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            let register_name = parts[0];
            let op_code = parts[1];
            
            // Handle variable definitions
            if op_code.starts_with("var-") {
                let var: Var;
                let var_name: String;
                
                match op_code {
                    "var-x" => {
                        var = Var::X;
                        var_name = "x".to_string();
                    },
                    "var-y" => {
                        var = Var::Y;
                        var_name = "y".to_string();
                    },
                    "var-z" => {
                        var = Var::Z;
                        var_name = "z".to_string();
                    },
                    _ => {
                        // Handle custom variables (like var-w, var-radius, etc.)
                        if let Some(name) = op_code.strip_prefix("var-") {
                            var = Var::new();
                            var_name = name.to_string();
                        } else {
                            return Err(PyValueError::new_err(format!("Invalid variable definition: {}", op_code)));
                        }
                    }
                };
                
                // Create variable node and add to maps
                let node = ctx.var(var.clone());
                node_map.insert(register_name.to_string(), node);
                var_names.insert(var, var_name);
            }
        }
        
        // Second pass: process operations
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }
            
            let register_name = parts[0];
            let op_code = parts[1];
            
            // Skip variable definitions (already processed)
            if op_code.starts_with("var-") {
                continue;
            }
            
            // Process constants
            if op_code == "const" && parts.len() >= 3 {
                if let Ok(value) = parts[2].parse::<f64>() {
                    let node = ctx.constant(value);
                    node_map.insert(register_name.to_string(), node);
                } else {
                    return Err(PyValueError::new_err(format!("Invalid constant value: {}", parts[2])));
                }
                continue;
            }
            
            // Process unary operations
            if parts.len() >= 3 {
                let arg_name = parts[2];
                let arg_node = match node_map.get(arg_name) {
                    Some(node) => *node,
                    None => return Err(PyValueError::new_err(format!("Unknown register: {}", arg_name))),
                };
                
                let node = match op_code {
                    "neg" => ctx.neg(arg_node),
                    "abs" => ctx.abs(arg_node),
                    "recip" => ctx.recip(arg_node),
                    "sqrt" => ctx.sqrt(arg_node),
                    "square" => ctx.square(arg_node),
                    "floor" => ctx.floor(arg_node),
                    "ceil" => ctx.ceil(arg_node),
                    "round" => ctx.round(arg_node),
                    "sin" => ctx.sin(arg_node),
                    "cos" => ctx.cos(arg_node),
                    "tan" => ctx.tan(arg_node),
                    "asin" => ctx.asin(arg_node),
                    "acos" => ctx.acos(arg_node),
                    "atan" => ctx.atan(arg_node),
                    "exp" => ctx.exp(arg_node),
                    "ln" => ctx.ln(arg_node),
                    "not" => ctx.not(arg_node),
                    
                    // Binary operations (need another operand)
                    _ => {
                        if parts.len() >= 4 {
                            let arg2_name = parts[3];
                            let arg2_node = match node_map.get(arg2_name) {
                                Some(node) => *node,
                                None => return Err(PyValueError::new_err(format!("Unknown register: {}", arg2_name))),
                            };
                            
                            match op_code {
                                "add" => ctx.add(arg_node, arg2_node),
                                "sub" => ctx.sub(arg_node, arg2_node),
                                "mul" => ctx.mul(arg_node, arg2_node),
                                "div" => ctx.div(arg_node, arg2_node),
                                "min" => ctx.min(arg_node, arg2_node),
                                "max" => ctx.max(arg_node, arg2_node),
                                "compare" => ctx.compare(arg_node, arg2_node),
                                "mod" => ctx.modulo(arg_node, arg2_node),
                                "and" => ctx.and(arg_node, arg2_node),
                                "or" => ctx.or(arg_node, arg2_node),
                                "atan2" => ctx.atan2(arg_node, arg2_node),
                                _ => return Err(PyValueError::new_err(format!("Unknown operation: {}", op_code))),
                            }
                        } else {
                            return Err(PyValueError::new_err(format!("Binary operation {} requires two operands", op_code)));
                        }
                    }
                };
                
                match node {
                    Ok(n) => {
                        node_map.insert(register_name.to_string(), n);
                    },
                    Err(e) => return Err(PyValueError::new_err(format!("Operation error: {}", e))),
                }
            }
        }
        
        // Get the final node (should be the last one defined)
        let last_register = match text.lines().filter(|l| !l.trim().is_empty() && !l.trim().starts_with('#'))
                                .last()
                                .and_then(|line| line.split_whitespace().next()) {
            Some(reg) => reg.to_string(),
            None => return Err(PyValueError::new_err("No operations found in VM format")),
        };
        
        let final_node = match node_map.get(&last_register) {
            Some(node) => *node,
            None => return Err(PyValueError::new_err(format!("Final register {} not found", last_register))),
        };
        
        // Export the node to a tree
        let tree = ctx.export(final_node).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Create a PyExpr from the tree with our collected variable names
        Ok(PyExpr { tree, var_names })
    }

    /// Convert an SDFExpr to VM format.
    ///
    /// Args:
    ///     expr: The SDF expression to convert
    ///
    /// Returns:
    ///     String representation in VM format
    #[pyo3(text_signature = "(expr)")]
    pub fn to_vm(&mut self, expr: &PyExpr) -> PyResult<String> {
        // Import the tree into our context
        let node = self.ctx.import(&expr.tree);
        
        // Create a string buffer for the VM format
        let mut output = Vec::new();
        
        // Write the header
        writeln!(output, "# Fidget VM format export").unwrap();
        writeln!(output, "# Generated by fidgetpy").unwrap();
        
        // We need to reconstruct the VM representation by walking through the tree
        // and generating the appropriate lines
        let mut node_map: HashMap<*const fidget::context::Op, String> = HashMap::new();
        let mut next_id: u32 = 0;
        
        // Helper function to write a node and its dependencies
        fn write_node_vm(
            node: fidget::context::Node,
            ctx: &Context,
            output: &mut Vec<u8>,
            node_map: &mut HashMap<*const fidget::context::Op, String>,
            next_id: &mut u32,
            written: &mut HashSet<*const fidget::context::Op>,
            var_names: &HashMap<Var, String>
        ) -> PyResult<String> {
            let op = ctx.get_op(node).unwrap();
            let ptr = op as *const _;
            
            // If we've already written this node, just return its ID
            if let Some(id) = node_map.get(&ptr) {
                return Ok(id.clone());
            }
            
            // Generate a new ID for this node
            let id = format!("_{:x}", *next_id);
            *next_id += 1;
            node_map.insert(ptr, id.clone());
            
            // Write dependencies first
            match op {
                fidget::context::Op::Binary(_, a, b) => {
                    let a_id = write_node_vm(*a, ctx, output, node_map, next_id, written, var_names)?;
                    let b_id = write_node_vm(*b, ctx, output, node_map, next_id, written, var_names)?;
                    
                    // Only write the node if it hasn't been written yet
                    if !written.contains(&ptr) {
                        let op_name = match op {
                            fidget::context::Op::Binary(opcode, _, _) => {
                                match opcode {
                                    fidget::context::BinaryOpcode::Add => "add",
                                    fidget::context::BinaryOpcode::Sub => "sub",
                                    fidget::context::BinaryOpcode::Mul => "mul",
                                    fidget::context::BinaryOpcode::Div => "div",
                                    fidget::context::BinaryOpcode::Min => "min",
                                    fidget::context::BinaryOpcode::Max => "max",
                                    fidget::context::BinaryOpcode::Compare => "compare",
                                    fidget::context::BinaryOpcode::Mod => "mod",
                                    fidget::context::BinaryOpcode::And => "and",
                                    fidget::context::BinaryOpcode::Or => "or",
                                    fidget::context::BinaryOpcode::Atan => "atan2",
                                }
                            },
                            _ => unreachable!(),
                        };
                        
                        writeln!(output, "{} {} {} {}", id, op_name, a_id, b_id).unwrap();
                        written.insert(ptr);
                    }
                },
                fidget::context::Op::Unary(_, a) => {
                    let a_id = write_node_vm(*a, ctx, output, node_map, next_id, written, var_names)?;
                    
                    // Only write the node if it hasn't been written yet
                    if !written.contains(&ptr) {
                        let op_name = match op {
                            fidget::context::Op::Unary(opcode, _) => {
                                match opcode {
                                    fidget::context::UnaryOpcode::Neg => "neg",
                                    fidget::context::UnaryOpcode::Abs => "abs",
                                    fidget::context::UnaryOpcode::Recip => "recip",
                                    fidget::context::UnaryOpcode::Sqrt => "sqrt",
                                    fidget::context::UnaryOpcode::Square => "square",
                                    fidget::context::UnaryOpcode::Floor => "floor",
                                    fidget::context::UnaryOpcode::Ceil => "ceil",
                                    fidget::context::UnaryOpcode::Round => "round",
                                    fidget::context::UnaryOpcode::Sin => "sin",
                                    fidget::context::UnaryOpcode::Cos => "cos",
                                    fidget::context::UnaryOpcode::Tan => "tan",
                                    fidget::context::UnaryOpcode::Asin => "asin",
                                    fidget::context::UnaryOpcode::Acos => "acos",
                                    fidget::context::UnaryOpcode::Atan => "atan",
                                    fidget::context::UnaryOpcode::Exp => "exp",
                                    fidget::context::UnaryOpcode::Ln => "ln",
                                    fidget::context::UnaryOpcode::Not => "not",
                                }
                            },
                            _ => unreachable!(),
                        };
                        
                        writeln!(output, "{} {} {}", id, op_name, a_id).unwrap();
                        written.insert(ptr);
                    }
                },
                fidget::context::Op::Const(c) => {
                    // Only write the node if it hasn't been written yet
                    if !written.contains(&ptr) {
                        writeln!(output, "{} const {}", id, c.0).unwrap();
                        written.insert(ptr);
                    }
                },
                fidget::context::Op::Input(v) => {
                    // Only write the node if it hasn't been written yet
                    if !written.contains(&ptr) {
                        match v {
                            Var::X => writeln!(output, "{} var-x", id).unwrap(),
                            Var::Y => writeln!(output, "{} var-y", id).unwrap(),
                            Var::Z => writeln!(output, "{} var-z", id).unwrap(),
                            _ => {
                                // Check if we have a name for this variable
                                if let Some(name) = var_names.get(v) {
                                    // Use the variable name in the VM format
                                    writeln!(output, "{} var-{}", id, name).unwrap();
                                } else {
                                    // Fallback to var-custom if no name is available
                                    writeln!(output, "{} var-custom", id).unwrap();
                                }
                            },
                        }
                        written.insert(ptr);
                    }
                },
            }
            
            Ok(id)
        }
        
        // Write the tree
        let mut written = HashSet::new();
        write_node_vm(node, &self.ctx, &mut output, &mut node_map, &mut next_id, &mut written, &expr.var_names)?;
        
        // Convert the output to a string
        match String::from_utf8(output) {
            Ok(s) => Ok(s),
            Err(_) => Err(PyValueError::new_err("Failed to convert output to UTF-8 string")),
        }
    }

    /// Convert an SDFExpr to F-Rep format.
    ///
    /// Args:
    ///     expr: The SDF expression to convert
    ///
    /// Returns:
    ///     String representation in F-Rep format
    #[pyo3(text_signature = "(expr)")]
    pub fn to_frep(&self, expr: &PyExpr) -> PyResult<String> {
        // Fortunately, PyExpr already has a method to generate F-Rep strings
        Ok(expr.f_rep())
    }

    /// Import a F-Rep format string and create a new SDFExpr.
    ///
    /// Args:
    ///     text: String in F-Rep format
    ///
    /// Returns:
    ///     An SDF expression parsed from the F-Rep format
    pub fn from_frep(&self, text: String) -> PyResult<PyExpr> {
        // To properly parse the F-Rep, we'll:
        // 1. Create a VM representation of the F-Rep string
        // 2. Use our working VM parser to build the expression
        
        // First, tokenize the F-Rep string
        let processed_text = text
            .replace("(", " ( ")
            .replace(")", " ) ")
            .replace(",", " , ");
        
        let tokens: Vec<&str> = processed_text.split_whitespace().collect();
        
        // Initialize VM format text
        let mut vm_text = String::new();
        vm_text.push_str("# Fidget VM format export\n");
        vm_text.push_str("# Generated from F-Rep\n");
        
        // Register counter and maps
        let mut next_reg = 0;
        let mut var_map: HashMap<String, String> = HashMap::new();
        
        // First pass: identify all variables and add them to the mapping
        for token in &tokens {
            match *token {
                "x" | "X" => {
                    if !var_map.contains_key("x") {
                        let reg = format!("_{:x}", next_reg);
                        next_reg += 1;
                        var_map.insert("x".to_string(), reg.clone());
                        vm_text.push_str(&format!("{} var-x\n", reg));
                    }
                },
                "y" | "Y" => {
                    if !var_map.contains_key("y") {
                        let reg = format!("_{:x}", next_reg);
                        next_reg += 1;
                        var_map.insert("y".to_string(), reg.clone());
                        vm_text.push_str(&format!("{} var-y\n", reg));
                    }
                },
                "z" | "Z" => {
                    if !var_map.contains_key("z") {
                        let reg = format!("_{:x}", next_reg);
                        next_reg += 1;
                        var_map.insert("z".to_string(), reg.clone());
                        vm_text.push_str(&format!("{} var-z\n", reg));
                    }
                },
                token => {
                    // Check if this is a custom variable (not a function name, not a number, not a special token)
                    if !["add", "sub", "mul", "div", "min", "max", "mod", "and", "or", "atan2", "neg",
                         "abs", "recip", "sqrt", "square", "floor", "ceil", "round", "sin", "cos", "tan",
                         "asin", "acos", "atan", "exp", "ln", "not", "compare",
                         "(", ")", ","].contains(&token) &&
                       !token.parse::<f64>().is_ok() {
                        // It's a custom variable
                        if !var_map.contains_key(token) {
                            let reg = format!("_{:x}", next_reg);
                            next_reg += 1;
                            var_map.insert(token.to_string(), reg.clone());
                            vm_text.push_str(&format!("{} var-{}\n", reg, token));
                        }
                    }
                    // Numbers will be handled in the second pass
                },
            }
        }
        
        // Second pass: recursive parsing using a stack-based approach
        // This stack will hold the operations and operands
        let mut op_stack: Vec<&str> = Vec::new();
        let mut output_queue: Vec<String> = Vec::new();
        
        // Helper to get a new register
        let mut get_new_reg = || {
            let reg = format!("_{:x}", next_reg);
            next_reg += 1;
            reg
        };
        
        // Pre-parse into postfix notation using the shunting-yard algorithm
        let mut i = 0;
        while i < tokens.len() {
            match tokens[i] {
                "(" => {
                    op_stack.push("(");
                },
                ")" => {
                    // Pop operators until we find the matching opening parenthesis
                    while let Some(op) = op_stack.pop() {
                        if op == "(" {
                            break;
                        }
                        output_queue.push(op.to_string());
                    }
                },
                "," => {
                    // Commas are used to separate function arguments
                    // Pop operators until we hit a left parenthesis
                    while let Some(op) = op_stack.last() {
                        if *op == "(" {
                            break;
                        }
                        output_queue.push(op_stack.pop().unwrap().to_string());
                    }
                },
                token => {
                    // Check if this is a function
                    if ["add", "sub", "mul", "div", "min", "max", "mod", "and", "or", "atan2", "neg",
                        "abs", "recip", "sqrt", "square", "floor", "ceil", "round", "sin", "cos", "tan",
                        "asin", "acos", "atan", "exp", "ln", "not", "compare"].contains(&token) {
                        // Push function onto operator stack
                        op_stack.push(token);
                    } else if var_map.contains_key(token) {
                        // It's a variable, push it to the output
                        output_queue.push(var_map.get(token).unwrap().clone());
                    } else if let Ok(val) = token.parse::<f64>() {
                        // It's a constant, add it to the VM
                        let reg = get_new_reg();
                        vm_text.push_str(&format!("{} const {}\n", reg, val));
                        output_queue.push(reg);
                    }
                }
            }
            i += 1;
        }
        
        // Pop any remaining operators from the stack
        while let Some(op) = op_stack.pop() {
            if op == "(" {
                return Err(PyValueError::new_err("Mismatched parentheses in F-Rep"));
            }
            output_queue.push(op.to_string());
        }
        
        // Third pass: Convert postfix notation to VM format
        let mut arg_stack: Vec<String> = Vec::new();
        for token in output_queue {
            if ["add", "sub", "mul", "div", "min", "max", "mod", "and", "or", "atan2"].contains(&token.as_str()) {
                // Binary operations
                if arg_stack.len() < 2 {
                    return Err(PyValueError::new_err(format!("Not enough arguments for binary operation {}", token)));
                }
                let arg2 = arg_stack.pop().unwrap();
                let arg1 = arg_stack.pop().unwrap();
                let result_reg = get_new_reg();
                vm_text.push_str(&format!("{} {} {} {}\n", result_reg, token, arg1, arg2));
                arg_stack.push(result_reg);
            } else if ["neg", "abs", "recip", "sqrt", "square", "floor", "ceil", "round",
                       "sin", "cos", "tan", "asin", "acos", "atan", "exp", "ln", "not"].contains(&token.as_str()) {
                // Unary operations
                if arg_stack.is_empty() {
                    return Err(PyValueError::new_err(format!("Not enough arguments for unary operation {}", token)));
                }
                let arg = arg_stack.pop().unwrap();
                let result_reg = get_new_reg();
                vm_text.push_str(&format!("{} {} {}\n", result_reg, token, arg));
                arg_stack.push(result_reg);
            } else {
                // It's a register (variable or constant), just push it to the stack
                arg_stack.push(token);
            }
        }
        
        // Ensure we have exactly one value left on the stack
        if arg_stack.len() != 1 {
            return Err(PyValueError::new_err("Invalid F-Rep expression: too many values left"));
        }
        
        // Use our working VM parser to build the final expression
        self.from_vm(vm_text)
    }
    /// Args:
    ///     handle: The handle of the imported expression.
    ///     values: A numpy array of shape (N, num_vars), where each row is a mapping of variable values.
    ///     variables: A list of Var objects corresponding to the columns of the array.
    ///     backend: Optional string specifying the backend ('vm' or 'jit'). Defaults to 'jit' if available, else 'vm'.
    ///
    /// Returns:
    ///     A numpy array of shape (N,) with the evaluation results.
    #[allow(clippy::too_many_arguments)]
    pub fn eval(
        &self,
        py: Python,
        handle: usize,
        values: PyReadonlyArray2<f32>,
        variables: Vec<PyVar>,
        backend: Option<String>,
    ) -> PyResult<Py<PyArray1<f32>>> {
        let tree = self.trees.get(&handle)
            .ok_or_else(|| PyValueError::new_err("Invalid handle for evaluation (handle must correspond to an imported expression)"))?;

        let use_jit = determine_backend(backend)?;

        let values_view = values.as_array();
        let rust_vars: Vec<Var> = variables.iter().map(|py_var| py_var.var.clone()).collect();

         let results = if use_jit {
             _evaluate_bulk_jit(tree, &values_view, &rust_vars)?
         } else {
             _evaluate_bulk_vm(tree, &values_view, &rust_vars)?
        };

        Ok(PyArray1::from_vec(py, results).into())
    }

}
