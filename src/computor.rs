use {
    crate::{
        error::EvaluationError,
        expression::{builtin::BuiltinFunction, Expression},
    },
    std::{
        collections::{HashMap, HashSet},
        fmt,
        ops::Sub,
    },
};

#[derive(Debug, Clone, PartialEq)]
pub enum EquationSolution {
    NoSolution,
    Infinite,
    Finite(Vec<Expression>),
}

impl fmt::Display for EquationSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquationSolution::NoSolution => write!(f, "No solution")?,
            EquationSolution::Infinite => write!(f, "Infinite solution (equality)")?,
            EquationSolution::Finite(roots) => match roots.len() {
                1 => write!(f, "The solution is: {}", roots.iter().next().unwrap())?,
                2 => {
                    write!(f, "The solution are: ")?;
                    for (index, root) in roots.iter().enumerate() {
                        write!(f, "{}{}", if index == 0 { "" } else { ", " }, root)?;
                    }
                }
                _ => unreachable!(),
            },
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EquationResult {
    expression: Expression,
    solution: EquationSolution,
}

impl fmt::Display for EquationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} = 0\n{}", self.expression, self.solution)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    pub name: String,
    pub expression: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Symbol {
    Variable(Variable),
    Function(FunctionDefinition),
    BuiltinFunction(BuiltinFunction),
}

impl Symbol {
    pub fn name(&self) -> &str {
        match self {
            Symbol::Variable(variable) => &variable.name,
            Symbol::Function(function) => &function.name,
            Symbol::BuiltinFunction(builtin) => builtin.name(),
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone)]
pub struct Computor {
    table: HashMap<String, Symbol>,
}

impl Default for Computor {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for Computor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (_, value) in self.table.iter() {
            match value {
                Symbol::Variable(Variable { name, expression }) => {
                    writeln!(f, "{} = {}", name, expression)?;
                }
                Symbol::Function(FunctionDefinition { name, params, body }) => {
                    write!(f, "{}", name)?;
                    write!(f, "(")?;
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", param)?;
                    }
                    writeln!(f, ") = {}", body)?;
                }
                Symbol::BuiltinFunction(_) => {}
            }
        }

        Ok(())
    }
}

impl Computor {
    const BUILTINS: &'static [BuiltinFunction] = &[
        BuiltinFunction::Rad,
        BuiltinFunction::Norm,
        BuiltinFunction::Abs,
        BuiltinFunction::Sqrt,
        BuiltinFunction::Cos,
        BuiltinFunction::Sin,
        BuiltinFunction::Tan,
        BuiltinFunction::Dot,
        BuiltinFunction::Cross,
        BuiltinFunction::Exp,
    ];

    pub fn new() -> Self {
        let table = Self::BUILTINS
            .iter()
            .map(|f| (f.name().to_string(), Symbol::BuiltinFunction(f.clone())))
            .collect();

        Computor { table }
    }

    pub fn evaluate_expression(
        &self,
        expression: &Expression,
    ) -> Result<Expression, EvaluationError> {
        expression.evaluate(self)?.simplify()
    }

    pub fn evaluate_equation(
        &self,
        left: &Expression,
        right: &Expression,
    ) -> Result<EquationResult, EvaluationError> {
        let left = left.simplify()?;
        let right = right.simplify()?;
        let expression = self.evaluate_expression(&left.sub(right)?)?;

        if expression.is_zero() {
            return Ok(EquationResult {
                expression,
                solution: EquationSolution::Infinite,
            });
        }

        let roots = expression.find_roots()?;
        match roots.len() {
            0 => Ok(EquationResult {
                expression,
                solution: EquationSolution::NoSolution,
            }),
            1 => Ok(EquationResult {
                expression,
                solution: EquationSolution::Finite(roots),
            }),
            2 => Ok(EquationResult {
                expression,
                solution: EquationSolution::Finite(roots),
            }),
            _ => unreachable!(),
        }
    }

    pub fn assign(&mut self, name: String, symbol: Symbol) -> Result<Expression, EvaluationError> {
        let name = name.to_ascii_lowercase();

        if matches!(self.table.get(&name), Some(Symbol::BuiltinFunction(_))) {
            return Err(EvaluationError::CannotOverrideBuiltin(name));
        }

        // Check for circular dependencies before inserting
        self.check_circular_dependency(&name, &symbol)?;

        // Extract the expression to return evaluated result
        let expression = match &symbol {
            Symbol::Variable(Variable { expression, .. }) => expression.clone(),
            Symbol::Function(FunctionDefinition { body, .. }) => body.clone(),
            Symbol::BuiltinFunction(_) => unreachable!(),
        };

        // Store the original (unevaluated) expression
        self.table.insert(name, symbol);

        // Evaluate to show the current value, but don't store it
        expression.evaluate(self)?.simplify()
    }

    /// Detects circular dependencies like: a = b, then b = a
    fn check_circular_dependency(
        &self,
        name: &str,
        symbol: &Symbol,
    ) -> Result<(), EvaluationError> {
        let mut visited = HashSet::new();
        visited.insert(name.to_string());

        match symbol {
            Symbol::Variable(Variable { expression, .. }) => {
                self.detect_cycle_variable(name, expression, &mut visited)
            }
            Symbol::Function(FunctionDefinition { body, params, .. }) => {
                self.detect_cycle_function(name, body, &mut visited, params)
            }
            Symbol::BuiltinFunction(_) => Ok(()),
        }
    }

    /// Recursively check if assigning `name` to `expression` creates a cycle
    fn detect_cycle_variable(
        &self,
        name: &str,
        expression: &Expression,
        visited: &mut HashSet<String>,
    ) -> Result<(), EvaluationError> {
        let variables = expression.collect_variables();

        for variable in variables {
            if variable == name {
                return Err(EvaluationError::InvalidOperation(format!(
                    "Circular dependency detected: cannot define '{}' in terms of itself",
                    name
                )));
            }

            // If we've already visited this variable in this path, we have a cycle
            if visited.contains(&variable) {
                return Err(EvaluationError::InvalidOperation(format!(
                    "Circular dependency detected: '{}' depends on '{}' which depends back on '{}'",
                    name, variable, name
                )));
            }

            visited.insert(variable.clone());

            // Recursively check this variable's dependencies
            if let Some(Symbol::Variable(Variable {
                expression: var_expr,
                ..
            })) = self.get_symbol(&variable)
            {
                self.detect_cycle_variable(name, var_expr, visited)?;
            } else if let Some(Symbol::Function(FunctionDefinition { body, params, .. })) =
                self.get_symbol(&variable)
            {
                // For functions, we need to check the free variables in the body
                // (excluding parameters which are bound)
                let func_vars = body.collect_variables();

                // Filter out parameters (they're local bindings)
                let free_vars: Vec<String> = func_vars
                    .into_iter()
                    .filter(|v| !params.contains(v))
                    .collect();

                // Check each free variable
                for free_var in free_vars {
                    if free_var == name {
                        return Err(EvaluationError::InvalidOperation(format!(
                            "Circular dependency detected: '{}' depends on function '{}' which uses '{}'",
                            name, variable, free_var
                        )));
                    }

                    if visited.contains(&free_var) {
                        return Err(EvaluationError::InvalidOperation(format!(
                            "Circular dependency detected: '{}' -> '{}' -> '{}' -> '{}'",
                            name, variable, free_var, name
                        )));
                    }

                    visited.insert(free_var.clone());

                    if let Some(Symbol::Variable(Variable {
                        expression: fv_expr,
                        ..
                    })) = self.get_symbol(&free_var)
                    {
                        self.detect_cycle_variable(name, fv_expr, visited)?;
                    } else if let Some(Symbol::Function(FunctionDefinition {
                        body: fv_body,
                        params: fv_params,
                        ..
                    })) = self.get_symbol(&free_var)
                    {
                        self.detect_cycle_function(name, fv_body, visited, fv_params)?;
                    }

                    visited.remove(&free_var);
                }
            }

            visited.remove(&variable);
        }

        Ok(())
    }

    /// Check cycles in function bodies, excluding parameters
    fn detect_cycle_function(
        &self,
        name: &str,
        expression: &Expression,
        visited: &mut HashSet<String>,
        params: &[String],
    ) -> Result<(), EvaluationError> {
        let variables = expression.collect_variables();

        for variable in variables {
            // Skip function parameters - they're local, not free variables
            if params.iter().any(|p| p == &variable) {
                continue;
            }

            if variable == name {
                return Err(EvaluationError::InvalidOperation(format!(
                    "Circular dependency detected: cannot define '{}' in terms of itself",
                    name
                )));
            }

            if visited.contains(&variable) {
                return Err(EvaluationError::InvalidOperation(format!(
                    "Circular dependency detected: '{}' depends on '{}' which depends back on '{}'",
                    name, variable, name
                )));
            }

            visited.insert(variable.clone());

            if let Some(Symbol::Variable(Variable {
                expression: var_expr,
                ..
            })) = self.get_symbol(&variable)
            {
                self.detect_cycle_variable(name, var_expr, visited)?;
            } else if let Some(Symbol::Function(FunctionDefinition {
                body,
                params: fn_params,
                ..
            })) = self.get_symbol(&variable)
            {
                // Check free variables in nested function
                let func_vars = body.collect_variables();

                let free_vars: Vec<String> = func_vars
                    .into_iter()
                    .filter(|v| !fn_params.contains(v))
                    .collect();

                for free_var in free_vars {
                    if free_var == name {
                        return Err(EvaluationError::InvalidOperation(format!(
                            "Circular dependency detected: '{}' depends on function '{}' which uses '{}'",
                            name, variable, free_var
                        )));
                    }

                    if visited.contains(&free_var) {
                        continue;
                    }

                    visited.insert(free_var.clone());

                    if let Some(Symbol::Variable(Variable {
                        expression: fv_expr,
                        ..
                    })) = self.get_symbol(&free_var)
                    {
                        self.detect_cycle_variable(name, fv_expr, visited)?;
                    } else if let Some(Symbol::Function(FunctionDefinition {
                        body: fv_body,
                        params: fv_params,
                        ..
                    })) = self.get_symbol(&free_var)
                    {
                        self.detect_cycle_function(name, fv_body, visited, fv_params)?;
                    }

                    visited.remove(&free_var);
                }
            }

            visited.remove(&variable);
        }

        Ok(())
    }

    pub fn unset(&mut self, name: &str) -> Option<Symbol> {
        self.table.remove(name.to_ascii_lowercase().as_str())
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.table.get(name.to_ascii_lowercase().as_str())
    }
}
