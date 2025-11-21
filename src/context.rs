use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Sub;

use crate::equation::{Equation, EquationSolution};
use crate::error::{ComputorError, EvaluationError};
use crate::expression::Expression;
use crate::parser::Parser;

#[derive(Debug, Clone)]
pub enum Statement {
    Assignment { name: String, value: Symbol },
    Equation { left: Expression, right: Expression },
    Query { expression: Expression },
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatementResult {
    Value(Expression),
    Solution(EquationSolution),
}

impl fmt::Display for StatementResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StatementResult::Value(value) => write!(f, "{}", value),
            StatementResult::Solution(solution) => write!(f, "{}", solution),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinFunction {
    Rad,
    Norm,
    Abs,
    Sqrt,
    Cos,
    Sin,
    Tan,
    Dot,
    Cross,
    Exp,
}

impl BuiltinFunction {
    pub fn name(&self) -> &str {
        match self {
            BuiltinFunction::Rad => "rad",
            BuiltinFunction::Norm => "norm",
            BuiltinFunction::Abs => "abs",
            BuiltinFunction::Sqrt => "sqrt",
            BuiltinFunction::Cos => "cos",
            BuiltinFunction::Sin => "sin",
            BuiltinFunction::Tan => "tan",
            BuiltinFunction::Dot => "dot",
            BuiltinFunction::Cross => "cross",
            BuiltinFunction::Exp => "exp",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            BuiltinFunction::Rad => 1,
            BuiltinFunction::Norm => 1,
            BuiltinFunction::Abs => 1,
            BuiltinFunction::Sqrt => 1,
            BuiltinFunction::Cos => 1,
            BuiltinFunction::Sin => 1,
            BuiltinFunction::Tan => 1,
            BuiltinFunction::Dot => 2,
            BuiltinFunction::Cross => 2,
            BuiltinFunction::Exp => 1,
        }
    }

    pub fn call(&self, args: &[Expression]) -> Result<Expression, EvaluationError> {
        match self {
            BuiltinFunction::Rad => args[0].rad(),
            BuiltinFunction::Norm => args[0].norm(),
            BuiltinFunction::Abs => args[0].abs(),
            BuiltinFunction::Sqrt => args[0].sqrt(),
            BuiltinFunction::Cos => args[0].cos(),
            BuiltinFunction::Sin => args[0].sin(),
            BuiltinFunction::Tan => args[0].tan(),
            BuiltinFunction::Dot => args[0].dot(args[1].clone()),
            BuiltinFunction::Cross => args[0].cross(args[1].clone()),
            BuiltinFunction::Exp => args[0].exp(),
        }
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
pub struct Context {
    table: HashMap<String, Symbol>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        const BUILTINS: &[BuiltinFunction] = &[
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

        let table = BUILTINS
            .iter()
            .map(|f| (f.name().to_string(), Symbol::BuiltinFunction(f.clone())))
            .collect();

        Context { table }
    }

    pub fn compute(&mut self, line: &str) -> Result<StatementResult, ComputorError> {
        let statement = Parser::parse(line).map_err(|e| ComputorError::Parsing(e.to_string()))?;
        let result = self.execute(statement).map_err(ComputorError::Evaluation)?;

        Ok(result)
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.table.get(name.to_ascii_lowercase().as_str())
    }

    pub fn print_table(&self) {
        for (_, value) in self.table.iter() {
            match value {
                Symbol::Variable(Variable { name, expression }) => {
                    println!("{} = {}", name, expression)
                }
                Symbol::Function(FunctionDefinition { name, params, body }) => {
                    print!("{}", name);
                    print!("(");
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 {
                            print!(", ");
                        }
                        print!("{}", param);
                    }
                    println!(") = {}", body);
                }
                Symbol::BuiltinFunction(_) => {}
            }
        }
    }

    pub fn execute(&mut self, statement: Statement) -> Result<StatementResult, EvaluationError> {
        match statement {
            Statement::Assignment { name, value } => {
                let result = self.assign(name, value)?;
                Ok(StatementResult::Value(result))
            }
            Statement::Query { expression } => {
                let result = self.evaluate_expression(&expression)?;
                Ok(StatementResult::Value(result))
            }
            Statement::Equation { left, right } => {
                let result = self.evaluate_equation(&left, &right)?;
                Ok(StatementResult::Solution(result))
            }
        }
    }

    pub fn evaluate_expression(
        &self,
        expression: &Expression,
    ) -> Result<Expression, EvaluationError> {
        expression.evaluate(self)?.reduce()
    }

    pub fn evaluate_equation(
        &self,
        left: &Expression,
        right: &Expression,
    ) -> Result<EquationSolution, EvaluationError> {
        let expression = (left.clone()).sub(right.clone())?;
        let expression = self.evaluate_expression(&expression)?;

        let equation = Equation::new(expression);
        equation.solve()
    }

    fn is_builtin(&self, name: &str) -> bool {
        matches!(self.table.get(name), Some(Symbol::BuiltinFunction(_)))
    }

    pub fn assign(&mut self, name: String, symbol: Symbol) -> Result<Expression, EvaluationError> {
        let name = name.to_ascii_lowercase();

        if self.is_builtin(&name) {
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
        expression.evaluate(self)?.reduce()
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
                self.detect_cycle(name, expression, &mut visited)
            }
            Symbol::Function(FunctionDefinition { body, params, .. }) => {
                self.detect_cycle_function(name, body, &mut visited, params)
            }
            Symbol::BuiltinFunction(_) => Ok(()),
        }
    }

    /// Recursively check if assigning `name` to `expression` creates a cycle
    fn detect_cycle(
        &self,
        name: &str,
        expression: &Expression,
        visited: &mut HashSet<String>,
    ) -> Result<(), EvaluationError> {
        let mut variables = Vec::new();
        Self::collect_all_variables(expression, &mut variables);

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
                self.detect_cycle(name, var_expr, visited)?;
            } else if let Some(Symbol::Function(FunctionDefinition { body, params, .. })) =
                self.get_symbol(&variable)
            {
                // For functions, we need to check the free variables in the body
                // (excluding parameters which are bound)
                let mut func_vars = Vec::new();
                Self::collect_all_variables(body, &mut func_vars);

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
                        self.detect_cycle(name, fv_expr, visited)?;
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
        let mut variables = Vec::new();
        Self::collect_all_variables(expression, &mut variables);

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
                self.detect_cycle(name, var_expr, visited)?;
            } else if let Some(Symbol::Function(FunctionDefinition {
                body,
                params: fn_params,
                ..
            })) = self.get_symbol(&variable)
            {
                // Check free variables in nested function
                let mut func_vars = Vec::new();
                Self::collect_all_variables(body, &mut func_vars);

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
                        self.detect_cycle(name, fv_expr, visited)?;
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

    /// Collect all variables referenced in an expression (transitive)
    fn collect_all_variables(expression: &Expression, variables: &mut Vec<String>) {
        match expression {
            Expression::Variable(name) => {
                if !variables.contains(name) {
                    variables.push(name.clone());
                }
            }
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                Self::collect_all_variables(left, variables);
                Self::collect_all_variables(right, variables);
            }
            Expression::Neg(inner) | Expression::Paren(inner) => {
                Self::collect_all_variables(inner, variables);
            }
            Expression::FunctionCall(fc) => {
                // Collect variables from arguments
                for arg in &fc.args {
                    Self::collect_all_variables(arg, variables);
                }
                // Also add the function name itself as a dependency
                if !variables.contains(&fc.name) {
                    variables.push(fc.name.clone());
                }
            }
            Expression::Vector(v) => {
                for elem in v {
                    Self::collect_all_variables(elem, variables);
                }
            }
            Expression::Matrix(data, _, _) => {
                for elem in data {
                    Self::collect_all_variables(elem, variables);
                }
            }
            _ => {}
        }
    }

    pub fn unset(&mut self, name: &str) -> Option<Symbol> {
        self.table.remove(name.to_ascii_lowercase().as_str())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        context::{Context, StatementResult},
        equation::EquationSolution,
        error::{ComputorError, EvaluationError},
        expression::Expression,
    };

    fn compute_ok(line: &str) -> StatementResult {
        Context::new().compute(line).expect("computation failed")
    }

    #[test]
    fn assign() {
        let cases = vec![
            ("a = 5", 5.0),
            ("b = 12 * 5 - 10", 50.0),
            ("c = (35 / 5 + 5) * 12", 144.0),
        ];

        for (input, expected) in cases {
            let result = compute_ok(input);
            assert_eq!(result, StatementResult::Value(Expression::Real(expected)));
        }
    }

    #[test]
    fn addition() {
        let cases = vec![
            ("2 + 3 = ?", 5.0),
            ("10 + 10 = ?", 20.0),
            ("3.2 + 4 + 4 + 4 = ?", 15.2),
        ];

        for (input, expected) in cases {
            let result = compute_ok(input);
            assert_eq!(result, StatementResult::Value(Expression::Real(expected)));
        }
    }

    #[test]
    fn division_by_zero() {
        let mut context = Context::new();
        let result = context.compute("5 / 0 = ?");

        assert!(result.is_err(), "division by zero should fail");
        assert_eq!(
            result.unwrap_err(),
            ComputorError::Evaluation(EvaluationError::DivisionByZero)
        );
    }

    #[test]
    fn multiply() {
        let mut context = Context::new();
        let result = context
            .compute("5 * ( 5 + 7 ) = ?")
            .expect("computation should succeed");
        assert_eq!(result, StatementResult::Value(Expression::Real(60.0)));
    }

    #[test]
    fn multiply_by_zero() {
        let mut context = Context::new();
        let result = context
            .compute("0 * ( -50 + 7 ) = ?")
            .expect("computation should succeed");
        assert_eq!(result, StatementResult::Value(Expression::Real(0.0)));
    }

    #[test]
    fn first_degree_equation() {
        let mut context = Context::new();
        let result = context
            .compute("2x = 4 ?")
            .expect("computation should succeed");

        assert_eq!(
            result,
            StatementResult::Solution(EquationSolution::Finite {
                variable: "x".to_string(),
                solutions: vec![Expression::Real(2.0)]
            })
        );
    }

    #[test]
    fn second_degree_equation() {
        let mut context = Context::new();
        let result = context
            .compute("-4x^2 + 12x + 4 = 4 ?")
            .expect("computation should succeed");

        assert_eq!(
            result,
            StatementResult::Solution(EquationSolution::Finite {
                variable: "x".to_string(),
                solutions: vec![Expression::Real(0.0), Expression::Real(3.0)]
            })
        );
    }

    #[test]
    fn variable_assignment_and_reuse() {
        let mut context = Context::new();

        context
            .compute("a = 5")
            .expect("first assignment should succeed");
        let result = context
            .compute("a + 3 = ?")
            .expect("computation should succeed");

        assert_eq!(result, StatementResult::Value(Expression::Real(8.0)));
    }

    #[test]
    fn invalid_syntax_should_fail() {
        let mut context = Context::new();
        let result = context.compute("--5 + +3 = ?");

        assert!(result.is_err(), "invalid syntax should fail");
        assert!(matches!(result, Err(ComputorError::Parsing(_))));
    }

    #[test]
    fn undefined_variable_should_work() {
        let mut context = Context::new();
        let result = context.compute("undefined + 5 = ?");

        assert!(result.is_ok(), "undefined variable should work");
    }

    #[test]
    fn zero_equation_should_be_infinite() {
        let mut context = Context::new();
        let result = context
            .compute("x = x ?")
            .expect("computation should succeed");

        match result {
            StatementResult::Solution(EquationSolution::Infinite { .. }) => {}
            _ => panic!("equation with infinite solutions should return Infinite variant"),
        }
    }

    #[test]
    fn impossible_equation_should_be_empty() {
        let mut context = Context::new();
        let result = context
            .compute("0 = 1 ?")
            .expect("computation should succeed");

        match result {
            StatementResult::Solution(EquationSolution::NoSolution { .. }) => {}
            _ => panic!("impossible equation should return Empty variant"),
        }
    }
}
