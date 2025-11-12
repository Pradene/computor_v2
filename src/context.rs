use std::collections::HashMap;
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

#[derive(Debug, Clone)]
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
pub struct FunctionDefinition {
    pub params: Vec<String>,
    pub body: Expression,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Symbol {
    Variable(Expression),
    Function(FunctionDefinition),
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
        Context {
            table: HashMap::new(),
        }
    }

    pub fn compute(&mut self, line: &str) -> Result<StatementResult, ComputorError> {
        let statement = Parser::parse(line).map_err(|e| ComputorError::Parsing(e.to_string()))?;
        let result = self
            .execute(statement)
            .map_err(ComputorError::Evaluation)?;

        Ok(result)
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.table.get(name)
    }

    pub fn print_table(&self) {
        for (name, symbol) in self.table.iter() {
            match symbol {
                Symbol::Variable(variable) => println!("{} = {}", name, variable),
                Symbol::Function(FunctionDefinition { params, body }) => {
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

    pub fn evaluate_expression(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        expr.evaluate(self)?.reduce()
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

    pub fn assign(&mut self, name: String, symbol: Symbol) -> Result<Expression, EvaluationError> {
        self.table.insert(name, symbol.clone());
        let expr = match symbol {
            Symbol::Variable(expr) => expr,
            Symbol::Function(func) => func.body,
        };

        expr.evaluate(self)?.reduce()
    }
}
