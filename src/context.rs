use std::collections::HashMap;
use std::ops::Sub;

use crate::equation::{Equation, EquationSolution};
use crate::error::EvaluationError;
use crate::expression::Expression;

#[derive(Debug, Clone)]
pub enum Statement {
    Assignment { name: String, value: Symbol },
    Equation { left: Expression, right: Expression },
    Query { expression: Expression },
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
    symbols: HashMap<String, Symbol>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Context {
            symbols: HashMap::new(),
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
        let expression = self.prepare_equation(&expression)?;

        let equation = Equation::new(expression);
        equation.solve()
    }

    fn prepare_equation(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::FunctionCall(_) => self.evaluate_expression(expr),
            Expression::Add(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Add(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Div(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Div(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Mul(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Mul(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Mod(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Mod(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Pow(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Pow(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Sub(left, right) => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::Sub(Box::new(left_prep), Box::new(right_prep)))
            }
            Expression::Neg(inner) => {
                let inner = self.prepare_equation(inner)?;
                Ok(Expression::Neg(Box::new(inner)))
            }
            _ => Ok(expr.clone()),
        }
    }

    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    pub fn assign(&mut self, name: String, symbol: Symbol) -> Result<Expression, EvaluationError> {
        self.symbols.insert(name, symbol.clone());
        let expr = match symbol {
            Symbol::Variable(expr) => expr,
            Symbol::Function(func) => func.body,
        };

        expr.evaluate(self)?.reduce()
    }
}
