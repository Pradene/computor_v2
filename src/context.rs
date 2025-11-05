use std::collections::HashMap;
use std::ops::Sub;

use crate::error::EvaluationError;
use crate::evaluator::ExpressionEvaluator;
use crate::solver::{EquationSolution, EquationSolver};
use crate::types::expression::Expression;

#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    Variable(Expression),
    Function {
        params: Vec<String>,
        body: Expression,
    },
}

#[derive(Debug, Clone)]
pub struct Context {
    variables: HashMap<String, ContextValue>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    pub fn new() -> Self {
        Context {
            variables: HashMap::new(),
        }
    }

    pub fn evaluate_expression(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        ExpressionEvaluator::new(self).evaluate(expr)
    }

    pub fn evaluate_equation(
        &self,
        left: &Expression,
        right: &Expression,
    ) -> Result<EquationSolution, EvaluationError> {
        let equation = (left.clone()).sub(right.clone())?;
        let prepared_equation = self.prepare_equation(&equation)?;

        EquationSolver::solve(&prepared_equation)
    }

    fn prepare_equation(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::FunctionCall { .. } => self.evaluate_expression(expr),
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

    pub fn get_variable(&self, name: &str) -> Option<&ContextValue> {
        self.variables.get(name)
    }

    pub fn assign(
        &mut self,
        name: String,
        value: ContextValue,
    ) -> Result<Expression, EvaluationError> {
        self.variables.insert(name, value.clone());
        let expr = match value {
            ContextValue::Variable(expr) => expr,
            ContextValue::Function { body, .. } => body,
        };

        ExpressionEvaluator::new(self).evaluate(&expr)
    }
}
