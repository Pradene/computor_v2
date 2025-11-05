use std::collections::HashMap;

use crate::error::EvaluationError;
use crate::evaluator::ExpressionEvaluator;
use crate::solver::EquationSolver;
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
    ) -> Result<String, EvaluationError> {
        let equation = (left.clone()).sub(right.clone())?;
        let prepared_equation = self.prepare_equation(&equation)?;

        let solution = EquationSolver::solve(&prepared_equation)?;

        Ok(format!("{}", solution))
    }

    fn prepare_equation(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::FunctionCall { .. } => self.evaluate_expression(expr),
            Expression::BinaryOp { left, op, right } => {
                let left_prep = self.prepare_equation(left)?;
                let right_prep = self.prepare_equation(right)?;
                Ok(Expression::BinaryOp {
                    left: Box::new(left_prep),
                    op: op.clone(),
                    right: Box::new(right_prep),
                })
            }
            Expression::UnaryOp { op, operand } => {
                let operand_prep = self.prepare_equation(operand)?;
                Ok(Expression::UnaryOp {
                    op: op.clone(),
                    operand: Box::new(operand_prep),
                })
            }
            _ => Ok(expr.clone()),
        }
    }

    #[allow(unused)]
    fn contains_variables(expr: &Expression) -> bool {
        match expr {
            Expression::Variable(_) => true,
            Expression::BinaryOp { left, right, .. } => {
                Self::contains_variables(left) || Self::contains_variables(right)
            }
            Expression::UnaryOp { operand, .. } => Self::contains_variables(operand),
            Expression::FunctionCall { args, .. } => args.iter().any(Self::contains_variables),
            Expression::Matrix(matrix) => matrix.iter().any(Self::contains_variables),
            _ => false,
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
