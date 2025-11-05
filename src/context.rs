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
        // Move everything to the left side: left - right = 0
        let equation = (left.clone()).sub(right.clone())?;
        let solution = EquationSolver::solve(&equation)?;

        Ok(format!("{}", solution))
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
