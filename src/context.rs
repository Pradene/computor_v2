use std::collections::HashMap;

use crate::error::EvaluationError;
use crate::expression::{BinaryOperator, Expression, Power, UnaryOperator};

#[derive(Debug, Clone, PartialEq)]
pub enum ContextValue {
    Variable(Expression),
    Function {
        params: Vec<String>,
        body: Expression,
    },
}

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
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn assign(&mut self, name: String, value: ContextValue) -> Expression {
        self.variables.insert(name, value.clone());
        match value {
            ContextValue::Variable(expr) => expr,
            ContextValue::Function { body, .. } => body,
        }
    }

    pub fn evaluate_expression(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        self.evaluate_expression_with_scope(expr, &HashMap::new())?
            .reduce()
    }

    fn evaluate_expression_with_scope(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::Real(_) | Expression::Complex(_) | Expression::Matrix(_) => {
                Ok(expr.clone())
            }

            Expression::Variable(name) => self.resolve_variable(name, param_scope),

            Expression::FunctionCall { name, args } => {
                self.evaluate_function_call(name, args, param_scope)
            }

            Expression::BinaryOp { left, op, right } => {
                let left_eval = self
                    .evaluate_expression_with_scope(left, param_scope)?
                    .reduce()?;
                let right_eval = self
                    .evaluate_expression_with_scope(right, param_scope)?
                    .reduce()?;

                match op {
                    BinaryOperator::Add => left_eval + right_eval,
                    BinaryOperator::Subtract => left_eval - right_eval,
                    BinaryOperator::Multiply => left_eval * right_eval,
                    BinaryOperator::Divide => left_eval / right_eval,
                    BinaryOperator::Modulo => left_eval % right_eval,
                    BinaryOperator::Power => left_eval.pow(right_eval),
                }
            }

            Expression::UnaryOp { op, operand } => {
                let operand_eval = self.evaluate_expression_with_scope(operand, param_scope)?;

                match op {
                    UnaryOperator::Plus => Ok(operand_eval),
                    UnaryOperator::Minus => -operand_eval,
                }
            }
        }
    }

    fn resolve_variable(
        &self,
        name: &str,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        if let Some(expr) = param_scope.get(name) {
            return Ok(expr.clone());
        }

        match self.variables.get(name) {
            Some(ContextValue::Variable(expr)) => {
                self.evaluate_expression_with_scope(expr, param_scope)
            }
            Some(ContextValue::Function { .. }) => Err(EvaluationError::InvalidOperation(format!(
                "Cannot use function '{}' as variable",
                name
            ))),
            None => Ok(Expression::Variable(name.to_string())), // Keep symbolic
        }
    }

    fn evaluate_function_call(
        &self,
        name: &str,
        args: &[Expression],
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self.variables.get(name) {
            Some(ContextValue::Function { params, body }) => {
                if args.len() != params.len() {
                    return Err(EvaluationError::WrongArgumentCount {
                        expected: params.len(),
                        got: args.len(),
                    });
                }

                // Evaluate arguments first
                let evaluated_args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| self.evaluate_expression_with_scope(arg, param_scope))
                    .collect();

                let evaluated_args = evaluated_args?;

                // Create function scope
                let mut function_scope = param_scope.clone();
                for (param, arg) in params.iter().zip(evaluated_args.iter()) {
                    function_scope.insert(param.clone(), arg.clone());
                }

                self.evaluate_expression_with_scope(body, &function_scope)
            }
            Some(ContextValue::Variable(_)) => Err(EvaluationError::InvalidOperation(format!(
                "'{}' is not a function",
                name
            ))),
            None => {
                // Evaluate arguments and keep as symbolic function call
                let evaluated_args: Result<Vec<_>, _> = args
                    .iter()
                    .map(|arg| self.evaluate_expression_with_scope(arg, param_scope))
                    .collect();

                Ok(Expression::FunctionCall {
                    name: name.to_string(),
                    args: evaluated_args?,
                })
            }
        }
    }
}
