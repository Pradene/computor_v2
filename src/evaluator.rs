use std::collections::HashMap;

use crate::context::{Context, ContextValue};
use crate::error::EvaluationError;
use crate::types::{
    expression::{BinaryOperator, Expression, UnaryOperator},
    matrix::Matrix,
};

#[derive(Debug, Clone)]
pub struct ExpressionEvaluator<'a> {
    context: &'a Context,
    param_scope: HashMap<String, Expression>,
}

impl<'a> ExpressionEvaluator<'a> {
    pub fn new(context: &'a Context) -> Self {
        Self {
            context,
            param_scope: HashMap::new(),
        }
    }

    pub fn evaluate(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        self.evaluate_internal(expr)?.reduce()
    }

    fn evaluate_internal(&self, expr: &Expression) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::Real(_) | Expression::Complex(_) => Ok(expr.clone()),

            Expression::Matrix(matrix) => {
                let mut evaluated_matrix = Vec::new();
                for row in 0..matrix.rows() {
                    for col in 0..matrix.cols() {
                        let evaluated_element = self
                            .evaluate_internal(matrix.get(row, col).unwrap())?
                            .reduce()?;
                        evaluated_matrix.push(evaluated_element);
                    }
                }
                let result = Matrix::new(evaluated_matrix, matrix.rows(), matrix.cols());
                match result {
                    Ok(matrix) => Ok(Expression::Matrix(matrix)),
                    Err(e) => Err(EvaluationError::InvalidOperation(e)),
                }
            }

            Expression::Variable(name) => self.resolve_variable(name),

            Expression::FunctionCall { name, args } => self.evaluate_function_call(name, args),

            Expression::BinaryOp { left, op, right } => {
                self.evaluate_binary_operation(left, op, right)
            }

            Expression::UnaryOp { op, operand } => self.evaluate_unary_operation(op, operand),
        }
    }

    fn resolve_variable(&self, name: &str) -> Result<Expression, EvaluationError> {
        // First check function parameter scope
        if let Some(expr) = self.param_scope.get(name) {
            return Ok(expr.clone());
        }

        // Then check context variables
        match self.context.get_variable(name) {
            Some(ContextValue::Variable(expr)) => {
                // Recursively evaluate the variable's expression
                self.evaluate_internal(expr)
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
    ) -> Result<Expression, EvaluationError> {
        match self.context.get_variable(name) {
            Some(ContextValue::Function { params, body }) => {
                if args.len() != params.len() {
                    return Err(EvaluationError::WrongArgumentCount {
                        expected: params.len(),
                        got: args.len(),
                    });
                }

                // Evaluate arguments first
                let evaluated_args: Result<Vec<_>, _> =
                    args.iter().map(|arg| self.evaluate_internal(arg)).collect();

                let evaluated_args = evaluated_args?;

                // Create function scope by combining current scope with function parameters
                let mut function_scope = self.param_scope.clone();
                for (param, arg) in params.iter().zip(evaluated_args.iter()) {
                    function_scope.insert(param.clone(), arg.clone());
                }

                // Create new evaluator with function scope and evaluate body
                let function_evaluator = Self {
                    context: self.context,
                    param_scope: function_scope,
                };

                function_evaluator.evaluate_internal(body)
            }
            Some(ContextValue::Variable(_)) => Err(EvaluationError::InvalidOperation(format!(
                "'{}' is not a function",
                name
            ))),
            None => {
                // Evaluate arguments and keep as symbolic function call
                let evaluated_args: Result<Vec<_>, _> =
                    args.iter().map(|arg| self.evaluate_internal(arg)).collect();

                Ok(Expression::FunctionCall {
                    name: name.to_string(),
                    args: evaluated_args?,
                })
            }
        }
    }

    fn evaluate_binary_operation(
        &self,
        left: &Expression,
        op: &BinaryOperator,
        right: &Expression,
    ) -> Result<Expression, EvaluationError> {
        let left_eval = self.evaluate_internal(left)?.reduce()?;
        let right_eval = self.evaluate_internal(right)?.reduce()?;

        match op {
            BinaryOperator::Add => left_eval.add(right_eval),
            BinaryOperator::Subtract => left_eval.sub(right_eval),
            BinaryOperator::Multiply => left_eval.mul(right_eval),
            BinaryOperator::Divide => left_eval.div(right_eval),
            BinaryOperator::Modulo => left_eval.rem(right_eval),
            BinaryOperator::Power => left_eval.pow(right_eval),
        }
    }

    fn evaluate_unary_operation(
        &self,
        op: &UnaryOperator,
        operand: &Expression,
    ) -> Result<Expression, EvaluationError> {
        let operand_eval = self.evaluate_internal(operand)?;

        match op {
            UnaryOperator::Plus => Ok(operand_eval),
            UnaryOperator::Minus => operand_eval.neg(),
        }
    }
}
