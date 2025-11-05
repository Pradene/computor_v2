use std::collections::HashMap;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::context::{Context, ContextValue};
use crate::error::EvaluationError;
use crate::types::expression::{FunctionCall, Value};
use crate::types::vector::Vector;
use crate::types::{expression::Expression, matrix::Matrix};

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
            Expression::Value(Value::Real(_)) | Expression::Value(Value::Complex(_)) => {
                Ok(expr.clone())
            }

            Expression::Value(Value::Vector(vector)) => {
                let mut evaluated_vector = Vec::new();
                for element in vector.iter() {
                    let evaluated_element = self.evaluate_internal(element)?.reduce()?;
                    evaluated_vector.push(evaluated_element);
                }
                let result = Vector::new(evaluated_vector);
                match result {
                    Ok(vector) => Ok(Expression::Value(Value::Vector(vector))),
                    Err(e) => Err(EvaluationError::InvalidOperation(e)),
                }
            }

            Expression::Value(Value::Matrix(matrix)) => {
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
                    Ok(matrix) => Ok(Expression::Value(Value::Matrix(matrix))),
                    Err(e) => Err(EvaluationError::InvalidOperation(e)),
                }
            }

            Expression::Variable(name) => self.resolve_variable(name),

            Expression::FunctionCall(fc) => self.evaluate_function_call(fc),

            Expression::Add(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.add(right_eval)
            }
            Expression::Sub(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.sub(right_eval)
            }
            Expression::Mul(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.mul(right_eval)
            }
            Expression::Div(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.div(right_eval)
            }
            Expression::Mod(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.rem(right_eval)
            }
            Expression::Pow(left, right) => {
                let left_eval = self.evaluate_internal(left)?.reduce()?;
                let right_eval = self.evaluate_internal(right)?.reduce()?;
                left_eval.pow(right_eval)
            }
            Expression::Neg(inner) => self.evaluate_internal(inner)?.neg(),
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

    fn evaluate_function_call(&self, fc: &FunctionCall) -> Result<Expression, EvaluationError> {
        match self.context.get_variable(fc.name.as_str()) {
            Some(ContextValue::Function(fun)) => {
                if fc.args.len() != fun.params.len() {
                    return Err(EvaluationError::WrongArgumentCount {
                        expected: fun.params.len(),
                        got: fc.args.len(),
                    });
                }

                // Evaluate arguments first
                let evaluated_args: Result<Vec<_>, _> = fc
                    .args
                    .iter()
                    .map(|arg| self.evaluate_internal(arg))
                    .collect();

                let evaluated_args = evaluated_args?;

                // Create function scope by combining current scope with function parameters
                let mut function_scope = self.param_scope.clone();
                for (param, arg) in fun.params.iter().zip(evaluated_args.iter()) {
                    function_scope.insert(param.clone(), arg.clone());
                }

                // Create new evaluator with function scope and evaluate body
                let function_evaluator = Self {
                    context: self.context,
                    param_scope: function_scope,
                };

                function_evaluator.evaluate_internal(&fun.body)
            }
            Some(ContextValue::Variable(_)) => Err(EvaluationError::InvalidOperation(format!(
                "'{}' is not a function",
                fc.name
            ))),
            None => {
                // Evaluate arguments and keep as symbolic function call
                let evaluated_args: Result<Vec<_>, _> = fc
                    .args
                    .iter()
                    .map(|arg| self.evaluate_internal(arg))
                    .collect();

                Ok(Expression::FunctionCall(FunctionCall {
                    name: fc.name.to_string(),
                    args: evaluated_args?,
                }))
            }
        }
    }
}
