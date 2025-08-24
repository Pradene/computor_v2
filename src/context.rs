use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expression, UnaryOperator, Value};
use crate::error::EvaluationError;

pub struct Context {
    variables: HashMap<String, Value>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }

    pub fn set_variable(&mut self, name: String, value: Value) {
        self.variables.insert(name, value);
    }

    pub fn get_variable(&self, name: &str) -> Option<&Value> {
        self.variables.get(name)
    }

    pub fn assign(&mut self, name: String, value: Value) {
        self.set_variable(name, value);
    }

    pub fn evaluate_query(&self, expression: &Expression) -> Result<f64, EvaluationError> {
        self.evaluate_expression(expression, &HashMap::new())
    }

    // TODO: Need to return a string or maybe an expression
    // because undefined variable can be a thing
    // also because of matrix type, and complex numbers later
    pub fn evaluate_expression(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, f64>,
    ) -> Result<f64, EvaluationError> {
        match expr {
            Expression::Number(n) => Ok(*n),
            Expression::Matrix(_) => Ok(0f64),
            Expression::Variable(name) => {
                // First check parameter scope, then context variables
                if let Some(&value) = param_scope.get(name) {
                    Ok(value)
                } else {
                    match self.get_variable(name) {
                        Some(Value::Variable(expr)) => self.evaluate_expression(expr, param_scope),
                        Some(Value::Function { .. }) => {
                            Err(EvaluationError::NotAVariable(name.clone()))
                        }
                        None => Err(EvaluationError::UndefinedVariable(name.clone())),
                    }
                }
            }
            Expression::FunctionCall { name, args } => {
                match self.get_variable(name) {
                    Some(Value::Function { params, body }) => {
                        if args.len() != params.len() {
                            return Err(EvaluationError::WrongArgumentCount {
                                expected: params.len(),
                                got: args.len(),
                            });
                        }

                        // Create parameter scope for function call
                        let mut function_scope = HashMap::new();
                        for (param, arg) in params.iter().zip(args.iter()) {
                            let arg_value = self.evaluate_expression(arg, param_scope)?;
                            function_scope.insert(param.clone(), arg_value);
                        }

                        self.evaluate_expression(body, &function_scope)
                    }
                    Some(Value::Variable(_)) => Err(EvaluationError::NotAFunction(name.clone())),
                    None => Err(EvaluationError::UndefinedFunction(name.clone())),
                }
            }
            Expression::BinaryOp { left, op, right } => {
                let left_val = self.evaluate_expression(left, param_scope)?;
                let right_val = self.evaluate_expression(right, param_scope)?;

                match op {
                    BinaryOperator::Add => Ok(left_val + right_val),
                    BinaryOperator::Subtract => Ok(left_val - right_val),
                    BinaryOperator::Multiply => Ok(left_val * right_val),
                    BinaryOperator::Divide => {
                        if right_val == 0.0 {
                            Err(EvaluationError::DivisionByZero)
                        } else {
                            Ok(left_val / right_val)
                        }
                    }
                    BinaryOperator::Power => Ok(left_val.powf(right_val)),
                }
            }
            Expression::UnaryOp { op, operand } => {
                let operand_val = self.evaluate_expression(operand, param_scope)?;
                match op {
                    UnaryOperator::Plus => Ok(operand_val),
                    UnaryOperator::Minus => Ok(-operand_val),
                }
            }
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
