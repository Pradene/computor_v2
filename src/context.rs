use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expression, UnaryOperator, Value};
use crate::error::EvaluationError;
use crate::Matrix;

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

    pub fn evaluate_query(&self, expression: &Expression) -> Result<Expression, EvaluationError> {
        self.evaluate_expression(expression, &HashMap::new())
    }

    pub fn evaluate_expression(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        let reduced = self.reduce_expression(expr, param_scope)?;
        Ok(reduced)
    }

    pub fn reduce_expression(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::Number(_) => Ok(expr.clone()),

            Expression::Matrix(matrix) => {
                let mut reduced_data = Vec::new();
                for row in matrix.iter() {
                    let mut reduced_row = Vec::new();
                    for elem in row {
                        let reduced_elem = self.reduce_expression(elem, param_scope)?;
                        reduced_row.push(reduced_elem);
                    }
                    reduced_data.push(reduced_row);
                }
                let reduced_matrix =
                    Matrix::new(reduced_data).map_err(|e| EvaluationError::InvalidMatrix(e))?;
                Ok(Expression::Matrix(reduced_matrix))
            }

            Expression::Variable(name) => {
                // First check parameter scope
                if let Some(expr) = param_scope.get(name) {
                    self.reduce_expression(expr, param_scope)
                } else {
                    match self.get_variable(name) {
                        Some(Value::Variable(expr)) => self.reduce_expression(expr, param_scope),
                        Some(Value::Function { .. }) => {
                            Err(EvaluationError::NotAVariable(name.clone()))
                        }
                        None => Ok(Expression::Variable(name.clone())), // Keep symbolic
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

                        // Reduce arguments first
                        let mut reduced_args = Vec::new();
                        for arg in args {
                            reduced_args.push(self.reduce_expression(arg, param_scope)?);
                        }

                        // Create parameter scope for function call
                        let mut function_scope = HashMap::new();
                        for (param, reduced_arg) in params.iter().zip(reduced_args.iter()) {
                            function_scope.insert(param.clone(), reduced_arg.clone());
                        }

                        self.reduce_expression(body, &function_scope)
                    }
                    Some(Value::Variable(_)) => Err(EvaluationError::NotAFunction(name.clone())),
                    None => {
                        // Reduce arguments and keep as function call
                        let mut reduced_args = Vec::new();
                        for arg in args {
                            reduced_args.push(self.reduce_expression(arg, param_scope)?);
                        }
                        Ok(Expression::FunctionCall {
                            name: name.clone(),
                            args: reduced_args,
                        })
                    }
                }
            }

            Expression::BinaryOp { left, op, right } => {
                let left_reduced = self.reduce_expression(left, param_scope)?;
                let right_reduced = self.reduce_expression(right, param_scope)?;
                self.reduce_binary_op(&left_reduced, op, &right_reduced)
            }

            Expression::UnaryOp { op, operand } => {
                let operand_reduced = self.reduce_expression(operand, param_scope)?;
                self.reduce_unary_op(op, &operand_reduced)
            }
        }
    }

    fn reduce_binary_op(
        &self,
        left: &Expression,
        op: &BinaryOperator,
        right: &Expression,
    ) -> Result<Expression, EvaluationError> {
        match (left, right) {
            // Both are numbers - compute directly
            (Expression::Number(a), Expression::Number(b)) => match op {
                BinaryOperator::Add => Ok(Expression::Number(a + b)),
                BinaryOperator::Subtract => Ok(Expression::Number(a - b)),
                BinaryOperator::Multiply => Ok(Expression::Number(a * b)),
                BinaryOperator::Divide => {
                    if *b == 0.0 {
                        Err(EvaluationError::DivisionByZero)
                    } else {
                        Ok(Expression::Number(a / b))
                    }
                }
                BinaryOperator::Power => Ok(Expression::Number(a.powf(*b))),
            },

            // Addition simplifications
            (Expression::Number(0.0), right) if matches!(op, BinaryOperator::Add) => {
                Ok(right.clone())
            }
            (left, Expression::Number(0.0)) if matches!(op, BinaryOperator::Add) => {
                Ok(left.clone())
            }

            // Subtraction simplifications
            (left, Expression::Number(0.0)) if matches!(op, BinaryOperator::Subtract) => {
                Ok(left.clone())
            }
            (Expression::Number(0.0), right) if matches!(op, BinaryOperator::Subtract) => {
                self.reduce_unary_op(&UnaryOperator::Minus, right)
            }

            // Multiplication simplifications
            (Expression::Number(0.0), _) | (_, Expression::Number(0.0))
                if matches!(op, BinaryOperator::Multiply) =>
            {
                Ok(Expression::Number(0.0))
            }
            (Expression::Number(1.0), right) if matches!(op, BinaryOperator::Multiply) => {
                Ok(right.clone())
            }
            (left, Expression::Number(1.0)) if matches!(op, BinaryOperator::Multiply) => {
                Ok(left.clone())
            }

            // Division simplifications
            (left, Expression::Number(1.0)) if matches!(op, BinaryOperator::Divide) => {
                Ok(left.clone())
            }
            (Expression::Number(0.0), right) if matches!(op, BinaryOperator::Divide) => {
                // Check if right is zero
                if let Expression::Number(0.0) = right {
                    Err(EvaluationError::DivisionByZero)
                } else {
                    Ok(Expression::Number(0.0))
                }
            }

            // Power simplifications
            (_, Expression::Number(0.0)) if matches!(op, BinaryOperator::Power) => {
                Ok(Expression::Number(1.0)) // x^0 = 1 (assuming x != 0)
            }
            (left, Expression::Number(1.0)) if matches!(op, BinaryOperator::Power) => {
                Ok(left.clone())
            }
            (Expression::Number(1.0), _) if matches!(op, BinaryOperator::Power) => {
                Ok(Expression::Number(1.0))
            }
            (Expression::Number(0.0), right) if matches!(op, BinaryOperator::Power) => {
                // 0^x = 0 if x > 0, undefined if x <= 0
                match right {
                    Expression::Number(n) if *n > 0.0 => Ok(Expression::Number(0.0)),
                    Expression::Number(n) if *n <= 0.0 => Err(EvaluationError::UndefinedOperation),
                    _ => Ok(Expression::BinaryOp {
                        left: Box::new(left.clone()),
                        op: op.clone(),
                        right: Box::new(right.clone()),
                    }),
                }
            }

            // Combine like terms: x + x = 2*x
            (left, right)
                if self.expressions_equal(left, right) && matches!(op, BinaryOperator::Add) =>
            {
                Ok(Expression::BinaryOp {
                    left: Box::new(Expression::Number(2.0)),
                    op: BinaryOperator::Multiply,
                    right: Box::new(left.clone()),
                })
            }

            // Combine like terms: x - x = 0
            (left, right)
                if self.expressions_equal(left, right)
                    && matches!(op, BinaryOperator::Subtract) =>
            {
                Ok(Expression::Number(0.0))
            }

            // Associativity and distributivity optimizations
            // (a + b) + c -> a + (b + c) if b and c are numbers
            (
                Expression::BinaryOp {
                    left: a,
                    op: BinaryOperator::Add,
                    right: b,
                },
                c,
            ) if matches!(op, BinaryOperator::Add) => {
                if let (Expression::Number(b_val), Expression::Number(c_val)) = (b.as_ref(), c) {
                    let bc_sum = Expression::Number(b_val + c_val);
                    self.reduce_binary_op(a, &BinaryOperator::Add, &bc_sum)
                } else {
                    Ok(Expression::BinaryOp {
                        left: Box::new(left.clone()),
                        op: op.clone(),
                        right: Box::new(right.clone()),
                    })
                }
            }

            // a * (b + c) -> a*b + a*c if we want to expand (optional)
            // For now, we'll keep it factored

            // Fraction simplifications: (a/b) * (c/d) = (a*c)/(b*d)
            (
                Expression::BinaryOp {
                    left: a,
                    op: BinaryOperator::Divide,
                    right: b,
                },
                Expression::BinaryOp {
                    left: c,
                    op: BinaryOperator::Divide,
                    right: d,
                },
            ) if matches!(op, BinaryOperator::Multiply) => {
                let numerator = self.reduce_binary_op(a, &BinaryOperator::Multiply, c)?;
                let denominator = self.reduce_binary_op(b, &BinaryOperator::Multiply, d)?;
                self.reduce_binary_op(&numerator, &BinaryOperator::Divide, &denominator)
            }

            // (a/b) / (c/d) = (a*d)/(b*c)
            (
                Expression::BinaryOp {
                    left: a,
                    op: BinaryOperator::Divide,
                    right: b,
                },
                Expression::BinaryOp {
                    left: c,
                    op: BinaryOperator::Divide,
                    right: d,
                },
            ) if matches!(op, BinaryOperator::Divide) => {
                let numerator = self.reduce_binary_op(a, &BinaryOperator::Multiply, d)?;
                let denominator = self.reduce_binary_op(b, &BinaryOperator::Multiply, c)?;
                self.reduce_binary_op(&numerator, &BinaryOperator::Divide, &denominator)
            }

            // Default case - no further reduction possible
            _ => Ok(Expression::BinaryOp {
                left: Box::new(left.clone()),
                op: op.clone(),
                right: Box::new(right.clone()),
            }),
        }
    }

    fn reduce_unary_op(
        &self,
        op: &UnaryOperator,
        operand: &Expression,
    ) -> Result<Expression, EvaluationError> {
        match (op, operand) {
            // Direct computation for numbers
            (UnaryOperator::Plus, Expression::Number(n)) => Ok(Expression::Number(*n)),
            (UnaryOperator::Minus, Expression::Number(n)) => Ok(Expression::Number(-n)),

            // Double negative: -(-x) = x
            (
                UnaryOperator::Minus,
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: inner,
                },
            ) => Ok(inner.as_ref().clone()),

            // +(-x) = -x
            (
                UnaryOperator::Plus,
                expr @ Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    ..
                },
            ) => Ok(expr.clone()),

            // -(+x) = -x
            (
                UnaryOperator::Minus,
                Expression::UnaryOp {
                    op: UnaryOperator::Plus,
                    operand: inner,
                },
            ) => self.reduce_unary_op(&UnaryOperator::Minus, inner),

            // +x = x if x is already simplified
            (UnaryOperator::Plus, operand) => Ok(operand.clone()),

            // Distribute minus over addition/subtraction
            // -(a + b) = -a - b
            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
            ) => {
                let neg_left = self.reduce_unary_op(&UnaryOperator::Minus, left)?;
                let neg_right = self.reduce_unary_op(&UnaryOperator::Minus, right)?;
                self.reduce_binary_op(&neg_left, &BinaryOperator::Subtract, &neg_right)
            }

            // -(a - b) = -a + b = b - a
            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Subtract,
                    right,
                },
            ) => {
                let neg_left = self.reduce_unary_op(&UnaryOperator::Minus, left)?;
                self.reduce_binary_op(right, &BinaryOperator::Add, &neg_left)
            }

            // -(a * b) = (-a) * b
            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Multiply,
                    right,
                },
            ) => {
                let neg_left = self.reduce_unary_op(&UnaryOperator::Minus, left)?;
                self.reduce_binary_op(&neg_left, &BinaryOperator::Multiply, right)
            }

            // Default case
            _ => Ok(Expression::UnaryOp {
                op: op.clone(),
                operand: Box::new(operand.clone()),
            }),
        }
    }

    fn expressions_equal(&self, left: &Expression, right: &Expression) -> bool {
        match (left, right) {
            (Expression::Number(a), Expression::Number(b)) => (a - b).abs() < f64::EPSILON,
            (Expression::Variable(a), Expression::Variable(b)) => a == b,
            (
                Expression::BinaryOp {
                    left: l1,
                    op: op1,
                    right: r1,
                },
                Expression::BinaryOp {
                    left: l2,
                    op: op2,
                    right: r2,
                },
            ) => op1 == op2 && self.expressions_equal(l1, l2) && self.expressions_equal(r1, r2),
            (
                Expression::UnaryOp {
                    op: op1,
                    operand: o1,
                },
                Expression::UnaryOp {
                    op: op2,
                    operand: o2,
                },
            ) => op1 == op2 && self.expressions_equal(o1, o2),
            _ => false,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}
