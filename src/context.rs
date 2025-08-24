use std::collections::HashMap;

use crate::ast::{BinaryOperator, Expression, UnaryOperator, Value};
use crate::error::EvaluationError;
use crate::Matrix;

pub struct Context {
    variables: HashMap<String, Value>,
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
        self.reduce_expression(expression, &HashMap::new())
    }

    pub fn evaluate_expression(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        self.reduce_expression(expr, param_scope)
    }

    pub fn reduce_expression(
        &self,
        expr: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match expr {
            Expression::Number(_) => Ok(expr.clone()),
            Expression::Matrix(matrix) => self.reduce_matrix(matrix, param_scope),
            Expression::Variable(name) => self.resolve_variable(name, param_scope),
            Expression::FunctionCall { name, args } => {
                self.reduce_function_call(name, args, param_scope)
            }
            Expression::BinaryOp { left, op, right } => {
                self.reduce_binary_operation(left, op, right, param_scope)
            }
            Expression::UnaryOp { op, operand } => {
                self.reduce_unary_operation(op, operand, param_scope)
            }
        }
    }

    fn resolve_variable(
        &self,
        name: &str,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        // Check parameter scope first
        if let Some(expr) = param_scope.get(name) {
            return Ok(expr.clone());
        }

        // Check context variables
        match self.get_variable(name) {
            Some(Value::Variable(expr)) => self.reduce_expression(expr, param_scope),
            Some(Value::Function { .. }) => Err(EvaluationError::NotAVariable(name.to_string())),
            None => Ok(Expression::Variable(name.to_string())), // Keep symbolic
        }
    }

    fn reduce_matrix(
        &self,
        matrix: &Matrix,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        let mut reduced_data = Vec::new();
        for row in matrix.iter() {
            let mut reduced_row = Vec::new();
            for elem in row {
                reduced_row.push(self.reduce_expression(elem, param_scope)?);
            }
            reduced_data.push(reduced_row);
        }
        let reduced_matrix = Matrix::new(reduced_data).map_err(EvaluationError::InvalidMatrix)?;
        Ok(Expression::Matrix(reduced_matrix))
    }

    fn reduce_function_call(
        &self,
        name: &str,
        args: &[Expression],
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self.get_variable(name) {
            Some(Value::Function { params, body }) => {
                self.execute_function(params, body, args, param_scope)
            }
            Some(Value::Variable(_)) => Err(EvaluationError::NotAFunction(name.to_string())),
            None => {
                // Reduce arguments and keep as function call
                let reduced_args = args
                    .iter()
                    .map(|arg| self.reduce_expression(arg, param_scope))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(Expression::FunctionCall {
                    name: name.to_string(),
                    args: reduced_args,
                })
            }
        }
    }

    fn execute_function(
        &self,
        params: &[String],
        body: &Expression,
        args: &[Expression],
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        if args.len() != params.len() {
            return Err(EvaluationError::WrongArgumentCount {
                expected: params.len(),
                got: args.len(),
            });
        }

        // Reduce arguments first
        let reduced_args = args
            .iter()
            .map(|arg| self.reduce_expression(arg, param_scope))
            .collect::<Result<Vec<_>, _>>()?;

        // Create function scope
        let function_scope = params
            .iter()
            .zip(reduced_args.iter())
            .map(|(param, arg)| (param.clone(), arg.clone()))
            .collect();

        self.reduce_expression(body, &function_scope)
    }

    fn reduce_binary_operation(
        &self,
        left: &Expression,
        op: &BinaryOperator,
        right: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        let left_reduced = self.reduce_expression(left, param_scope)?;
        let right_reduced = self.reduce_expression(right, param_scope)?;

        BinaryOperation::new(left_reduced, op.clone(), right_reduced).simplify()
    }

    fn reduce_unary_operation(
        &self,
        op: &UnaryOperator,
        operand: &Expression,
        param_scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        let operand_reduced = self.reduce_expression(operand, param_scope)?;
        UnaryOperation::new(op.clone(), operand_reduced).simplify()
    }
}

struct BinaryOperation {
    left: Expression,
    op: BinaryOperator,
    right: Expression,
}

impl BinaryOperation {
    fn new(left: Expression, op: BinaryOperator, right: Expression) -> Self {
        Self { left, op, right }
    }

    fn simplify(self) -> Result<Expression, EvaluationError> {
        // Handle numeric computations first
        if let (Expression::Number(a), Expression::Number(b)) = (&self.left, &self.right) {
            return self.compute_numeric(*a, *b);
        }

        // Apply basic algebraic simplifications
        match self.op {
            BinaryOperator::Add => self.simplify_addition(),
            BinaryOperator::Subtract => self.simplify_subtraction(),
            BinaryOperator::Multiply => self.simplify_multiplication(),
            BinaryOperator::Divide => self.simplify_division(),
            BinaryOperator::Power => self.simplify_power(),
        }
    }

    fn compute_numeric(&self, a: f64, b: f64) -> Result<Expression, EvaluationError> {
        let result = match self.op {
            BinaryOperator::Add => a + b,
            BinaryOperator::Subtract => a - b,
            BinaryOperator::Multiply => a * b,
            BinaryOperator::Divide => {
                if b == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                a / b
            }
            BinaryOperator::Power => a.powf(b),
        };
        Ok(Expression::Number(result))
    }

    fn simplify_addition(self) -> Result<Expression, EvaluationError> {
        match (&self.left, &self.right) {
            // 0 + x = x
            (Expression::Number(n), right) if *n == 0.0 => Ok(right.clone()),
            // x + 0 = x
            (left, Expression::Number(n)) if *n == 0.0 => Ok(left.clone()),
            // x + x = 2*x
            (left, right) if ExpressionComparer::equal(left, right) => Ok(Expression::BinaryOp {
                left: Box::new(Expression::Number(2.0)),
                op: BinaryOperator::Multiply,
                right: Box::new(left.clone()),
            }),
            _ => Ok(self.to_expression()),
        }
    }

    fn simplify_subtraction(self) -> Result<Expression, EvaluationError> {
        match (&self.left, &self.right) {
            // x - 0 = x
            (left, Expression::Number(n)) if *n == 0.0 => Ok(left.clone()),
            // 0 - x = -x
            (Expression::Number(n), right) if *n == 0.0 => {
                UnaryOperation::new(UnaryOperator::Minus, right.clone()).simplify()
            }
            // x - x = 0
            (left, right) if ExpressionComparer::equal(left, right) => Ok(Expression::Number(0.0)),
            _ => Ok(self.to_expression()),
        }
    }

    fn simplify_multiplication(self) -> Result<Expression, EvaluationError> {
        match (&self.left, &self.right) {
            // 0 * x = 0
            (Expression::Number(n), _) | (_, Expression::Number(n)) if *n == 0.0 => {
                Ok(Expression::Number(0.0))
            }
            // 1 * x = x
            (Expression::Number(n), right) if *n == 1.0 => Ok(right.clone()),
            // x * 1 = x
            (left, Expression::Number(n)) if *n == 1.0 => Ok(left.clone()),
            // Distribute multiplication: a * (b + c) = a*b + a*c
            (
                a,
                Expression::BinaryOp {
                    left: b,
                    op: BinaryOperator::Add,
                    right: c,
                },
            ) => {
                let ab =
                    BinaryOperation::new(a.clone(), BinaryOperator::Multiply, b.as_ref().clone())
                        .simplify()?;
                let ac =
                    BinaryOperation::new(a.clone(), BinaryOperator::Multiply, c.as_ref().clone())
                        .simplify()?;
                BinaryOperation::new(ab, BinaryOperator::Add, ac).simplify()
            }
            // (a + b) * c = a*c + b*c
            (
                Expression::BinaryOp {
                    left: a,
                    op: BinaryOperator::Add,
                    right: b,
                },
                c,
            ) => {
                let ac =
                    BinaryOperation::new(a.as_ref().clone(), BinaryOperator::Multiply, c.clone())
                        .simplify()?;
                let bc =
                    BinaryOperation::new(b.as_ref().clone(), BinaryOperator::Multiply, c.clone())
                        .simplify()?;
                BinaryOperation::new(ac, BinaryOperator::Add, bc).simplify()
            }
            _ => Ok(self.to_expression()),
        }
    }

    fn simplify_division(self) -> Result<Expression, EvaluationError> {
        match (&self.left, &self.right) {
            // x / 1 = x
            (left, Expression::Number(n)) if *n == 1.0 => Ok(left.clone()),
            // 0 / x = 0 (assuming x != 0)
            (Expression::Number(n), right) if *n == 0.0 => {
                if let Expression::Number(0.0) = right {
                    Err(EvaluationError::DivisionByZero)
                } else {
                    Ok(Expression::Number(0.0))
                }
            }
            _ => Ok(self.to_expression()),
        }
    }

    fn simplify_power(self) -> Result<Expression, EvaluationError> {
        match (&self.left, &self.right) {
            // x^0 = 1
            (_, Expression::Number(n)) if *n == 0.0 => Ok(Expression::Number(1.0)),
            // x^1 = x
            (left, Expression::Number(n)) if *n == 1.0 => Ok(left.clone()),
            // 1^x = 1
            (Expression::Number(n), _) if *n == 1.0 => Ok(Expression::Number(1.0)),
            // 0^x
            (Expression::Number(n), right) if *n == 0.0 => match right {
                Expression::Number(exp) if *exp > 0.0 => Ok(Expression::Number(0.0)),
                Expression::Number(exp) if *exp <= 0.0 => Err(EvaluationError::UndefinedOperation),
                _ => Ok(self.to_expression()),
            },
            _ => Ok(self.to_expression()),
        }
    }

    fn to_expression(self) -> Expression {
        Expression::BinaryOp {
            left: Box::new(self.left),
            op: self.op,
            right: Box::new(self.right),
        }
    }
}

struct UnaryOperation {
    op: UnaryOperator,
    operand: Expression,
}

impl UnaryOperation {
    fn new(op: UnaryOperator, operand: Expression) -> Self {
        Self { op, operand }
    }

    fn simplify(self) -> Result<Expression, EvaluationError> {
        match (&self.op, &self.operand) {
            // Direct computation for numbers
            (UnaryOperator::Plus, Expression::Number(n)) => Ok(Expression::Number(*n)),
            (UnaryOperator::Minus, Expression::Number(n)) => Ok(Expression::Number(-n)),

            // +x = x
            (UnaryOperator::Plus, operand) => Ok(operand.clone()),

            // Double negative: -(-x) = x
            (
                UnaryOperator::Minus,
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: inner,
                },
            ) => Ok(inner.as_ref().clone()),

            // Distribute minus over binary operations
            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
            ) => {
                let neg_left =
                    UnaryOperation::new(UnaryOperator::Minus, left.as_ref().clone()).simplify()?;
                let neg_right =
                    UnaryOperation::new(UnaryOperator::Minus, right.as_ref().clone()).simplify()?;
                BinaryOperation::new(neg_left, BinaryOperator::Subtract, neg_right).simplify()
            }

            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Subtract,
                    right,
                },
            ) => {
                let neg_left =
                    UnaryOperation::new(UnaryOperator::Minus, left.as_ref().clone()).simplify()?;
                BinaryOperation::new(right.as_ref().clone(), BinaryOperator::Add, neg_left)
                    .simplify()
            }

            (
                UnaryOperator::Minus,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Multiply,
                    right,
                },
            ) => {
                let neg_left =
                    UnaryOperation::new(UnaryOperator::Minus, left.as_ref().clone()).simplify()?;
                BinaryOperation::new(neg_left, BinaryOperator::Multiply, right.as_ref().clone())
                    .simplify()
            }

            // Default case
            _ => Ok(Expression::UnaryOp {
                op: self.op,
                operand: Box::new(self.operand),
            }),
        }
    }
}

struct ExpressionComparer;

impl ExpressionComparer {
    fn equal(left: &Expression, right: &Expression) -> bool {
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
            ) => op1 == op2 && Self::equal(l1, l2) && Self::equal(r1, r2),
            (
                Expression::UnaryOp {
                    op: op1,
                    operand: o1,
                },
                Expression::UnaryOp {
                    op: op2,
                    operand: o2,
                },
            ) => op1 == op2 && Self::equal(o1, o2),
            _ => false,
        }
    }
}
