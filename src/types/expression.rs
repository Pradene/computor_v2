use std::collections::HashMap;
use std::fmt;

use crate::error::EvaluationError;
use crate::types::complex::Complex;
use crate::types::matrix::Matrix;
use crate::types::vector::Vector;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Modulo,
    Multiply,
    Divide,
    Power,
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            BinaryOperator::Add => "+",
            BinaryOperator::Subtract => "-",
            BinaryOperator::Multiply => "*",
            BinaryOperator::Divide => "/",
            BinaryOperator::Modulo => "%",
            BinaryOperator::Power => "^",
        };
        write!(f, "{}", symbol)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Plus,
    Minus,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol = match self {
            UnaryOperator::Plus => "+",
            UnaryOperator::Minus => "-",
        };
        write!(f, "{}", symbol)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Complex(Complex),
    Matrix(Matrix),
    Vector(Vector),
    Variable(String),
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Real(n) => write!(f, "{}", n),
            Expression::Complex(n) => write!(f, "{}", n),
            Expression::Matrix(matrix) => write!(f, "{}", matrix),
            Expression::Vector(vector) => write!(f, "{}", vector),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expression::BinaryOp { left, op, right } => {
                let needs_left_parens = matches!(
                    (op, left.as_ref()),
                    (
                        BinaryOperator::Multiply | BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        }
                    ) | (BinaryOperator::Power, Expression::BinaryOp { .. })
                );

                let needs_right_parens = matches!(
                    (op, right.as_ref()),
                    (
                        BinaryOperator::Subtract,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        }
                    ) | (
                        BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Multiply | BinaryOperator::Divide,
                            ..
                        }
                    ) | (BinaryOperator::Power, Expression::BinaryOp { .. })
                );

                if needs_left_parens {
                    write!(f, "({})", left)?;
                } else {
                    write!(f, "{}", left)?;
                }

                write!(f, " {} ", op)?;

                if needs_right_parens {
                    write!(f, "({})", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
                Ok(())
            }
            Expression::UnaryOp { op, operand } => {
                if matches!(operand.as_ref(), Expression::BinaryOp { .. }) {
                    write!(f, "{}({})", op, operand)
                } else {
                    write!(f, "{}{}", op, operand)
                }
            }
        }
    }
}

impl Expression {
    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => *n == 0.0,
            Expression::Complex(c) => c.is_zero(),
            _ => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Expression::Real(n) => *n == 1.0,
            Expression::Complex(c) => c.is_one(),
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Expression::Real(_) | Expression::Complex(_))
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Real(_) => true,
            Expression::Complex(c) => c.is_real(),
            _ => false,
        }
    }

    pub fn is_matrix(&self) -> bool {
        matches!(self, Expression::Matrix(_))
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Expression::Variable(_))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Expression::FunctionCall { .. })
    }
}

impl Expression {
    pub fn reduce(&self) -> Result<Expression, EvaluationError> {
        let mut current = self.clone();
        const MAX_ITERATIONS: usize = 64;

        for _ in 0..MAX_ITERATIONS {
            let collected = current.collect_terms()?;
            if collected == current {
                break;
            }
            current = collected;
        }

        Ok(current)
    }

    fn collect_terms(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::BinaryOp {
                op: BinaryOperator::Add,
                ..
            } => {
                let mut terms = HashMap::new();
                self.collect_addition_terms(&mut terms, 1.0)?;
                self.rebuild_from_terms(terms)
            }
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => {
                // Convert a - b to a + (-b) for easier term collection
                let neg_right = Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: right.clone(),
                };
                let as_addition = Expression::BinaryOp {
                    left: left.clone(),
                    op: BinaryOperator::Add,
                    right: Box::new(neg_right),
                };
                as_addition.collect_terms()
            }
            Expression::BinaryOp { left, op, right } => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left_collected == **left && right_collected == **right {
                    Ok(self.clone())
                } else {
                    Ok(Expression::BinaryOp {
                        left: Box::new(left_collected),
                        op: op.clone(),
                        right: Box::new(right_collected),
                    })
                }
            }
            Expression::UnaryOp { op, operand } => {
                let operand_collected = operand.collect_terms()?;

                match op {
                    UnaryOperator::Plus => Ok(operand_collected),
                    UnaryOperator::Minus => {
                        if let Expression::UnaryOp {
                            op: UnaryOperator::Minus,
                            operand: inner,
                        } = &operand_collected
                        {
                            Ok(*inner.clone())
                        } else if operand_collected == **operand {
                            Ok(self.clone())
                        } else {
                            Ok(Expression::UnaryOp {
                                op: op.clone(),
                                operand: Box::new(operand_collected),
                            })
                        }
                    }
                }
            }
            _ => Ok(self.clone()),
        }
    }

    fn collect_addition_terms(
        &self,
        terms: &mut HashMap<String, f64>,
        coeff: f64,
    ) -> Result<(), EvaluationError> {
        match self {
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Add,
                right,
            } => {
                left.collect_addition_terms(terms, coeff)?;
                right.collect_addition_terms(terms, coeff)?;
            }
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => {
                left.collect_addition_terms(terms, coeff)?;
                right.collect_addition_terms(terms, -coeff)?;
            }
            Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
            } => {
                operand.collect_addition_terms(terms, -coeff)?;
            }
            Expression::UnaryOp {
                op: UnaryOperator::Plus,
                operand,
            } => {
                operand.collect_addition_terms(terms, coeff)?;
            }
            Expression::Real(n) => {
                *terms.entry("__constant__".to_string()).or_insert(0.0) += coeff * n;
            }
            Expression::Complex(c) if c.is_real() => {
                *terms.entry("__constant__".to_string()).or_insert(0.0) += coeff * c.real;
            }
            Expression::Variable(name) => {
                *terms.entry(name.clone()).or_insert(0.0) += coeff;
            }
            Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                ..
            } => {
                let (extracted_coeff, variables) = self.extract_multiplication_parts();
                let total_coeff = coeff * extracted_coeff;

                let key = if variables.is_empty() {
                    "__constant__".to_string()
                } else if variables.len() == 1 {
                    variables[0].clone()
                } else {
                    let mut sorted_vars = variables;
                    sorted_vars.sort();
                    sorted_vars.join(" * ")
                };

                *terms.entry(key).or_insert(0.0) += total_coeff;
            }
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Power,
                right,
            } => {
                left.collect_addition_terms(terms, coeff)?;
                right.collect_addition_terms(terms, coeff)?;
            }
            _ => {
                let key = format!("{}", self);
                *terms.entry(key).or_insert(0.0) += coeff;
            }
        }
        Ok(())
    }

    fn extract_multiplication_parts(&self) -> (f64, Vec<String>) {
        let mut coefficient = 1.0;
        let mut variables = Vec::new();
        self.collect_multiplication_parts(&mut coefficient, &mut variables);
        (coefficient, variables)
    }

    fn collect_multiplication_parts(&self, coefficient: &mut f64, variables: &mut Vec<String>) {
        match self {
            Expression::Real(n) => *coefficient *= n,
            Expression::Variable(name) => variables.push(name.clone()),
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Multiply,
                right,
            } => {
                left.collect_multiplication_parts(coefficient, variables);
                right.collect_multiplication_parts(coefficient, variables);
            }
            Expression::Complex(c) if c.is_real() => *coefficient *= c.real,
            _ => {}
        }
    }

    fn rebuild_from_terms(
        &self,
        terms: HashMap<String, f64>,
    ) -> Result<Expression, EvaluationError> {
        let mut result_terms = Vec::new();

        for (term_str, coeff) in terms {
            if coeff.abs() < f64::EPSILON {
                continue;
            }

            let term_expr = if term_str == "__constant__" {
                if coeff < 0.0 {
                    Expression::UnaryOp {
                        op: UnaryOperator::Minus,
                        operand: Box::new(Expression::Real(-coeff)),
                    }
                } else {
                    Expression::Real(coeff)
                }
            } else if (coeff - 1.0).abs() < f64::EPSILON {
                Expression::Variable(term_str)
            } else if (coeff + 1.0).abs() < f64::EPSILON {
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(Expression::Variable(term_str)),
                }
            } else if coeff < 0.0 {
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(Expression::BinaryOp {
                        left: Box::new(Expression::Real(-coeff)),
                        op: BinaryOperator::Multiply,
                        right: Box::new(Expression::Variable(term_str)),
                    }),
                }
            } else {
                Expression::BinaryOp {
                    left: Box::new(Expression::Real(coeff)),
                    op: BinaryOperator::Multiply,
                    right: Box::new(Expression::Variable(term_str)),
                }
            };

            result_terms.push(term_expr);
        }

        if result_terms.is_empty() {
            return Ok(Expression::Real(0.0));
        }

        let mut result = result_terms[0].clone();
        for term in result_terms.iter().skip(1) {
            match term {
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand,
                } => {
                    result = Expression::BinaryOp {
                        left: Box::new(result),
                        op: BinaryOperator::Subtract,
                        right: operand.clone(),
                    };
                }
                _ => {
                    result = Expression::BinaryOp {
                        left: Box::new(result),
                        op: BinaryOperator::Add,
                        right: Box::new(term.clone()),
                    };
                }
            }
        }

        Ok(result)
    }
}

impl Expression {
    pub fn pow(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a.powf(*b))),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(*a, 0.0).pow(*b)))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a.pow(Complex::new(*b, 0.0))))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a.pow(*b))),
            (Expression::Matrix(a), Expression::Real(b)) => {
                Ok(Expression::Matrix(a.pow(*b as i32)?))
            }
            (Expression::Matrix(_), _) | (_, Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Matrix exponentiation is not supported".to_string(),
                ))
            }
            _ => Ok(Expression::BinaryOp {
                left: Box::new(self),
                op: BinaryOperator::Power,
                right: Box::new(rhs),
            }),
        }
    }

    pub fn add(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a + b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(*a, 0.0) + *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(*a + Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(*a + *b)),
            (Expression::Matrix(a), Expression::Matrix(b)) => (a.clone() + b.clone())
                .map(Expression::Matrix)
                .map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix addition failed".to_string())
                }),
            (Expression::Matrix(_), Expression::Real(_) | Expression::Complex(_))
            | (Expression::Real(_) | Expression::Complex(_), Expression::Matrix(_)) => Err(
                EvaluationError::InvalidOperation("Cannot add scalar and matrix".to_string()),
            ),
            _ => {
                // Algebraic simplifications
                if rhs.is_zero() && !self.is_matrix() {
                    return Ok(self);
                }
                if self.is_zero() && !rhs.is_matrix() {
                    return Ok(rhs);
                }
                if self == rhs && (self.is_numeric() || self.is_variable()) {
                    return Expression::Real(2.0).mul(self);
                }

                Ok(Expression::BinaryOp {
                    left: Box::new(self),
                    op: BinaryOperator::Add,
                    right: Box::new(rhs),
                })
            }
        }
    }

    pub fn sub(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a - b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(*a, 0.0) - *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(*a - Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(*a - *b)),
            (Expression::Matrix(a), Expression::Matrix(b)) => (a.clone() - b.clone())
                .map(Expression::Matrix)
                .map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix subtraction failed".to_string())
                }),
            (Expression::Matrix(_), Expression::Real(_) | Expression::Complex(_))
            | (Expression::Real(_) | Expression::Complex(_), Expression::Matrix(_)) => Err(
                EvaluationError::InvalidOperation("Cannot subtract scalar and matrix".to_string()),
            ),
            _ => Ok(Expression::BinaryOp {
                left: Box::new(self),
                op: BinaryOperator::Subtract,
                right: Box::new(rhs),
            }),
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a * b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(*a, 0.0) * *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(*a * Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(*a * *b)),
            (Expression::Matrix(a), Expression::Matrix(b)) => (a.clone() * b.clone())
                .map(Expression::Matrix)
                .map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix multiplication failed".to_string())
                }),
            (Expression::Real(s), Expression::Matrix(m))
            | (Expression::Matrix(m), Expression::Real(s)) => {
                (m.clone() * *s).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }
            (Expression::Complex(s), Expression::Matrix(m))
            | (Expression::Matrix(m), Expression::Complex(s)) => {
                (m.clone() * *s).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }
            _ => {
                // Algebraic simplifications
                if self.is_zero() || rhs.is_zero() {
                    return Ok(Expression::Real(0.0));
                }
                if self.is_one() {
                    return Ok(rhs);
                }
                if rhs.is_one() {
                    return Ok(self);
                }

                // Distributive property
                if let Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                } = &rhs
                {
                    let ab = (self.clone().mul(*left.clone()))?;
                    let ac = (self.clone().mul(*right.clone()))?;
                    return ab.add(ac);
                }

                if let Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                } = &rhs
                {
                    let ab = (self.clone().mul(*left.clone()))?;
                    let ac = (self.clone().mul(*right.clone()))?;
                    return ab.add(ac);
                }

                Ok(Expression::BinaryOp {
                    left: Box::new(self),
                    op: BinaryOperator::Multiply,
                    right: Box::new(rhs),
                })
            }
        }
    }

    pub fn div(self, rhs: Self) -> Result<Expression, EvaluationError> {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (&self, &rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a / b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(*a, 0.0) / *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(*a / Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(*a / *b)),
            (Expression::Matrix(m), Expression::Real(s)) => {
                (m.clone() / *s).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix division failed".to_string())
                })
            }
            (Expression::Matrix(m), Expression::Complex(s)) => {
                (m.clone() / *s).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix division failed".to_string())
                })
            }
            (Expression::Real(_) | Expression::Complex(_), Expression::Matrix(_)) => Err(
                EvaluationError::InvalidOperation("Cannot divide scalar by matrix".to_string()),
            ),
            (Expression::Matrix(_), Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Matrix division is not implemented".to_string(),
                ))
            }
            _ => {
                // Algebraic simplifications
                if rhs.is_one() {
                    return Ok(self);
                }
                if self.is_zero() {
                    return Ok(Expression::Real(0.0));
                }

                Ok(Expression::BinaryOp {
                    left: Box::new(self),
                    op: BinaryOperator::Divide,
                    right: Box::new(rhs),
                })
            }
        }
    }

    pub fn rem(self, rhs: Self) -> Result<Expression, EvaluationError> {
        if rhs.is_real() && rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a % b)),
            (Expression::Complex(a), Expression::Complex(b)) if a.is_real() && b.is_real() => {
                Ok(Expression::Real(a.real % b.real))
            }
            // All other cases are invalid
            _ => Err(EvaluationError::UnsupportedOperation(
                "Modulo operation is only supported for real numbers".to_string(),
            )),
        }
    }

    pub fn neg(self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(-n)),
            Expression::Complex(c) => Ok(Expression::Complex(-c)),
            Expression::Matrix(m) => (-m).map(Expression::Matrix).map_err(|_| {
                EvaluationError::InvalidOperation("Matrix negation failed".to_string())
            }),
            // Double negative: -(-x) = x
            Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
            } => Ok(*operand),
            // Double negative: - +x = -x
            Expression::UnaryOp {
                op: UnaryOperator::Plus,
                operand,
            } => (*operand).neg(),
            // Distribute minus over binary operations
            Expression::BinaryOp { left, op, right } => match op {
                BinaryOperator::Add => {
                    let neg_left = (*left).neg()?;
                    let neg_right = (*right).neg()?;
                    neg_left.sub(neg_right)
                }
                BinaryOperator::Subtract => (*right).sub(*left),
                BinaryOperator::Multiply => {
                    let neg_left = (*left).neg()?;
                    neg_left.mul(*right)
                }
                _ => Ok(Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(Expression::BinaryOp { left, op, right }),
                }),
            },
            _ => Ok(Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(self),
            }),
        }
    }

    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.sqrt())),
            Expression::Complex(n) => Ok(Expression::Complex(n.sqrt())),
            _ => Err(EvaluationError::InvalidOperation(
                "Sqrt is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Complex(n) => Ok(Expression::Real(n.abs())),
            _ => Err(EvaluationError::InvalidOperation(
                "Abs is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.exp())),
            Expression::Complex(n) => Ok(Expression::Complex(n.exp())),
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
    }
}
