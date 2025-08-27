use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::complex::Complex;
use crate::error::EvaluationError;
use crate::matrix::Matrix;

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
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::Power => write!(f, "^"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Plus,
    Minus,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOperator::Plus => write!(f, "+"),
            UnaryOperator::Minus => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Complex(Complex),
    Matrix(Matrix),
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
}

impl Add for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a + b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) + b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a + Complex::new(b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a + b)),

            // Matrix operations
            (Expression::Matrix(_a), Expression::Matrix(_b)) => {
                // a.add(&b).map(Expression::Matrix).map_err(|_| {
                //     EvaluationError::InvalidOperation("Matrix addition failed".to_string())
                // })

                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix addition is not supported".to_string(),
                ));
            }

            // Algebraic simplifications
            (expr, rhs) if rhs.is_zero() => Ok(expr), // x + 0 = x
            (lhs, expr) if lhs.is_zero() => Ok(expr), // 0 + x = x
            (lhs, rhs) if lhs == rhs => {
                // x + x = 2*x
                Expression::Real(2.0) * lhs
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Add,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Sub for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, &rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a - b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) - *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a - Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a - *b)),

            // Matrix operations
            (Expression::Matrix(_a), Expression::Matrix(_b)) => {
                // a.sub(&b).map(Expression::Matrix).map_err(|_| {
                //     EvaluationError::InvalidOperation("Matrix subtraction failed".to_string())
                // })

                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix subtraction is not supported".to_string(),
                ));
            }

            // Algebraic simplifications
            (expr, rhs) if rhs.is_zero() => Ok(expr), // x - 0 = x
            (lhs, _) if lhs.is_zero() => -rhs.clone(), // 0 - x = -x
            (lhs, rhs) if lhs == *rhs => Ok(Expression::Real(0.0)), // x - x = 0

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Subtract,
                right: Box::new(rhs.clone()),
            }),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a * b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) * b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a * Complex::new(b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a * b)),

            // Matrix operations
            (Expression::Matrix(_a), Expression::Matrix(_b)) => {
                // a.mul(&b).map(Expression::Matrix).map_err(|_| {
                //     EvaluationError::InvalidOperation("Matrix multiplication failed".to_string())
                // })

                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix multiplication is not supported".to_string(),
                ));
            }

            // Scalar-matrix multiplication
            (Expression::Real(_scalar), Expression::Matrix(_matrix)) => {
                // matrix
                // .scalar_mul(Complex::new(scalar, 0.0))
                // .map(Expression::Matrix)
                // .map_err(|_| {
                //     EvaluationError::InvalidOperation(
                //         "Scalar-matrix multiplication failed".to_string(),
                //     )
                // }),
                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix multiplication is not supported".to_string(),
                ));
            }
            (Expression::Complex(_scalar), Expression::Matrix(_matrix)) => {
                // matrix
                //     .scalar_mul(scalar)
                //     .map(Expression::Matrix)
                //     .map_err(|_| {
                //         EvaluationError::InvalidOperation(
                //             "Scalar-matrix multiplication failed".to_string(),
                //         )
                //     }),
                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix multiplication is not supported".to_string(),
                ));
            }

            (Expression::Matrix(_matrix), Expression::Real(_scalar)) => {
                // matrix
                //     .scalar_mul(Complex::new(scalar, 0.0))
                //     .map(Expression::Matrix)
                //     .map_err(|_| {
                //         EvaluationError::InvalidOperation(
                //             "Scalar-matrix multiplication failed".to_string(),
                //         )
                //     }),
                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix multiplication is not supported".to_string(),
                ));
            }
            (Expression::Matrix(_matrix), Expression::Complex(_scalar)) => {
                //  matrix
                //     .scalar_mul(scalar)
                //     .map(Expression::Matrix)
                //     .map_err(|_| {
                //         EvaluationError::InvalidOperation(
                //             "Scalar-matrix multiplication failed".to_string(),
                //         )
                //     }),
                return Err(EvaluationError::UnsupportedOperation(
                    "Matrix multiplication is not supported".to_string(),
                ));
            }

            // Algebraic simplifications
            (_, expr) | (expr, _) if expr.is_zero() => Ok(Expression::Real(0.0)), // 0 * x = 0
            (expr, rhs) if rhs.is_one() => Ok(expr),                              // x * 1 = x
            (lhs, expr) if lhs.is_one() => Ok(expr),                              // 1 * x = x

            // Distributive property: a * (b + c) = a*b + a*c
            (
                a,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
            ) => {
                let ab = (a.clone() * *left)?;
                let ac = (a * *right)?;
                ab + ac
            }
            // (a + b) * c = a*c + b*c
            (
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
                c,
            ) => {
                let ac = (*left * c.clone())?;
                let bc = (*right * c)?;
                ac + bc
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Multiply,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Div for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a / b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) / b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a / Complex::new(b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a / b)),

            // Algebraic simplifications
            (expr, rhs) if rhs.is_one() => Ok(expr), // x / 1 = x
            (lhs, _) if lhs.is_zero() => Ok(Expression::Real(0.0)), // 0 / x = 0

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Divide,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Rem for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (self, rhs) {
            // Only support real numbers for modulo
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a % b)),
            (Expression::Complex(a), Expression::Complex(b)) if a.is_real() && b.is_real() => {
                Ok(Expression::Real(a.real % b.real))
            }

            // Algebraic simplifications
            (lhs, rhs) if lhs == rhs => Ok(Expression::Real(0.0)), // x % x = 0
            (Expression::Real(0.0), _) => Ok(Expression::Real(0.0)), // 0 % x = 0

            // Unsupported operations for complex numbers
            (Expression::Complex(_), _) | (_, Expression::Complex(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Modulo operation is only supported for real numbers".to_string(),
                ))
            }

            // Matrix modulo is not supported
            (Expression::Matrix(_), _) | (_, Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Modulo operation is not supported for matrices".to_string(),
                ))
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Modulo,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Neg for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn neg(self) -> Self::Output {
        match self {
            Expression::Real(n) => Ok(Expression::Real(-n)),
            Expression::Complex(c) => Ok(Expression::Complex(-c)),
            Expression::Matrix(_m) => {
                return Err(EvaluationError::InvalidOperation(
                    "Matrix negation not implemented".to_string(),
                ));
                // m.neg().map(Expression::Matrix).map_err(|_| {
                //     EvaluationError::InvalidOperation("Matrix negation failed".to_string())?
                // }),
            }
            // Double negative: -(-x) = x
            Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
            } => Ok(*operand),
            // Distribute minus over addition: -(a + b) = -a - b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Add,
                right,
            } => {
                let neg_left = (-*left)?;
                let neg_right = (-*right)?;
                neg_left - neg_right
            }
            // Distribute minus over subtraction: -(a - b) = b - a
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => *right - *left,
            // Distribute minus over multiplication: -(a * b) = (-a) * b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Multiply,
                right,
            } => {
                let neg_left = (-*left)?;
                neg_left * *right
            }
            expr => Ok(Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(expr),
            }),
        }
    }
}

pub trait Power<Rhs = Self> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl Power for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn pow(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a.powf(b))),
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a.pow(b))),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0).pow(b)))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a.pow(Complex::new(b, 0.0))))
            }

            // Algebraic simplifications
            (_, Expression::Real(0.0)) => Ok(Expression::Real(1.0)), // x^0 = 1
            (expr, Expression::Real(1.0)) => Ok(expr),               // x^1 = x
            (Expression::Real(1.0), _) => Ok(Expression::Real(1.0)), // 1^x = 1
            (Expression::Real(0.0), _) => Err(EvaluationError::InvalidOperation(
                "Zero to non-positive power is undefined".to_string(),
            )),

            // Matrix powers are not generally supported (except for square matrices and integer powers)
            (Expression::Matrix(_), _) | (_, Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Matrix exponentiation is not supported".to_string(),
                ))
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Power,
                right: Box::new(rhs),
            }),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Real(n) => write!(f, "{}", n)?,
            Expression::Complex(n) => {
                if n.is_real() {
                    if n.real.fract() == 0.0 {
                        write!(f, "{}", n.real as i64)?;
                    } else {
                        write!(f, "{}", n.real)?;
                    }
                } else if n.real == 0.0 {
                    if n.imag == 1.0 {
                        write!(f, "i")?;
                    } else if n.imag == -1.0 {
                        write!(f, "-i")?;
                    } else if n.imag.fract() == 0.0 {
                        write!(f, "{}i", n.imag as i64)?;
                    } else {
                        write!(f, "{}i", n.imag)?;
                    }
                } else {
                    if n.real.fract() == 0.0 {
                        write!(f, "{}", n.real as i64)?;
                    } else {
                        write!(f, "{}", n.real)?;
                    }

                    if n.imag > 0.0 {
                        write!(f, " + ")?;
                        if n.imag == 1.0 {
                            write!(f, "i")?;
                        } else if n.imag.fract() == 0.0 {
                            write!(f, "{}i", n.imag as i64)?;
                        } else {
                            write!(f, "{}i", n.imag)?;
                        }
                    } else {
                        write!(f, " - ")?;
                        let abs_imag = n.imag.abs();
                        if abs_imag == 1.0 {
                            write!(f, "i")?
                        } else if abs_imag.fract() == 0.0 {
                            write!(f, "{}i", abs_imag as i64)?;
                        } else {
                            write!(f, "{}i", abs_imag)?;
                        }
                    }
                }
            }
            Expression::Matrix(matrix) => {
                write!(f, "[")?;
                for r in 0..matrix.rows() {
                    if r > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for c in 0..matrix.cols() {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", matrix.get(r, c).unwrap())?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?;
            }
            Expression::Variable(name) => write!(f, "{}", name)?,
            Expression::FunctionCall { name, args } => {
                write!(f, "{}", name)?;
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")?;
            }
            Expression::BinaryOp { left, op, right } => {
                let needs_left_parens = match (op, left.as_ref()) {
                    (
                        BinaryOperator::Multiply | BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        },
                    ) => true,
                    (BinaryOperator::Power, Expression::BinaryOp { .. }) => true,
                    _ => false,
                };

                let needs_right_parens = match (op, right.as_ref()) {
                    (
                        BinaryOperator::Subtract,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        },
                    ) => true,
                    (
                        BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Multiply | BinaryOperator::Divide,
                            ..
                        },
                    ) => true,
                    (BinaryOperator::Power, Expression::BinaryOp { .. }) => true,
                    _ => false,
                };

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
            }
            Expression::UnaryOp { op, operand } => match operand.as_ref() {
                Expression::BinaryOp { .. } => write!(f, "{}({})", op, operand)?,
                _ => write!(f, "{}{}", op, operand)?,
            },
        }

        Ok(())
    }
}
