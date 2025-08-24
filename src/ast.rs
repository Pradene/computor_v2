use std::fmt;

use crate::complex::Complex;
use crate::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Variable(Expression),
    Function {
        params: Vec<String>,
        body: Expression,
    },
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Variable(expr) => write!(f, "{}", expr),
            Value::Function { body, .. } => {
                write!(f, "{}", body)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Number(Complex),
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

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Number(n) => {
                if n.is_real() {
                    if n.real.fract() == 0.0 {
                        write!(f, "{}", n.real as i64)?
                    } else {
                        write!(f, "{}", n.real)?
                    }
                } else if n.real == 0.0 {
                    if n.imag == 1.0 {
                        write!(f, "i")?
                    } else if n.imag == -1.0 {
                        write!(f, "-i")?
                    } else if n.imag.fract() == 0.0 {
                        write!(f, "{}i", n.imag as i64)?
                    } else {
                        write!(f, "{}i", n.imag)?
                    }
                } else {
                    if n.real.fract() == 0.0 {
                        write!(f, "{}", n.real as i64)?
                    } else {
                        write!(f, "{}", n.real)?
                    }

                    if n.imag > 0.0 {
                        write!(f, " + ")?;
                        if n.imag == 1.0 {
                            write!(f, "i")?
                        } else if n.imag.fract() == 0.0 {
                            write!(f, "{}i", n.imag as i64)?
                        } else {
                            write!(f, "{}i", n.imag)?
                        }
                    } else {
                        write!(f, " - ")?;
                        let abs_imag = n.imag.abs();
                        if abs_imag == 1.0 {
                            write!(f, "i")?
                        } else if abs_imag.fract() == 0.0 {
                            write!(f, "{}i", abs_imag as i64)?
                        } else {
                            write!(f, "{}i", abs_imag)?
                        }
                    }
                }
            }
            Expression::Matrix(matrix) => {
                write!(f, "[")?;
                for (i, row) in matrix.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "[")?;
                    for (j, expr) in row.iter().enumerate() {
                        if j > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", expr)?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?
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
                write!(f, ")")?
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
                    write!(f, "({})", right)?
                } else {
                    write!(f, "{}", right)?
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
