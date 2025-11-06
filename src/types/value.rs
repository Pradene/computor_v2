use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::error::EvaluationError;
use crate::types::{complex::Complex, matrix::Matrix, vector::Vector};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Real(f64),
    Complex(Complex),
    Vector(Vector),
    Matrix(Matrix),
}

impl Neg for Value {
    type Output = Result<Self, EvaluationError>;

    fn neg(self) -> Self::Output {
        match self {
            Value::Real(n) => Ok(Value::Real(-n)),
            Value::Complex(c) => Ok(Value::Complex(-c)),
            Value::Vector(v) => Ok(Value::Vector((-v).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector negation failed: {}", e))
            })?)),
            Value::Matrix(m) => Ok(Value::Matrix((-m).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix negation failed: {}", e))
            })?)),
        }
    }
}

impl Add for Value {
    type Output = Result<Self, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a + b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) + b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a + Complex::new(b, 0.0))),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector addition failed: {}", e))
            })?)),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix addition failed: {}", e))
            })?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} + {}",
                left, right
            ))),
        }
    }
}

impl Sub for Value {
    type Output = Result<Self, EvaluationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a - b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) - b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a - Complex::new(b, 0.0))),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector((a - b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector subtraction failed: {}", e))
            })?)),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a - b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix subtraction failed: {}", e))
            })?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} - {}",
                left, right
            ))),
        }
    }
}

impl Mul for Value {
    type Output = Result<Self, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a * b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) * b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a * Complex::new(b, 0.0))),

            // Scalar * Vector
            (Value::Real(s), Value::Vector(v)) | (Value::Vector(v), Value::Real(s)) => {
                Ok(Value::Vector((v * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Vector multiplication failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Complex(c), Value::Vector(v)) | (Value::Vector(v), Value::Complex(c)) => {
                Ok(Value::Vector((v * c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Complex-Vector multiplication failed: {}",
                        e
                    ))
                })?))
            }

            // Scalar * Matrix
            (Value::Real(s), Value::Matrix(m)) | (Value::Matrix(m), Value::Real(s)) => {
                Ok(Value::Matrix((m * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Matrix multiplication failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Complex(c), Value::Matrix(m)) | (Value::Matrix(m), Value::Complex(c)) => {
                Ok(Value::Matrix((m * c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Complex-Matrix multiplication failed: {}",
                        e
                    ))
                })?))
            }

            // Matrix * Matrix
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a * b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix multiplication failed: {}", e))
            })?)),

            // Matrix * Vector
            (Value::Matrix(a), Value::Vector(b)) => Ok(Value::Vector((a * b).map_err(|e| {
                EvaluationError::InvalidOperation(format!(
                    "Matrix-Vector multiplication failed: {}",
                    e
                ))
            })?)),

            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} * {}",
                left, right
            ))),
        }
    }
}

impl Div for Value {
    type Output = Result<Self, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Real(a / b))
            }
            (Value::Complex(a), Value::Complex(b)) => {
                if b.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(a / b))
            }
            (Value::Real(a), Value::Complex(b)) => {
                if b.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(Complex::new(a, 0.0) / b))
            }
            (Value::Complex(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(a / Complex::new(b, 0.0)))
            }
            (Value::Vector(v), Value::Real(s)) => {
                if s == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Vector((v / s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Vector-Scalar division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Vector(v), Value::Complex(c)) => {
                if c.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Vector((v / c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Vector-Complex division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Matrix(m), Value::Real(s)) => {
                if s == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Matrix((m / s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Matrix-Scalar division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Matrix(m), Value::Complex(c)) => {
                if c.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Matrix((m / c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Matrix-Complex division failed: {}",
                        e
                    ))
                })?))
            }
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} / {}",
                left, right
            ))),
        }
    }
}

impl Rem for Value {
    type Output = Result<Self, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::InvalidOperation(
                        "Modulo by zero".to_string(),
                    ));
                }
                Ok(Value::Real(a % b))
            }
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} % {}",
                left, right
            ))),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Real(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Complex(c) => write!(f, "{}", c),
            Value::Vector(v) => write!(f, "{}", v),
            Value::Matrix(m) => write!(f, "{}", m),
        }
    }
}

impl Value {
    pub fn pow(self, rhs: Self) -> Result<Self, EvaluationError> {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a.powf(b))),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a.pow(Complex::new(b, 0.0)))),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a.pow(b))),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0).pow(b))),
            (Value::Matrix(a), Value::Real(b)) => Ok(Value::Matrix(a.pow(b as i32)?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} ^ {}",
                left, right
            ))),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Real(n) => n.abs() < f64::EPSILON,
            Value::Complex(c) => c.is_zero(),
            _ => false,
        }
    }
}
