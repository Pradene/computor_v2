use std::{
    collections::HashMap,
    fmt,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    context::{BuiltinFunction, Context, Symbol, Variable},
    error::EvaluationError,
    EPSILON,
};

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub args: Vec<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Complex(f64, f64),
    Vector(Vec<Expression>),
    Matrix(Vec<Expression>, usize, usize),
    Variable(String),
    FunctionCall(FunctionCall),
    Paren(Box<Expression>),
    Neg(Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    MatMul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Mod(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Real(n) => {
                write!(f, "{}", n)?;
            }
            Expression::Complex(real, imag) => {
                if real.abs() < EPSILON {
                    write!(f, "{}i", imag)?;
                } else if imag.abs() < EPSILON {
                    write!(f, "{}", real)?;
                } else if imag.abs() >= EPSILON {
                    write!(f, "{} + {}i", real, imag)?;
                } else {
                    write!(f, "{} - {}i", real, -imag)?;
                }
            }
            Expression::Vector(data) => {
                write!(f, "[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")?;
            }
            Expression::Matrix(data, rows, cols) => {
                write!(f, "[")?;
                for r in 0..*rows {
                    if r > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "[")?;
                    for c in 0..*cols {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", &data[r * cols + c])?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?;
            }
            Expression::Variable(name) => write!(f, "{}", name)?,
            Expression::FunctionCall(fc) => {
                write!(f, "{}(", fc.name)?;
                for (i, arg) in fc.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")?;
            }
            Expression::Paren(inner) => write!(f, "( {} )", inner)?,
            Expression::Neg(operand) => write!(f, "-{}", operand)?,
            Expression::Add(left, right) => write!(f, "{} + {}", left, right)?,
            Expression::Sub(left, right) => write!(f, "{} - {}", left, right)?,
            Expression::Mul(left, right) => write!(f, "{} * {}", left, right)?,
            Expression::MatMul(left, right) => write!(f, "{} ** {}", left, right)?,
            Expression::Div(left, right) => write!(f, "{} / {}", left, right)?,
            Expression::Mod(left, right) => write!(f, "{} % {}", left, right)?,
            Expression::Pow(left, right) => write!(f, "{} ^ {}", left, right)?,
        };

        Ok(())
    }
}

impl Expression {
    pub fn is_value(&self) -> bool {
        matches!(
            self,
            Expression::Real(_)
                | Expression::Complex(_, _)
                | Expression::Vector(_)
                | Expression::Matrix(_, _, _)
        )
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => n.abs() < EPSILON,
            Expression::Complex(r, i) => r.abs() < EPSILON && i.abs() < EPSILON,
            _ => false,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Real(_) => true,
            Expression::Complex(_, i) => i.abs() < EPSILON,
            _ => false,
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Expression::Variable(_))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Expression::FunctionCall(_))
    }
}

// Complex number helpers
impl Expression {
    fn complex_add(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r + b_r, a_i + b_i)
    }

    fn complex_sub(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r - b_r, a_i - b_i)
    }

    fn complex_mul(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r)
    }

    fn complex_div(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Result<Expression, EvaluationError> {
        if b_r.abs() < EPSILON && b_i.abs() < EPSILON {
            return Err(EvaluationError::DivisionByZero);
        }

        let denominator = b_r * b_r + b_i * b_i;
        let real = (a_r * b_r + a_i * b_i) / denominator;
        let imag = (a_i * b_r - a_r * b_i) / denominator;

        Ok(Expression::Complex(real, imag))
    }

    fn complex_pow(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        let a_is_zero = a_r.abs() < EPSILON && a_i.abs() < EPSILON;
        let b_is_zero = b_r.abs() < EPSILON && b_i.abs() < EPSILON;
        let b_is_one = (b_r - 1.0).abs() < EPSILON && b_i.abs() < EPSILON;

        if a_is_zero {
            if b_r > 0.0 || (b_i.abs() >= EPSILON && b_r < EPSILON) {
                return Expression::Complex(0.0, 0.0);
            } else if b_r < 0.0 {
                return Expression::Complex(f64::INFINITY, f64::INFINITY);
            } else {
                return Expression::Complex(1.0, 0.0);
            }
        }

        if b_is_zero {
            return Expression::Complex(1.0, 0.0);
        }

        if b_is_one {
            return Expression::Complex(a_r, a_i);
        }

        // ln(a) = ln(|a|) + i*arg(a)
        let magnitude = (a_r * a_r + a_i * a_i).sqrt();
        let phase = a_i.atan2(a_r);
        let ln_r = magnitude.ln();
        let ln_i = phase;

        // b * ln(a)
        let w_ln_r = b_r * ln_r - b_i * ln_i;
        let w_ln_i = b_r * ln_i + b_i * ln_r;

        // exp(b * ln(a))
        let exp_real = w_ln_r.exp();

        let real = exp_real * w_ln_i.cos();
        let imag = exp_real * w_ln_i.sin();

        let real = if real.abs() < EPSILON { 0.0 } else { real };
        let imag = if imag.abs() < EPSILON { 0.0 } else { imag };

        Expression::Complex(real, imag)
    }

    fn complex_sqrt(r: f64, i: f64) -> Expression {
        if r.abs() < EPSILON && i.abs() < EPSILON {
            return Expression::Complex(0.0, 0.0);
        }

        let magnitude = (r * r + i * i).sqrt();
        let phase = i.atan2(r);

        let sqrt_r = magnitude.sqrt();
        let half_theta = phase / 2.0;

        Expression::Complex(sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin())
    }

    fn complex_abs(r: f64, i: f64) -> f64 {
        (r * r + i * i).sqrt()
    }

    fn complex_exp(r: f64, i: f64) -> Expression {
        let exp_real = r.exp();
        Expression::Complex(exp_real * i.cos(), exp_real * i.sin())
    }
}

impl Add for Expression {
    type Output = Result<Self, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a + b)),
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_add(a_r, a_i, b_r, b_i))
            }
            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_add(a, 0.0, b_r, b_i))
            }
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(Self::complex_add(a_r, a_i, b, 0.0))
            }
            (Expression::Vector(a), Expression::Vector(b)) => {
                if a.len() != b.len() {
                    return Err(EvaluationError::InvalidOperation(
                        "Vector addition: vectors must have the same dimension".to_string(),
                    ));
                }

                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().add(y.clone()))
                    .collect();

                Ok(Expression::Vector(result?))
            }
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                if a_rows != b_rows || a_cols != b_cols {
                    return Err(EvaluationError::InvalidOperation(
                        "Matrix addition: matrices must have the same dimensions".to_string(),
                    ));
                }

                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().add(y.clone()))
                    .collect();

                Ok(Expression::Matrix(result?, a_rows, a_cols))
            }
            (left, right) => Ok(Expression::Add(Box::new(left), Box::new(right))),
        }
    }
}

impl Sub for Expression {
    type Output = Result<Self, EvaluationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a - b)),
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_sub(a_r, a_i, b_r, b_i))
            }
            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_sub(a, 0.0, b_r, b_i))
            }
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(Self::complex_sub(a_r, a_i, b, 0.0))
            }
            (Expression::Vector(a), Expression::Vector(b)) => {
                if a.len() != b.len() {
                    return Err(EvaluationError::InvalidOperation(
                        "Vector subtraction: vectors must have the same dimension".to_string(),
                    ));
                }

                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().sub(y.clone()))
                    .collect();

                Ok(Expression::Vector(result?))
            }
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                if a_rows != b_rows || a_cols != b_cols {
                    return Err(EvaluationError::InvalidOperation(
                        "Matrix subtraction: matrices must have the same dimensions".to_string(),
                    ));
                }

                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().sub(y.clone()))
                    .collect();

                Ok(Expression::Matrix(result?, a_rows, a_cols))
            }
            (left, right) => Ok(Expression::Sub(Box::new(left), Box::new(right))),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Self, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => {
                // Always return positive zero if either operand is zero
                if a.abs() < EPSILON || b.abs() < EPSILON {
                    Ok(Expression::Real(0.0))
                } else {
                    Ok(Expression::Real(a * b))
                }
            }
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                // Check if either is zero
                let a_is_zero = a_r.abs() < EPSILON && a_i.abs() < EPSILON;
                let b_is_zero = b_r.abs() < EPSILON && b_i.abs() < EPSILON;

                if a_is_zero || b_is_zero {
                    Ok(Expression::Complex(0.0, 0.0))
                } else {
                    Ok(Self::complex_mul(a_r, a_i, b_r, b_i))
                }
            }
            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                if a.abs() < EPSILON || (b_r.abs() < EPSILON && b_i.abs() < EPSILON) {
                    Ok(Expression::Complex(0.0, 0.0))
                } else {
                    Ok(Self::complex_mul(a, 0.0, b_r, b_i))
                }
            }
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                if (a_r.abs() < EPSILON && a_i.abs() < EPSILON) || b.abs() < EPSILON {
                    Ok(Expression::Complex(0.0, 0.0))
                } else {
                    Ok(Self::complex_mul(a_r, a_i, b, 0.0))
                }
            }

            // Scalar * Vector
            (Expression::Real(s), Expression::Vector(v))
            | (Expression::Vector(v), Expression::Real(s)) => {
                if s.abs() < EPSILON {
                    // Return zero vector
                    let zero_vec = vec![Expression::Real(0.0); v.len()];
                    Ok(Expression::Vector(zero_vec))
                } else {
                    let result: Result<Vec<Expression>, _> = v
                        .iter()
                        .map(|x| x.clone().mul(Expression::Real(s)))
                        .collect();
                    Ok(Expression::Vector(result?))
                }
            }
            (Expression::Complex(r, i), Expression::Vector(v))
            | (Expression::Vector(v), Expression::Complex(r, i)) => {
                if r.abs() < EPSILON && i.abs() < EPSILON {
                    let zero_vec = vec![Expression::Complex(0.0, 0.0); v.len()];
                    Ok(Expression::Vector(zero_vec))
                } else {
                    let result: Result<Vec<Expression>, _> = v
                        .iter()
                        .map(|x| x.clone().mul(Expression::Complex(r, i)))
                        .collect();
                    Ok(Expression::Vector(result?))
                }
            }

            // Scalar * Matrix
            (Expression::Real(s), Expression::Matrix(data, rows, cols))
            | (Expression::Matrix(data, rows, cols), Expression::Real(s)) => {
                if s.abs() < EPSILON {
                    let zero_matrix = vec![Expression::Real(0.0); rows * cols];
                    Ok(Expression::Matrix(zero_matrix, rows, cols))
                } else {
                    let result: Result<Vec<Expression>, _> = data
                        .iter()
                        .map(|x| x.clone().mul(Expression::Real(s)))
                        .collect();
                    Ok(Expression::Matrix(result?, rows, cols))
                }
            }
            (Expression::Complex(r, i), Expression::Matrix(data, rows, cols))
            | (Expression::Matrix(data, rows, cols), Expression::Complex(r, i)) => {
                if r.abs() < EPSILON && i.abs() < EPSILON {
                    let zero_matrix = vec![Expression::Complex(0.0, 0.0); rows * cols];
                    Ok(Expression::Matrix(zero_matrix, rows, cols))
                } else {
                    let result: Result<Vec<Expression>, _> = data
                        .iter()
                        .map(|x| x.clone().mul(Expression::Complex(r, i)))
                        .collect();
                    Ok(Expression::Matrix(result?, rows, cols))
                }
            }

            // Hadamard product (element-wise) for matrices
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                if a_rows != b_rows || a_cols != b_cols {
                    return Err(EvaluationError::InvalidOperation(
                        "Hadamard product: matrices must have the same dimensions".to_string(),
                    ));
                }

                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().mul(y.clone())?.reduce())
                    .collect();

                Ok(Expression::Matrix(result?, a_rows, a_cols))
            }

            // Matrix * Vector
            (Expression::Matrix(a, rows, cols), Expression::Vector(b)) => {
                if cols != b.len() {
                    return Err(EvaluationError::InvalidOperation(
                        "Matrix-Vector multiplication: matrix columns must equal vector size"
                            .to_string(),
                    ));
                }

                let mut result = Vec::with_capacity(rows);

                for i in 0..rows {
                    let mut sum = Expression::Complex(0.0, 0.0);
                    let row_start = i * cols;

                    for k in 0..cols {
                        let left = a[row_start + k].clone();
                        let right = b[k].clone();
                        let product = left.mul(right)?;
                        sum = sum.add(product)?;
                    }

                    result.push(sum.reduce()?);
                }

                Ok(Expression::Vector(result))
            }

            (Expression::Vector(_), Expression::Vector(_)) => Err(
                EvaluationError::InvalidOperation("Cannot multiply vector by vector".to_string()),
            ),

            (left, right) => Ok(Expression::Mul(Box::new(left), Box::new(right))),
        }
    }
}

impl Expression {
    pub fn mat_mul(self, rhs: Self) -> Result<Self, EvaluationError> {
        match (self, rhs) {
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                if a_cols != b_rows {
                    return Err(EvaluationError::InvalidOperation(
                        "Matrix multiplication: left matrix columns must equal right matrix rows"
                            .to_string(),
                    ));
                }

                let mut result = Vec::with_capacity(a_rows * b_cols);

                for i in 0..a_rows {
                    for j in 0..b_cols {
                        let mut sum = Expression::Complex(0.0, 0.0);

                        for k in 0..a_cols {
                            let left = a[i * a_cols + k].clone();
                            let right = b[k * b_cols + j].clone();

                            let product = left.mul(right)?;
                            sum = sum.add(product)?.reduce()?;
                        }

                        result.push(sum);
                    }
                }

                Ok(Expression::Matrix(result, a_rows, b_cols))
            }
            (_, _) => Err(EvaluationError::InvalidOperation(
                "Matrix multiplication not implemented for this type".to_string(),
            )),
        }
    }
}

impl Div for Expression {
    type Output = Result<Self, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a / b)),
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Self::complex_div(a_r, a_i, b_r, b_i)
            }
            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Self::complex_div(a, 0.0, b_r, b_i)
            }
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Self::complex_div(a_r, a_i, b, 0.0)
            }
            (Expression::Vector(v), Expression::Real(s)) => {
                let result: Result<Vec<Expression>, _> = v
                    .iter()
                    .map(|x| x.clone().div(Expression::Real(s)))
                    .collect();
                Ok(Expression::Vector(result?))
            }
            (Expression::Vector(v), Expression::Complex(r, i)) => {
                let result: Result<Vec<Expression>, _> = v
                    .iter()
                    .map(|x| x.clone().div(Expression::Complex(r, i)))
                    .collect();
                Ok(Expression::Vector(result?))
            }
            (Expression::Matrix(data, rows, cols), Expression::Real(s)) => {
                let result: Result<Vec<Expression>, _> = data
                    .iter()
                    .map(|x| x.clone().div(Expression::Real(s)))
                    .collect();
                Ok(Expression::Matrix(result?, rows, cols))
            }
            (Expression::Matrix(data, rows, cols), Expression::Complex(r, i)) => {
                let result: Result<Vec<Expression>, _> = data
                    .iter()
                    .map(|x| x.clone().div(Expression::Complex(r, i)))
                    .collect();
                Ok(Expression::Matrix(result?, rows, cols))
            }
            (left, right) => Ok(Expression::Div(Box::new(left), Box::new(right))),
        }
    }
}

impl Rem for Expression {
    type Output = Result<Self, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => {
                if b < EPSILON {
                    return Err(EvaluationError::InvalidOperation(
                        "Modulo by zero".to_string(),
                    ));
                }
                Ok(Expression::Real(a % b))
            }
            (left, right) => Ok(Expression::Mod(Box::new(left), Box::new(right))),
        }
    }
}

impl Neg for Expression {
    type Output = Result<Self, EvaluationError>;

    fn neg(self) -> Self::Output {
        match self {
            Expression::Real(n) if n.abs() < EPSILON => Ok(Expression::Real(0.0)),
            Expression::Real(n) => Ok(Expression::Real(-n)),
            Expression::Complex(r, i) if r.abs() < EPSILON && i.abs() < EPSILON => {
                Ok(Expression::Complex(0.0, 0.0))
            }
            Expression::Complex(r, i) => Ok(Expression::Complex(-r, -i)),
            Expression::Vector(v) => {
                let result: Result<Vec<Expression>, _> = v.into_iter().map(|x| x.neg()).collect();
                Ok(Expression::Vector(result?))
            }
            Expression::Matrix(data, rows, cols) => {
                let result: Result<Vec<Expression>, _> =
                    data.into_iter().map(|x| x.neg()).collect();
                Ok(Expression::Matrix(result?, rows, cols))
            }
            // -(a + b) = (-a) + (-b) = -a - b
            Expression::Add(left, right) => Expression::Real(0.0).sub(*left)?.sub(*right),
            // -(a - b) = -a + b = b - a
            Expression::Sub(left, right) => (*right).sub(*left),
            // -(a * b) = (-a) * b
            Expression::Mul(left, right) => {
                let neg_left = (*left).neg()?;
                neg_left.mul(*right)
            }
            // Double negative: -(-x) = x
            Expression::Neg(inner) => Ok(*inner),
            _ => Ok(Expression::Neg(Box::new(self))),
        }
    }
}

impl Expression {
    pub fn pow(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a.powf(b))),
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(Self::complex_pow(a_r, a_i, b, 0.0))
            }
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_pow(a_r, a_i, b_r, b_i))
            }
            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(Self::complex_pow(a, 0.0, b_r, b_i))
            }
            (Expression::Matrix(data, rows, cols), Expression::Real(b)) => {
                // Check if matrix is square
                if rows != cols {
                    return Err(EvaluationError::InvalidOperation(
                        "Matrix must be square for exponentiation".to_string(),
                    ));
                }

                let n = b as i32;

                // Handle negative powers
                if n < 0 {
                    return Err(EvaluationError::InvalidOperation(
                        "Negative powers not supported".to_string(),
                    ));
                }

                // Handle power of 0 - return identity matrix
                if n == 0 {
                    let mut identity = Vec::with_capacity(rows * cols);
                    for i in 0..rows {
                        for j in 0..cols {
                            if i == j {
                                identity.push(Expression::Real(1.0));
                            } else {
                                identity.push(Expression::Real(0.0));
                            }
                        }
                    }
                    return Ok(Expression::Matrix(identity, rows, cols));
                }

                // Handle power of 1
                if n == 1 {
                    return Ok(Expression::Matrix(data, rows, cols));
                }

                // For powers > 1, use repeated multiplication
                let mut result = Expression::Matrix(data.clone(), rows, cols);
                for _ in 1..n {
                    result = result.mat_mul(Expression::Matrix(data.clone(), rows, cols))?;
                }

                Ok(result)
            }
            (left, right) => Ok(Expression::Pow(Box::new(left), Box::new(right))),
        }
    }

    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => {
                if *n < 0.0 {
                    Ok(Expression::Complex(0.0, n.abs().sqrt()))
                } else {
                    Ok(Expression::Real(n.sqrt()))
                }
            }
            Expression::Complex(r, i) => Ok(Self::complex_sqrt(*r, *i)),
            _ => Err(EvaluationError::InvalidOperation(
                "Sqrt is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Complex(r, i) => Ok(Expression::Real(Self::complex_abs(*r, *i))),
            _ => Err(EvaluationError::InvalidOperation(
                "Abs is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.exp())),
            Expression::Complex(r, i) => Ok(Self::complex_exp(*r, *i)),
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn norm(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Vector(vector) => {
                let mut sum = Expression::Real(0.0);
                for expression in vector {
                    let x_squared = expression.clone().mul(expression.clone())?;
                    sum = sum.add(x_squared)?;
                }

                sum.sqrt()
            }
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn rad(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.to_radians())),
            _ => Err(EvaluationError::InvalidOperation(
                "Rad is not implemented for this type".to_string(),
            )),
        }
    }
}

impl Expression {
    pub fn evaluate(&self, context: &Context) -> Result<Expression, EvaluationError> {
        self.evaluate_internal(context, &HashMap::new())
    }

    pub fn evaluate_internal(
        &self,
        context: &Context,
        scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(_) => Ok(self.clone()),
            Expression::Complex(real, imag) => {
                if imag.abs() < EPSILON {
                    Ok(Expression::Real(*real))
                } else {
                    Ok(self.clone())
                }
            }

            Expression::Vector(vector) => {
                let mut evaluated_vector = Vec::new();
                for element in vector.iter() {
                    let evaluated_element = element.evaluate_internal(context, scope)?.reduce()?;
                    evaluated_vector.push(evaluated_element);
                }
                Ok(Expression::Vector(evaluated_vector))
            }

            Expression::Matrix(matrix, rows, cols) => {
                let mut evaluated_matrix = Vec::new();
                for element in matrix.iter() {
                    let evaluated_element = element.evaluate_internal(context, scope)?.reduce()?;
                    evaluated_matrix.push(evaluated_element);
                }
                Ok(Expression::Matrix(evaluated_matrix, *rows, *cols))
            }

            Expression::Variable(name) => {
                // First check function parameter scope
                if let Some(expression) = scope.get(name) {
                    return Ok(expression.clone());
                }

                // Then check context variables
                match context.get_symbol(name) {
                    Some(Symbol::Variable(Variable { expression, .. })) => {
                        // Recursively evaluate the variable's expression
                        expression.evaluate_internal(context, scope)
                    }
                    Some(Symbol::Function(_) | Symbol::BuiltinFunction(_)) => {
                        Err(EvaluationError::InvalidOperation(format!(
                            "Cannot use function '{}' as variable",
                            name
                        )))
                    }
                    None => Ok(Expression::Variable(name.to_string())), // Keep symbolic
                }
            }

            Expression::FunctionCall(fc) => {
                match context.get_symbol(fc.name.as_str()) {
                    Some(Symbol::Function(fun)) => {
                        if fc.args.len() != fun.params.len() {
                            return Err(EvaluationError::WrongArgumentCount {
                                name: fc.name.clone(),
                                expected: fun.params.len(),
                                got: fc.args.len(),
                            });
                        }

                        // Evaluate arguments first
                        let evaluated_args: Result<Vec<_>, _> = fc
                            .args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        let evaluated_args = evaluated_args?;

                        // Create function scope by combining current scope with function parameters
                        let mut function_scope = scope.clone();
                        for (param, arg) in fun.params.iter().zip(evaluated_args.iter()) {
                            function_scope.insert(param.clone(), arg.clone());
                        }

                        // Evaluate function body with new scope
                        fun.body.evaluate_internal(context, &function_scope)
                    }
                    Some(Symbol::Variable(_)) => Err(EvaluationError::InvalidOperation(format!(
                        "'{}' is not a function",
                        fc.name
                    ))),
                    Some(Symbol::BuiltinFunction(function)) => match function {
                        BuiltinFunction::Rad => {
                            if fc.args.len() != 1 {
                                return Err(EvaluationError::WrongArgumentCount {
                                    name: fc.name.clone(),
                                    expected: 1,
                                    got: fc.args.len(),
                                });
                            }

                            let arg = fc.args[0].evaluate_internal(context, scope)?;
                            arg.rad()
                        }
                        BuiltinFunction::Norm => {
                            if fc.args.len() != 1 {
                                return Err(EvaluationError::WrongArgumentCount {
                                    name: fc.name.clone(),
                                    expected: 1,
                                    got: fc.args.len(),
                                });
                            }

                            let arg = fc.args[0].evaluate_internal(context, scope)?;
                            arg.norm()
                        }
                    },
                    None => {
                        // Evaluate arguments and keep as symbolic function call
                        let evaluated_args: Result<Vec<_>, _> = fc
                            .args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        Ok(Expression::FunctionCall(FunctionCall {
                            name: fc.name.clone(),
                            args: evaluated_args?,
                        }))
                    }
                }
            }

            Expression::Paren(inner) => {
                let inner = inner.evaluate_internal(context, scope)?.reduce()?;

                match inner {
                    Expression::Real(_) | Expression::Complex(_, _) => Ok(inner),
                    _ => Ok(Expression::Paren(Box::new(inner))),
                }
            }

            Expression::Add(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.add(right_eval)
            }
            Expression::Sub(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.sub(right_eval)
            }
            Expression::Mul(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.mul(right_eval)
            }
            Expression::MatMul(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.mat_mul(right_eval)
            }
            Expression::Div(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.div(right_eval)
            }
            Expression::Mod(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.rem(right_eval)
            }
            Expression::Pow(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.pow(right_eval)
            }
            Expression::Neg(inner) => inner.evaluate_internal(context, scope)?.neg(),
        }
    }

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
            Expression::Complex(real, imag) => {
                if imag.abs() < EPSILON {
                    Ok(Expression::Real(*real))
                } else {
                    Ok(self.clone())
                }
            }
            Expression::Add(..) | Expression::Sub(..) => {
                let terms = self.extract_terms(1.0)?;
                self.combine_like_terms(terms)
            }
            Expression::Mul(left, right) => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left_collected.is_value() && right_collected.is_value() {
                    return left_collected.mul(right_collected);
                }

                if left_collected == **left && right_collected == **right {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Mul(
                        Box::new(left_collected),
                        Box::new(right_collected),
                    ))
                }
            }
            Expression::Neg(inner) => {
                let collected = inner.collect_terms()?;

                if let Expression::Neg(double_inner) = &collected {
                    Ok(*double_inner.clone())
                } else if collected == **inner {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Neg(Box::new(collected)))
                }
            }
            _ => Ok(self.clone()),
        }
    }

    // Extract all terms from an addition/subtraction expression
    fn extract_terms(&self, sign: f64) -> Result<Vec<Term>, EvaluationError> {
        match self {
            Expression::Add(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(sign)?);
                Ok(terms)
            }
            Expression::Sub(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(-sign)?);
                Ok(terms)
            }
            Expression::Neg(inner) => inner.extract_terms(-sign),
            Expression::Real(n) => Ok(vec![Term::constant(sign * n)]),
            Expression::Complex(r, i) if i.abs() < EPSILON => Ok(vec![Term::constant(sign * r)]),
            Expression::Mul(..) => {
                let (coeff, expression) = self.extract_coefficient();
                Ok(vec![Term::new(sign * coeff, expression)])
            }
            _ => {
                // Variable, Pow, FunctionCall, etc.
                Ok(vec![Term::new(sign, self.clone())])
            }
        }
    }

    // Extract coefficient from multiplication, returning (coefficient, remaining_expression)
    fn extract_coefficient(&self) -> (f64, Expression) {
        let mut coefficient = 1.0;
        let mut non_constant_parts = Vec::new();

        self.collect_mul_parts(&mut coefficient, &mut non_constant_parts);

        if non_constant_parts.is_empty() {
            (coefficient, Expression::Real(1.0))
        } else if non_constant_parts.len() == 1 {
            (coefficient, non_constant_parts[0].clone())
        } else {
            let mut result = non_constant_parts[0].clone();
            for part in non_constant_parts.iter().skip(1) {
                result = Expression::Mul(Box::new(result), Box::new(part.clone()));
            }
            (coefficient, result)
        }
    }

    fn collect_mul_parts(&self, coefficient: &mut f64, parts: &mut Vec<Expression>) {
        match self {
            Expression::Real(n) => {
                *coefficient *= n;
            }
            Expression::Complex(r, i) if i.abs() < EPSILON => {
                *coefficient *= r;
            }
            Expression::Mul(left, right) => {
                left.collect_mul_parts(coefficient, parts);
                right.collect_mul_parts(coefficient, parts);
            }
            _ => {
                // Variable, Pow, FunctionCall, etc. - keep as-is
                parts.push(self.clone());
            }
        }
    }

    // Combine terms with the same expression part
    fn combine_like_terms(&self, terms: Vec<Term>) -> Result<Expression, EvaluationError> {
        let mut combined: HashMap<String, Term> = HashMap::new();

        for term in terms {
            let key = term.key();
            combined
                .entry(key)
                .and_modify(|existing| existing.coefficient += term.coefficient)
                .or_insert(term);
        }

        let mut positive_terms: Vec<Term> = Vec::new();
        let mut negative_terms: Vec<Term> = Vec::new();

        for term in combined.into_values() {
            if term.coefficient.abs() < EPSILON {
                continue; // Skip zero terms
            }

            if term.coefficient > 0.0 {
                positive_terms.push(term);
            } else {
                // Store as positive coefficient, we'll subtract later
                negative_terms.push(Term::new(-term.coefficient, term.expression));
            }
        }

        if positive_terms.is_empty() && negative_terms.is_empty() {
            return Ok(Expression::Real(0.0));
        }

        // Build expression starting with positive terms (or first negative if no positive)
        let mut result = if !positive_terms.is_empty() {
            let mut pos_iter = positive_terms.into_iter();
            let mut result = pos_iter.next().unwrap().to_expression();

            for term in pos_iter {
                result = Expression::Add(Box::new(result), Box::new(term.to_expression()));
            }
            result
        } else {
            // All terms are negative, start with negation of first term
            let first = negative_terms.remove(0);
            Expression::Neg(Box::new(first.to_expression()))
        };

        // Subtract all negative terms
        for term in negative_terms {
            result = Expression::Sub(Box::new(result), Box::new(term.to_expression()));
        }

        Ok(result)
    }
}

#[derive(Debug, Clone)]
struct Term {
    coefficient: f64,
    expression: Expression,
}

impl Term {
    fn new(coefficient: f64, expression: Expression) -> Self {
        Term {
            coefficient,
            expression,
        }
    }

    fn constant(value: f64) -> Self {
        Term {
            coefficient: value,
            expression: Expression::Real(1.0),
        }
    }

    fn to_expression(&self) -> Expression {
        let coeff_abs = self.coefficient.abs();

        if coeff_abs < EPSILON {
            return Expression::Real(0.0);
        }

        // Check if expression is just the constant 1.0
        if matches!(self.expression, Expression::Real(n) if (n - 1.0).abs() < EPSILON) {
            return Expression::Real(self.coefficient);
        }

        let abs_coeff = self.coefficient.abs();
        let is_negative = self.coefficient < 0.0;

        let base_expr = if (abs_coeff - 1.0).abs() < EPSILON {
            // Coefficient is ±1, just return the expression (or its negation)
            self.expression.clone()
        } else {
            // Coefficient is not ±1, create multiplication with absolute value
            Expression::Mul(
                Box::new(Expression::Real(abs_coeff)),
                Box::new(self.expression.clone()),
            )
        };

        if is_negative {
            Expression::Neg(Box::new(base_expr))
        } else {
            base_expr
        }
    }

    // Create a unique key for grouping like terms
    fn key(&self) -> String {
        format!("{}", self.expression)
    }
}
