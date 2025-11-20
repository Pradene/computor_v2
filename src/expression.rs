use std::{
    collections::HashMap,
    fmt,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

use crate::{
    constant::EPSILON,
    context::{Context, Symbol, Variable},
    error::EvaluationError,
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
                if real.abs() < EPSILON && imag.abs() < EPSILON {
                    write!(f, "0")?;
                } else if real.abs() < EPSILON {
                    write!(f, "{}i", imag)?;
                } else if imag.abs() < EPSILON {
                    write!(f, "{}", real)?;
                } else if *imag >= 0.0 {
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
            Expression::Sub(left, right) => {
                write!(f, "{} - ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Mul(left, right) => {
                write!(f, "{} * ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::MatMul(left, right) => {
                write!(f, "{} ** ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Div(left, right) => {
                write!(f, "{} / ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Mod(left, right) => {
                write!(f, "{} % ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Pow(left, right) => {
                write!(f, "{} ^ ", left)?;
                // Add parentheses if right side is Add or Sub
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
        };

        Ok(())
    }
}

impl Expression {
    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => n.abs() < EPSILON,
            Expression::Complex(r, i) => r.abs() < EPSILON && i.abs() < EPSILON,
            _ => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        matches!(
            self,
            Expression::Real(_)
                | Expression::Complex(_, _)
                | Expression::Vector(_)
                | Expression::Matrix(_, _, _)
        )
    }

    pub fn is_symbolic(&self) -> bool {
        !self.is_concrete()
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

    pub fn contains_variable(&self, var_name: &str) -> bool {
        match self {
            Expression::Real(_) | Expression::Complex(_, _) => false,
            Expression::Variable(name) => name == var_name,
            Expression::Vector(v) => v.iter().any(|e| e.contains_variable(var_name)),
            Expression::Matrix(data, _, _) => data.iter().any(|e| e.contains_variable(var_name)),
            Expression::FunctionCall(fc) => fc.args.iter().any(|e| e.contains_variable(var_name)),
            Expression::Paren(inner) => inner.contains_variable(var_name),
            Expression::Neg(inner) => inner.contains_variable(var_name),
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::MatMul(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                left.contains_variable(var_name) || right.contains_variable(var_name)
            }
        }
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
                        format!("Vector addition: both vectors must have the same dimension (got {} and {})", 
                            a.len(), b.len()),
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
                        format!("Matrix addition: both matrices must have the same dimensions (got {}×{} and {}×{})", 
                            a_rows, a_cols, b_rows, b_cols),
                    ));
                }
                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().add(y.clone()))
                    .collect();
                Ok(Expression::Matrix(result?, a_rows, a_cols))
            }

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot add {} and {}: incompatible types",
                    left, right
                )))
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
                        format!("Vector subtraction: both vectors must have the same dimension (got {} and {})", a.len(), b.len()),
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
                        format!("Matrix subtraction: both matrices must have the same dimensions (got {}×{} and {}×{})", 
                            a_rows, a_cols, b_rows, b_cols),
                    ));
                }
                let result: Result<Vec<Expression>, _> = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| x.clone().sub(y.clone()))
                    .collect();
                Ok(Expression::Matrix(result?, a_rows, a_cols))
            }

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot subtract {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::Sub(Box::new(left), Box::new(right))),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Self, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Scalar * Scalar
            (Expression::Real(a), Expression::Real(b)) => {
                if a.abs() < EPSILON || b.abs() < EPSILON {
                    Ok(Expression::Real(0.0))
                } else {
                    Ok(Expression::Real(a * b))
                }
            }

            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
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

            // Matrix * Matrix (Hadamard product)
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                if a_rows != b_rows || a_cols != b_cols {
                    return Err(EvaluationError::InvalidOperation(
                        format!("Hadamard product: matrices must have the same dimensions (got {}×{} and {}×{})", 
                            a_rows, a_cols, b_rows, b_cols),
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
            (Expression::Matrix(a, rows, cols), Expression::Vector(b))
            | (Expression::Vector(b), Expression::Matrix(a, rows, cols)) => {
                if cols != b.len() {
                    return Err(EvaluationError::InvalidOperation(
                        format!("Matrix-Vector multiplication: matrix has {} columns but vector has {} elements", 
                            cols, b.len()),
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

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot multiply {} and {}: incompatible types",
                    left, right
                )))
            }

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

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot matrix multiply {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::MatMul(Box::new(left), Box::new(right.clone()))),
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

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot divide {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::Div(Box::new(left), Box::new(right.clone()))),
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
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a % b)),

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot modulo {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::Mod(
                Box::new(left.clone()),
                Box::new(right.clone()),
            )),
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
            (Expression::Matrix(_, _, _), _) => Err(EvaluationError::InvalidOperation(
                "Cannot do power of matrix".to_string(),
            )),
            (Expression::Vector(_), _) => Err(EvaluationError::InvalidOperation(
                "Cannot do power of vector".to_string(),
            )),
            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot power {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::Pow(
                Box::new(left.clone()),
                Box::new(right.clone()),
            )),
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

            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot sqrt {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "sqrt".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Complex(r, i) => Ok(Expression::Real(Self::complex_abs(*r, *i))),
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot abs {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "abs".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.exp())),
            Expression::Complex(r, i) => Ok(Self::complex_exp(*r, *i)),
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot exp {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "exp".to_string(), args: vec![expression.clone()] })),
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
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot norm {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "norm".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn cos(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => {
                let cos = n.cos();
                let res = if cos < EPSILON { 0.0 } else { cos };
                Ok(Expression::Real(res))
            }
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot cos {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "cos".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn sin(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => {
                let sin = n.sin();
                let res = if sin < EPSILON { 0.0 } else { sin };
                Ok(Expression::Real(res))
            }
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot sin {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "sin".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn tan(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => {
                let tan = n.tan();
                let res = if tan < EPSILON { 0.0 } else { tan };
                Ok(Expression::Real(res))
            }
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot tan {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "tan".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn rad(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.to_radians())),
            expression if expression.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot rad {}: incompatible type",
                    expression
                )))
            }

            expression => Ok(Expression::FunctionCall(FunctionCall { name: "rad".to_string(), args: vec![expression.clone()] })),
        }
    }

    pub fn dot(&self, rhs: Expression) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Vector(v1), Expression::Vector(v2)) => {
                if v1.len() != v2.len() {
                    return Err(EvaluationError::InvalidOperation(format!(
                        "Dot product: vectors must have the same dimensions (got {} and {})",
                        v1.len(),
                        v2.len()
                    )));
                }

                let result = v1.iter().zip(v2.iter()).try_fold(
                    Expression::Complex(0.0, 0.0),
                    |acc, (a, b)| {
                        let product = a.clone().mul(b.clone())?;
                        acc.add(product)
                    },
                )?;

                Ok(result)
            }
            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot dot {} and {}: incompatible type",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::FunctionCall(FunctionCall { name: "dot".to_string(), args: vec![ left.clone(), right.clone()] })),
        }
    }

    pub fn cross(&self, rhs: Expression) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Vector(v1), Expression::Vector(v2)) => {
                if v1.len() != 3 || v2.len() != 3 {
                    return Err(EvaluationError::InvalidOperation(format!(
                        "Cross product: vectors must be 3 dimensions (got {} and {})",
                        v1.len(),
                        v2.len()
                    )));
                }

                let result: Vec<Expression> = vec![
                    (v1[1].clone().mul(v2[2].clone())?).sub(v1[2].clone().mul(v2[1].clone())?)?,
                    (v1[2].clone().mul(v2[0].clone())?).sub(v1[0].clone().mul(v2[2].clone())?)?,
                    (v1[0].clone().mul(v2[1].clone())?).sub(v1[1].clone().mul(v2[0].clone())?)?,
                ];

                Ok(Expression::Vector(result))
            }
            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot cross {} and {}: incompatible type",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::FunctionCall(FunctionCall { name: "cross".to_string(), args: vec![ left.clone(), right.clone() ] })),
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
                        // Check for self-reference before recursing
                        if expression.contains_variable(name) {
                            return Err(EvaluationError::InvalidOperation(format!(
                                "Variable '{}' is defined in terms of itself",
                                name
                            )));
                        }
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
                    Some(Symbol::BuiltinFunction(function)) => {
                        if fc.args.len() != function.arity() {
                            return Err(EvaluationError::WrongArgumentCount {
                                name: fc.name.clone(),
                                expected: function.arity(),
                                got: fc.args.len(),
                            });
                        }

                        let evaluated_args: Vec<Expression> = fc
                            .args
                            .iter()
                            .map(|e| e.evaluate_internal(context, scope))
                            .collect::<Result<_, _>>()?;

                        function.call(&evaluated_args)
                    }
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
            let distributed = current.distribute()?;
            
            let collected = distributed.collect_terms()?;
            if collected == current {
                break;
            }
            current = collected;
        }

        Ok(current)
    }

    /// Distribute multiplication over addition: a * (b + c) = a*b + a*c
    fn distribute(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Mul(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;

                match (&left_dist, &right_dist) {
                    // a * (b + c) = a*b + a*c
                    (_, Expression::Add(b, c)) => {
                        let left_times_b = left_dist.clone().mul((**b).clone())?;
                        let left_times_c = left_dist.clone().mul((**c).clone())?;
                        left_times_b.add(left_times_c)
                    }
                    // a * (b - c) = a*b - a*c
                    (_, Expression::Sub(b, c)) => {
                        let left_times_b = left_dist.clone().mul((**b).clone())?;
                        let left_times_c = left_dist.clone().mul((**c).clone())?;
                        left_times_b.sub(left_times_c)
                    }
                    // (a + b) * c = a*c + b*c
                    (Expression::Add(a, b), _) => {
                        let a_times_right = (**a).clone().mul(right_dist.clone())?;
                        let b_times_right = (**b).clone().mul(right_dist.clone())?;
                        a_times_right.add(b_times_right)
                    }
                    // (a - b) * c = a*c - b*c
                    (Expression::Sub(a, b), _) => {
                        let a_times_right = (**a).clone().mul(right_dist.clone())?;
                        let b_times_right = (**b).clone().mul(right_dist.clone())?;
                        a_times_right.sub(b_times_right)
                    }
                    _ => Ok(Expression::Mul(
                        Box::new(left_dist),
                        Box::new(right_dist),
                    )),
                }
            }
            Expression::Add(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.add(right_dist)
            }
            Expression::Sub(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.sub(right_dist)
            }
            Expression::Neg(inner) => {
                let inner_dist = inner.distribute()?;
                inner_dist.neg()
            }
            Expression::Pow(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.pow(right_dist)
            }
            _ => Ok(self.clone()),
        }
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

                if left_collected.is_concrete() && right_collected.is_concrete() {
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
