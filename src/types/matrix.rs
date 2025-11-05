use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::error::EvaluationError;
use crate::types::complex::Complex;
use crate::types::expression::{Expression, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Expression>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Expression>, rows: usize, cols: usize) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Matrix cannot be empty".to_string());
        }

        if data.len() != rows * cols {
            return Err("Matrix wrong dimension".to_string());
        }

        Ok(Matrix { data, rows, cols })
    }

    fn identity(dimension: usize) -> Result<Self, EvaluationError> {
        let mut data = Vec::with_capacity(dimension * dimension);

        for i in 0..dimension {
            for j in 0..dimension {
                if i == j {
                    data.push(Expression::Value(Value::Real(1.0)));
                } else {
                    data.push(Expression::Value(Value::Real(0.0)));
                }
            }
        }

        Matrix::new(data, dimension, dimension).map_err(EvaluationError::InvalidOperation)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&Expression> {
        self.data.get(row * self.cols + col)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Expression> {
        self.data.iter()
    }

    pub fn pow(&self, n: i32) -> Result<Self, EvaluationError> {
        // Check if matrix is square
        if self.rows != self.cols {
            return Err(EvaluationError::InvalidOperation(
                "Matrix must be square for exponentiation".to_string(),
            ));
        }

        // Handle negative powers
        if n < 0 {
            return Err(EvaluationError::InvalidOperation(
                "Negative powers not supported".to_string(),
            ));
        }

        // Handle power of 0 - return identity matrix
        if n == 0 {
            return Self::identity(self.rows);
        }

        // Handle power of 1
        if n == 1 {
            return Ok(self.clone());
        }

        // For powers > 1, use repeated multiplication
        let mut result = self.clone();
        for _ in 1..n {
            result = result
                .mul(self.clone())
                .map_err(EvaluationError::InvalidOperation)?;
        }

        Ok(result)
    }
}

impl Add for Matrix {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err("Invalid operation matrix doesn't have the same dimension".to_string());
        }

        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone().add(b.clone()))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Sub for Matrix {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err("Invalid operation matrix doesn't have the same dimension".to_string());
        }

        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone().sub(b.clone()))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Mul for Matrix {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols != rhs.rows {
            return Err(
                "Invalid operation: left matrix columns must equal right matrix rows".to_string(),
            );
        }

        let mut result = Vec::with_capacity(self.rows * rhs.cols);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = Expression::Value(Value::Complex(Complex::new(0.0, 0.0)));

                for k in 0..self.cols {
                    let left_elem = self.get(i, k).ok_or("Index out of bounds")?;
                    let right_elem = rhs.get(k, j).ok_or("Index out of bounds")?;

                    let result = left_elem.clone().mul(right_elem.clone());
                    if let Err(e) = result {
                        return Err(e.to_string());
                    };

                    sum = (sum.add(result.unwrap())).map_err(|e| e.to_string())?;
                }

                result.push(sum);
            }
        }

        Matrix::new(result, self.rows, rhs.cols)
    }
}

impl Mul<f64> for Matrix {
    type Output = Result<Self, String>;

    fn mul(self, rhs: f64) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().mul(Expression::Value(Value::Real(rhs))))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Mul<Complex> for Matrix {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Complex) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().mul(Expression::Value(Value::Complex(rhs))))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Div<f64> for Matrix {
    type Output = Result<Self, String>;

    fn div(self, rhs: f64) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().div(Expression::Value(Value::Real(rhs))))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Div<Complex> for Matrix {
    type Output = Result<Self, String>;

    fn div(self, rhs: Complex) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().div(Expression::Value(Value::Complex(rhs))))
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Neg for Matrix {
    type Output = Result<Self, String>;

    fn neg(self) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self.data.into_iter().map(|x| x.neg()).collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for r in 0..self.rows {
            if r > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(r, c).unwrap())?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")?;

        Ok(())
    }
}
