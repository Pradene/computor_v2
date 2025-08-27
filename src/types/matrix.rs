use std::ops::{Add, Mul, Sub, Neg};

use crate::types::complex::Complex;
use crate::expression::Expression;

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
}

impl Add for Matrix {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            return Err("Invalid operation matrix doesn't have the same dimension".to_string());
        }

        let result: Result<Vec<Expression>, _> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols).map_err(|e| e),
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

        let result: Result<Vec<Expression>, _> = self.data.iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols).map_err(|e| e),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Mul for Matrix {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.cols != rhs.rows {
            return Err("Invalid operation: left matrix columns must equal right matrix rows".to_string());
        }

        let mut result = Vec::with_capacity(self.rows * rhs.cols);

        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = Expression::Complex(Complex::new(0.0, 0.0));
                
                for k in 0..self.cols {
                    let left_elem = self.get(i, k).ok_or("Index out of bounds")?;
                    let right_elem = rhs.get(k, j).ok_or("Index out of bounds")?;
                    
                    let result = left_elem.clone() * right_elem.clone();
                    match result {
                        Err(e) => return Err(e.to_string()),
                        Ok(_) => {},
                    };
                    
                    sum = (sum + result.unwrap()).map_err(|e| e.to_string())?;
                }
                
                result.push(sum);
            }
        }

        Matrix::new(result, self.rows, rhs.cols)
    }
}

impl Neg for Matrix {
    type Output = Result<Self, String>;

    fn neg(self) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self.data
            .into_iter()
            .map(|x| -x)
            .collect();

        match result {
            Ok(data) => Matrix::new(data, self.rows, self.cols).map_err(|e| e),
            Err(e) => Err(e.to_string()),
        }
    }
}
