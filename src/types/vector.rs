use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::expression::Expression;
use crate::types::complex::Complex;
use crate::types::value::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    data: Vec<Expression>,
}

impl Vector {
    pub fn new(data: Vec<Expression>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Vector cannot be empty".to_string());
        }

        Ok(Vector { data })
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, index: usize) -> Option<&Expression> {
        self.data.get(index)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Expression> {
        self.data.iter()
    }
}

impl Add for Vector {
    type Output = Result<Self, String>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            return Err("Invalid operation vector doesn't have the same dimension".to_string());
        }

        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone().add(b.clone()))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Sub for Vector {
    type Output = Result<Self, String>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            return Err("Invalid operation vector doesn't have the same dimension".to_string());
        }

        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.clone().sub(b.clone()))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Mul<f64> for Vector {
    type Output = Result<Self, String>;

    fn mul(self, rhs: f64) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().mul(Expression::Value(Value::Real(rhs))))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Mul<Complex> for Vector {
    type Output = Result<Self, String>;

    fn mul(self, rhs: Complex) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().mul(Expression::Value(Value::Complex(rhs))))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Div<f64> for Vector {
    type Output = Result<Self, String>;

    fn div(self, rhs: f64) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().div(Expression::Value(Value::Real(rhs))))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Div<Complex> for Vector {
    type Output = Result<Self, String>;

    fn div(self, rhs: Complex) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self
            .data
            .iter()
            .map(|m| m.clone().div(Expression::Value(Value::Complex(rhs))))
            .collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl Neg for Vector {
    type Output = Result<Self, String>;

    fn neg(self) -> Self::Output {
        let result: Result<Vec<Expression>, _> = self.data.into_iter().map(|x| x.neg()).collect();

        match result {
            Ok(data) => Vector::new(data),
            Err(e) => Err(e.to_string()),
        }
    }
}

impl fmt::Display for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, v) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }

            write!(f, "{}", v)?;
        }
        write!(f, "]")?;

        Ok(())
    }
}
