use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::error::EvaluationError;
use crate::types::complex::Complex;
use crate::types::matrix::Matrix;
use crate::types::vector::Vector;

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
            Value::Vector(v) => {
                let negated_data: Result<Vec<Expression>, _> =
                    v.iter().map(|elem| elem.clone().neg()).collect();
                match negated_data {
                    Ok(data) => Ok(Value::Vector(Vector::new(data).unwrap())),
                    Err(_) => panic!("Vector negation failed"),
                }
            }
            Value::Matrix(m) => {
                let negated_data: Result<Vec<Expression>, _> =
                    m.iter().map(|elem| elem.clone().neg()).collect();
                match negated_data {
                    Ok(data) => Ok(Value::Matrix(
                        Matrix::new(data, m.rows(), m.cols()).unwrap(),
                    )),
                    Err(_) => Err(EvaluationError::InvalidOperation(
                        "Matrix negation failed".to_string(),
                    )),
                }
            }
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
            _ => Err(EvaluationError::InvalidOperation(
                "Addition not supported for these Value types".to_string(),
            )),
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
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector addition failed: {}", e))
            })?)),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix addition failed: {}", e))
            })?)),
            _ => Err(EvaluationError::InvalidOperation(
                "Addition not supported for these Value types".to_string(),
            )),
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
            (Value::Vector(v), Value::Real(s)) | (Value::Real(s), Value::Vector(v)) => {
                Ok(Value::Vector((v * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Vector multiplication failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Matrix(m), Value::Real(s)) | (Value::Real(s), Value::Matrix(m)) => {
                Ok(Value::Matrix((m * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Matrix multiplication failed: {}",
                        e
                    ))
                })?))
            }
            _ => Err(EvaluationError::InvalidOperation(
                "Multiplication not supported for these Value types".to_string(),
            )),
        }
    }
}

impl Div for Value {
    type Output = Result<Self, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a / b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a / b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) / b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a / Complex::new(b, 0.0))),
            (Value::Vector(v), Value::Real(s)) => Ok(Value::Vector((v / s).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector-Scalar division failed: {}", e))
            })?)),
            (Value::Matrix(m), Value::Real(s)) => Ok(Value::Matrix((m / s).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix-Scalar division failed: {}", e))
            })?)),
            _ => Err(EvaluationError::InvalidOperation(
                "Division not supported for these Value types".to_string(),
            )),
        }
    }
}

impl Rem for Value {
    type Output = Result<Self, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a % b)),
            _ => Err(EvaluationError::InvalidOperation(
                "Modulo operation only supported for Real values".to_string(),
            )),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Real(n) => write!(f, "{}", n),
            Value::Complex(c) => write!(f, "{}", c),
            Value::Vector(v) => write!(f, "{}", v),
            Value::Matrix(m) => write!(f, "{}", m),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Value(Value),
    Variable(String),
    FunctionCall { name: String, args: Vec<Expression> },
    Neg(Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Mod(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Value(value) => write!(f, "{}", value),
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
            Expression::Neg(operand) => write!(f, "-{}", operand),
            Expression::Add(left, right) => write!(f, "({} + {})", left, right),
            Expression::Sub(left, right) => write!(f, "({} - {})", left, right),
            Expression::Mul(left, right) => write!(f, "({} * {})", left, right),
            Expression::Div(left, right) => write!(f, "({} / {})", left, right),
            Expression::Mod(left, right) => write!(f, "({} % {})", left, right),
            Expression::Pow(left, right) => write!(f, "({} ^ {})", left, right),
        }
    }
}

impl Expression {
    pub fn is_value(&self) -> bool {
        matches!(self, Expression::Value(_))
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Value(Value::Real(n)) => *n == 0.0,
            Expression::Value(Value::Complex(c)) => c.is_real() && c.real == 0.0,
            _ => false,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Value(Value::Real(_)) => true,
            Expression::Value(Value::Complex(c)) => c.is_real(),
            _ => false,
        }
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
            Expression::Add(..) => {
                let mut terms = HashMap::new();
                self.collect_addition_terms(&mut terms, 1.0)?;
                self.rebuild_from_terms(terms)
            }
            Expression::Sub(left, right) => {
                // Convert a - b to a + (-b) for easier term collection
                let neg_right = Expression::Neg(right.clone());
                let as_addition = Expression::Add(left.clone(), Box::new(neg_right));
                as_addition.collect_terms()
            }
            Expression::Mul(left, right) => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left.is_value() && right.is_value() {
                    return left.clone().mul(*right.clone());
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

                if let Expression::Neg(inner) = &collected {
                    Ok(*inner.clone())
                } else if collected == **inner {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Neg(Box::new(collected)))
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
            Expression::Add(left, right) => {
                left.collect_addition_terms(terms, coeff)?;
                right.collect_addition_terms(terms, coeff)?;
            }
            Expression::Sub(left, right) => {
                left.collect_addition_terms(terms, coeff)?;
                right.collect_addition_terms(terms, -coeff)?;
            }
            Expression::Neg(inner) => {
                inner.collect_addition_terms(terms, -coeff)?;
            }
            Expression::Value(v) => match v {
                Value::Real(n) => {
                    *terms.entry("__constant__".to_string()).or_insert(0.0) += coeff * n;
                }
                Value::Complex(c) if c.is_real() => {
                    *terms.entry("__constant__".to_string()).or_insert(0.0) += coeff * c.real;
                }
                _ => {
                    let key = format!("{}", self);
                    *terms.entry(key).or_insert(0.0) += coeff;
                }
            },
            Expression::Variable(name) => {
                *terms.entry(name.clone()).or_insert(0.0) += coeff;
            }
            Expression::Mul(..) => {
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
            Expression::Pow(left, right) => {
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
            Expression::Value(v) => match v {
                Value::Real(n) => *coefficient *= *n,
                Value::Complex(c) if c.is_real() => *coefficient *= c.real,
                _ => {}
            },
            Expression::Variable(name) => variables.push(name.clone()),
            Expression::Mul(left, right) => {
                left.collect_multiplication_parts(coefficient, variables);
                right.collect_multiplication_parts(coefficient, variables);
            }
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
                    Expression::Neg(Box::new(Expression::Value(Value::Real(-coeff))))
                } else {
                    Expression::Value(Value::Real(coeff))
                }
            } else if (coeff - 1.0).abs() < f64::EPSILON {
                Expression::Variable(term_str)
            } else if (coeff + 1.0).abs() < f64::EPSILON {
                Expression::Neg(Box::new(Expression::Variable(term_str)))
            } else if coeff < 0.0 {
                Expression::Neg(Box::new(Expression::Mul(
                    Box::new(Expression::Value(Value::Real(-coeff))),
                    Box::new(Expression::Variable(term_str)),
                )))
            } else {
                Expression::Mul(
                    Box::new(Expression::Value(Value::Real(coeff))),
                    Box::new(Expression::Variable(term_str)),
                )
            };

            result_terms.push(term_expr);
        }

        if result_terms.is_empty() {
            return Ok(Expression::Value(Value::Real(0.0)));
        }

        let mut result = result_terms[0].clone();
        for term in result_terms.iter().skip(1) {
            match term {
                Expression::Neg(inner) => {
                    result = Expression::Sub(Box::new(result), inner.clone());
                }
                _ => {
                    result = Expression::Add(Box::new(result), Box::new(term.clone()));
                }
            }
        }

        Ok(result)
    }
}

impl Expression {
    pub fn pow(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Value(Value::Real(a)), Expression::Value(Value::Real(b))) => {
                Ok(Expression::Value(Value::Real(a.powf(*b))))
            }
            (Expression::Value(Value::Complex(a)), Expression::Value(Value::Real(b))) => Ok(
                Expression::Value(Value::Complex(a.pow(Complex::new(*b, 0.0)))),
            ),
            (Expression::Value(Value::Complex(a)), Expression::Value(Value::Complex(b))) => {
                Ok(Expression::Value(Value::Complex(a.pow(*b))))
            }
            (Expression::Value(Value::Real(a)), Expression::Value(Value::Complex(b))) => Ok(
                Expression::Value(Value::Complex(Complex::new(*a, 0.0).pow(*b))),
            ),
            (Expression::Value(Value::Matrix(a)), Expression::Value(Value::Real(b))) => {
                Ok(Expression::Value(Value::Matrix(a.pow(*b as i32)?)))
            }
            (Expression::Value(Value::Matrix(_)), _) | (_, Expression::Value(Value::Matrix(_))) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Matrix exponentiation is not supported".to_string(),
                ))
            }
            (Expression::Value(Value::Vector(_)), _) | (_, Expression::Value(Value::Vector(_))) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Vector exponentiation is not supported".to_string(),
                ))
            }
            _ => Ok(Expression::Pow(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn add(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                let result = left.clone().add(right.clone())?;
                Ok(Expression::Value(result))
            }
            _ => Ok(Expression::Add(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn sub(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                let result = left.clone().sub(right.clone())?;
                Ok(Expression::Value(result))
            }
            _ => Ok(Expression::Sub(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                let result = left.clone().mul(right.clone())?;
                Ok(Expression::Value(result))
            }
            _ => Ok(Expression::Mul(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn div(self, rhs: Self) -> Result<Expression, EvaluationError> {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                let result = left.clone().div(right.clone())?;
                Ok(Expression::Value(result))
            }
            _ => Ok(Expression::Div(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn rem(self, rhs: Self) -> Result<Expression, EvaluationError> {
        if rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (self, rhs) {
            (Expression::Value(Value::Real(a)), Expression::Value(Value::Real(b))) => {
                Ok(Expression::Value(Value::Real(a % b)))
            }
            (Expression::Value(Value::Complex(a)), Expression::Value(Value::Complex(b)))
                if a.is_real() && b.is_real() =>
            {
                Ok(Expression::Value(Value::Real(a.real % b.real)))
            }
            // All other cases are invalid
            _ => Err(EvaluationError::UnsupportedOperation(
                "Modulo operation is only supported for real numbers".to_string(),
            )),
        }
    }

    pub fn neg(self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(n) => Ok(Expression::Value(n.neg()?)),
            // Distribute minus over binary operations
            Expression::Add(left, right) => {
                let neg_left = (*left).neg()?;
                let neg_right = (*right).neg()?;
                neg_left.sub(neg_right)
            }
            Expression::Sub(left, right) => (*right).sub(*left),
            Expression::Mul(left, right) => {
                let neg_left = (*left).neg()?;
                neg_left.mul(*right)
            }
            // // Double negative: -(-x) = x
            Expression::Neg(inner) => Ok(*inner.clone()),
            _ => Ok(Expression::Neg(Box::new(self))),
        }
    }

    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.sqrt()))),
            Expression::Value(Value::Complex(n)) => Ok(Expression::Value(Value::Complex(n.sqrt()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Sqrt is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.abs()))),
            Expression::Value(Value::Complex(n)) => Ok(Expression::Value(Value::Real(n.abs()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Abs is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.exp()))),
            Expression::Value(Value::Complex(n)) => Ok(Expression::Value(Value::Complex(n.exp()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
    }
}
