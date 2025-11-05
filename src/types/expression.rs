use std::collections::HashMap;
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

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub args: Vec<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Value(Value),
    Variable(String),
    FunctionCall(FunctionCall),
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
            Expression::FunctionCall(fc) => {
                write!(f, "{}(", fc.name)?;
                for (i, arg) in fc.args.iter().enumerate() {
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
            Expression::Value(v) => v.is_zero(),
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
        matches!(self, Expression::FunctionCall(_))
    }
}

impl Add for Expression {
    type Output = Result<Self, EvaluationError>;
    fn add(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().add(right.clone())?))
            }
            _ => Ok(Expression::Add(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Sub for Expression {
    type Output = Result<Self, EvaluationError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().sub(right.clone())?))
            }
            _ => Ok(Expression::Sub(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Self, EvaluationError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().mul(right.clone())?))
            }
            _ => Ok(Expression::Mul(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Div for Expression {
    type Output = Result<Self, EvaluationError>;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().div(right.clone())?))
            }
            _ => Ok(Expression::Div(Box::new(self), Box::new(rhs))),
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

        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().rem(right.clone())?))
            }
            _ => Ok(Expression::Mod(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Neg for Expression {
    type Output = Result<Self, EvaluationError>;
    fn neg(self) -> Self::Output {
        match self {
            Expression::Value(n) => Ok(Expression::Value(n.neg()?)),
            // -(a + b) = (-a) + (-b) = -a - b
            Expression::Add(left, right) => {
                Expression::Value(Value::Real(0.0)).sub(*left)?.sub(*right)
            }
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
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().pow(right.clone())?))
            }
            _ => Ok(Expression::Pow(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => {
                if *n < 0.0 {
                    Ok(Expression::Value(Value::Complex(Complex::new(
                        0.0,
                        n.abs().sqrt(),
                    ))))
                } else {
                    Ok(Expression::Value(Value::Real(n.sqrt())))
                }
            }
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Complex(c.sqrt()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Sqrt is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.abs()))),
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Real(c.abs()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Abs is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.exp()))),
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Complex(c.exp()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
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
            Expression::Add(..) | Expression::Sub(..) => {
                let mut terms = HashMap::new();
                self.collect_addition_terms(&mut terms, 1.0)?;
                self.rebuild_from_terms(terms)
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
            Expression::Pow(base, exponent) => {
                let mut base_coeff = 1.0;
                let mut base_vars = Vec::new();
                base.collect_multiplication_parts(&mut base_coeff, &mut base_vars);

                let mut exp_coeff = 1.0;
                let mut exp_vars = Vec::new();
                exponent.collect_multiplication_parts(&mut exp_coeff, &mut exp_vars);

                if base_vars.len() == 1 && exp_vars.is_empty() && base_coeff == 1.0 {
                    if let Expression::Value(Value::Real(n)) = &**exponent {
                        variables.push(format!("{}^{}", base_vars[0], n));
                    } else {
                        variables.push(format!("{}", self));
                    }
                } else {
                    variables.push(format!("{}", self));
                }
            }
            _ => {
                variables.push(format!("{}", self));
            }
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
                Expression::Value(Value::Real(coeff))
            } else if (coeff - 1.0).abs() < f64::EPSILON {
                Expression::Variable(term_str)
            } else if (coeff + 1.0).abs() < f64::EPSILON {
                Expression::Neg(Box::new(Expression::Variable(term_str)))
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
            result = Expression::Add(Box::new(result), Box::new(term.clone()));
        }

        Ok(result)
    }
}
