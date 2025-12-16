use {
    crate::{
        error::EvaluationError,
        expression::Expression,
        types::{complex, matrix, vector},
    },
    std::ops::{Add, Div, Mul, Neg, Rem, Sub},
};

impl Add for Expression {
    type Output = Result<Self, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a + b)),

            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(complex::add(a_r, a_i, b_r, b_i))
            }

            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(complex::add(a, 0.0, b_r, b_i))
            }

            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(complex::add(a_r, a_i, b, 0.0))
            }

            (Expression::Vector(a), Expression::Vector(b)) => vector::add(a, b),

            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                matrix::add(a, a_rows, a_cols, b, b_rows, b_cols)
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
                Ok(complex::sub(a_r, a_i, b_r, b_i))
            }

            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(complex::sub(a, 0.0, b_r, b_i))
            }

            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(complex::sub(a_r, a_i, b, 0.0))
            }

            (Expression::Vector(a), Expression::Vector(b)) => vector::sub(a, b),

            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                matrix::sub(a, a_rows, a_cols, b, b_rows, b_cols)
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
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a * b)),

            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(complex::mul(a_r, a_i, b_r, b_i))
            }

            (Expression::Real(a), Expression::Complex(b_r, b_i)) => {
                Ok(complex::mul(a, 0.0, b_r, b_i))
            }

            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(complex::mul(a_r, a_i, b, 0.0))
            }

            // Scalar * Vector
            (Expression::Real(s), Expression::Vector(v))
            | (Expression::Vector(v), Expression::Real(s)) => {
                vector::scalar_mul(Expression::Real(s), v)
            }

            (Expression::Complex(r, i), Expression::Vector(v))
            | (Expression::Vector(v), Expression::Complex(r, i)) => {
                vector::scalar_mul(Expression::Complex(r, i), v)
            }

            // Scalar * Matrix
            (Expression::Real(s), Expression::Matrix(data, rows, cols))
            | (Expression::Matrix(data, rows, cols), Expression::Real(s)) => {
                matrix::scalar_mul(Expression::Real(s), data, rows, cols)
            }

            (Expression::Complex(r, i), Expression::Matrix(data, rows, cols))
            | (Expression::Matrix(data, rows, cols), Expression::Complex(r, i)) => {
                matrix::scalar_mul(Expression::Complex(r, i), data, rows, cols)
            }

            // Matrix * Matrix
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                matrix::mul(a, a_rows, a_cols, b, b_rows, b_cols)
            }

            // Matrix * Vector
            (Expression::Matrix(a, rows, cols), Expression::Vector(b)) => {
                matrix::mul_vector(a, rows, cols, b)
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

impl Div for Expression {
    type Output = Result<Self, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a / b)),

            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                complex::div(a_r, a_i, b_r, b_i)
            }

            (Expression::Real(a), Expression::Complex(b_r, b_i)) => complex::div(a, 0.0, b_r, b_i),

            (Expression::Complex(a_r, a_i), Expression::Real(b)) => complex::div(a_r, a_i, b, 0.0),

            (Expression::Vector(v), Expression::Real(s)) => {
                vector::scalar_div(v, Expression::Real(s))
            }

            (Expression::Vector(v), Expression::Complex(r, i)) => {
                vector::scalar_div(v, Expression::Complex(r, i))
            }

            (Expression::Matrix(data, rows, cols), Expression::Real(s)) => {
                matrix::scalar_div(data, rows, cols, Expression::Real(s))
            }

            (Expression::Matrix(data, rows, cols), Expression::Complex(r, i)) => {
                matrix::scalar_div(data, rows, cols, Expression::Complex(r, i))
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
            return Err(EvaluationError::DivisionByZero);
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
            Expression::Real(n) => Ok(Expression::Real(-n)),
            Expression::Complex(r, i) => Ok(Expression::Complex(-r, -i)),
            Expression::Vector(v) => vector::neg(v),
            Expression::Matrix(data, rows, cols) => matrix::neg(data, rows, cols),
            // -(a + b) = (-a) + (-b) = -a - b
            Expression::Add(left, right) => (*left).neg()? - *right,
            // -(a - b) = -a + b = b - a
            Expression::Sub(left, right) => *right - *left,
            // -(a * b) = (-a) * b
            Expression::Mul(left, right) => (*left).neg()? * *right,
            // Double negative: -(-x) = x
            Expression::Neg(inner) => Ok(*inner),
            _ => Ok(Expression::Neg(Box::new(self))),
        }
    }
}

impl Expression {
    pub fn hadamard(self, rhs: Self) -> Result<Self, EvaluationError> {
        match (self, rhs) {
            (Expression::Matrix(a, a_rows, a_cols), Expression::Matrix(b, b_rows, b_cols)) => {
                matrix::hadamard(a, a_rows, a_cols, b, b_rows, b_cols)
            }

            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot hadamard product {} and {}: incompatible types",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::Hadamard(
                Box::new(left),
                Box::new(right.clone()),
            )),
        }
    }

    pub fn pow(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a.powf(b))),
            (Expression::Complex(a_r, a_i), Expression::Real(b)) => {
                Ok(complex::pow(a_r, a_i, b, 0.0))
            }
            (Expression::Complex(a_r, a_i), Expression::Complex(b_r, b_i)) => {
                Ok(complex::pow(a_r, a_i, b_r, b_i))
            }
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
}
