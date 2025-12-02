use {
    crate::{constant::EPSILON, error::EvaluationError, expression::Expression},
    std::ops::{Add, Div, Mul, Neg, Rem, Sub},
};

impl Expression {
    pub fn complex_add(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r + b_r, a_i + b_i)
    }

    pub fn complex_sub(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r - b_r, a_i - b_i)
    }

    pub fn complex_mul(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
        Expression::Complex(a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r)
    }

    pub fn complex_div(
        a_r: f64,
        a_i: f64,
        b_r: f64,
        b_i: f64,
    ) -> Result<Expression, EvaluationError> {
        if b_r.abs() < EPSILON && b_i.abs() < EPSILON {
            return Err(EvaluationError::DivisionByZero);
        }

        let denominator = b_r * b_r + b_i * b_i;
        let real = (a_r * b_r + a_i * b_i) / denominator;
        let imag = (a_i * b_r - a_r * b_i) / denominator;

        Ok(Expression::Complex(real, imag))
    }

    pub fn complex_pow(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
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

    pub fn complex_sqrt(r: f64, i: f64) -> Expression {
        if r.abs() < EPSILON && i.abs() < EPSILON {
            return Expression::Complex(0.0, 0.0);
        }

        let magnitude = (r * r + i * i).sqrt();
        let phase = i.atan2(r);

        let sqrt_r = magnitude.sqrt();
        let half_theta = phase / 2.0;

        Expression::Complex(sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin())
    }

    pub fn complex_abs(r: f64, i: f64) -> f64 {
        (r * r + i * i).sqrt()
    }

    pub fn complex_exp(r: f64, i: f64) -> Expression {
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

            // Matrix * Matrix
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

            // Matrix * Vector
            (Expression::Matrix(a, rows, cols), Expression::Vector(b)) => {
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
    pub fn hadamard(self, rhs: Self) -> Result<Self, EvaluationError> {
        match (self, rhs) {
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
}
