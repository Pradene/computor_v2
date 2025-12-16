use crate::{error::EvaluationError, expression::Expression, types::complex};

/// Vector addition
pub fn add(a: Vec<Expression>, b: Vec<Expression>) -> Result<Expression, EvaluationError> {
    if a.len() != b.len() {
        return Err(EvaluationError::InvalidOperation(format!(
            "Vector addition: both vectors must have the same dimension (got {} and {})",
            a.len(),
            b.len()
        )));
    }

    let result: Result<Vec<Expression>, _> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect();

    Ok(Expression::Vector(result?))
}

/// Vector subtraction
pub fn sub(a: Vec<Expression>, b: Vec<Expression>) -> Result<Expression, EvaluationError> {
    if a.len() != b.len() {
        return Err(EvaluationError::InvalidOperation(format!(
            "Vector subtraction: both vectors must have the same dimension (got {} and {})",
            a.len(),
            b.len()
        )));
    }

    let result: Result<Vec<Expression>, _> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.clone() - y.clone())
        .collect();

    Ok(Expression::Vector(result?))
}

/// Scalar-Vector multiplication
pub fn scalar_mul(
    scalar: Expression,
    vector: Vec<Expression>,
) -> Result<Expression, EvaluationError> {
    match scalar {
        Expression::Real(s) => {
            let result: Result<Vec<Expression>, _> = vector
                .iter()
                .map(|x| x.clone() * Expression::Real(s))
                .collect();
            Ok(Expression::Vector(result?))
        }
        Expression::Complex(r, i) => {
            let result: Result<Vec<Expression>, _> = vector
                .iter()
                .map(|x| x.clone() * Expression::Complex(r, i))
                .collect();
            Ok(Expression::Vector(result?))
        }
        _ => Err(EvaluationError::InvalidOperation(
            "Scalar multiplication requires a scalar value".to_string(),
        )),
    }
}

/// Vector-Scalar division
pub fn scalar_div(
    vector: Vec<Expression>,
    scalar: Expression,
) -> Result<Expression, EvaluationError> {
    if scalar.is_zero() {
        return Err(EvaluationError::DivisionByZero);
    }

    let result: Result<Vec<Expression>, _> =
        vector.iter().map(|x| x.clone() / scalar.clone()).collect();

    Ok(Expression::Vector(result?))
}

/// Vector negation
pub fn neg(vector: Vec<Expression>) -> Result<Expression, EvaluationError> {
    let result: Result<Vec<Expression>, _> = vector.into_iter().map(|x| -x).collect();

    Ok(Expression::Vector(result?))
}

/// Dot product of two vectors
pub fn dot(a: &[Expression], b: &[Expression]) -> Result<Expression, EvaluationError> {
    if a.len() != b.len() {
        return Err(EvaluationError::InvalidOperation(format!(
            "Dot product: vectors must have the same dimension (got {} and {})",
            a.len(),
            b.len()
        )));
    }

    let mut sum = Expression::Complex(0.0, 0.0);
    for (x, y) in a.iter().zip(b.iter()) {
        let product = (x.clone() * y.clone())?;
        sum = (sum + product)?.simplify()?;
    }

    Ok(sum)
}

/// Cross product of two 3D vectors
pub fn cross(a: &[Expression], b: &[Expression]) -> Result<Expression, EvaluationError> {
    if a.len() != 3 || b.len() != 3 {
        return Err(EvaluationError::InvalidOperation(
            "Cross product is only defined for 3D vectors".to_string(),
        ));
    }

    let i = ((a[1].clone() * b[2].clone())? - (a[2].clone() * b[1].clone())?)?;
    let j = ((a[2].clone() * b[0].clone())? - (a[0].clone() * b[2].clone())?)?;
    let k = ((a[0].clone() * b[1].clone())? - (a[1].clone() * b[0].clone())?)?;

    Ok(Expression::Vector(vec![i, j, k]))
}

/// Vector magnitude (Euclidean norm)
pub fn magnitude(v: &[Expression]) -> Result<Expression, EvaluationError> {
    let mut sum_squares = Expression::Real(0.0);

    for elem in v {
        let squared = (elem.clone() * elem.clone())?;
        sum_squares = (sum_squares + squared)?;
    }

    // Return sqrt of sum
    match sum_squares {
        Expression::Real(val) => Ok(Expression::Real(val.sqrt())),
        Expression::Complex(r, i) => Ok(complex::sqrt(r, i)),
        _ => Err(EvaluationError::InvalidOperation(
            "Cannot compute magnitude".to_string(),
        )),
    }
}

/// Normalize a vector to unit length
pub fn normalize(v: Vec<Expression>) -> Result<Expression, EvaluationError> {
    let mag = magnitude(&v)?;

    if mag.is_zero() {
        return Err(EvaluationError::InvalidOperation(
            "Cannot normalize zero vector".to_string(),
        ));
    }

    scalar_div(v, mag)
}
