use crate::{error::EvaluationError, expression::Expression};

/// Matrix addition
pub fn add(
    a: Vec<Expression>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<Expression>,
    b_rows: usize,
    b_cols: usize,
) -> Result<Expression, EvaluationError> {
    if a_rows != b_rows || a_cols != b_cols {
        return Err(EvaluationError::InvalidOperation(format!(
            "Matrix addition: both matrices must have the same dimensions (got {}×{} and {}×{})",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    let result: Result<Vec<Expression>, _> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.clone() + y.clone())
        .collect();

    Ok(Expression::Matrix(result?, a_rows, a_cols))
}

/// Matrix subtraction
pub fn sub(
    a: Vec<Expression>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<Expression>,
    b_rows: usize,
    b_cols: usize,
) -> Result<Expression, EvaluationError> {
    if a_rows != b_rows || a_cols != b_cols {
        return Err(EvaluationError::InvalidOperation(format!(
            "Matrix subtraction: both matrices must have the same dimensions (got {}×{} and {}×{})",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    let result: Result<Vec<Expression>, _> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| x.clone() - y.clone())
        .collect();

    Ok(Expression::Matrix(result?, a_rows, a_cols))
}

/// Scalar-Matrix multiplication
pub fn scalar_mul(
    scalar: Expression,
    data: Vec<Expression>,
    rows: usize,
    cols: usize,
) -> Result<Expression, EvaluationError> {
    match scalar {
        Expression::Real(s) => {
            let result: Result<Vec<Expression>, _> = data
                .iter()
                .map(|x| x.clone() * Expression::Real(s))
                .collect();
            Ok(Expression::Matrix(result?, rows, cols))
        }
        Expression::Complex(r, i) => {
            let result: Result<Vec<Expression>, _> = data
                .iter()
                .map(|x| x.clone() * Expression::Complex(r, i))
                .collect();
            Ok(Expression::Matrix(result?, rows, cols))
        }
        _ => Err(EvaluationError::InvalidOperation(
            "Scalar multiplication requires a scalar value".to_string(),
        )),
    }
}

/// Matrix-Matrix multiplication
pub fn mul(
    a: Vec<Expression>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<Expression>,
    b_rows: usize,
    b_cols: usize,
) -> Result<Expression, EvaluationError> {
    if a_cols != b_rows {
        return Err(EvaluationError::InvalidOperation(format!(
            "Matrix multiplication: left matrix columns ({}) must equal right matrix rows ({})",
            a_cols, b_rows
        )));
    }

    let mut result = Vec::with_capacity(a_rows * b_cols);

    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut sum = Expression::Complex(0.0, 0.0);

            for k in 0..a_cols {
                let left = a[i * a_cols + k].clone();
                let right = b[k * b_cols + j].clone();

                let product = (left * right)?;
                sum = (sum + product)?.simplify()?;
            }

            result.push(sum);
        }
    }

    Ok(Expression::Matrix(result, a_rows, b_cols))
}

/// Matrix-Vector multiplication
pub fn mul_vector(
    matrix: Vec<Expression>,
    rows: usize,
    cols: usize,
    vector: Vec<Expression>,
) -> Result<Expression, EvaluationError> {
    if cols != vector.len() {
        return Err(EvaluationError::InvalidOperation(format!(
            "Matrix-Vector multiplication: matrix has {} columns but vector has {} elements",
            cols,
            vector.len()
        )));
    }

    let mut result = Vec::with_capacity(rows);

    for i in 0..rows {
        let mut sum = Expression::Complex(0.0, 0.0);
        let row_start = i * cols;

        for k in 0..cols {
            let left = matrix[row_start + k].clone();
            let right = vector[k].clone();
            let product = (left * right)?;
            sum = (sum + product)?;
        }

        result.push(sum.simplify()?);
    }

    Ok(Expression::Vector(result))
}

/// Matrix-Scalar division
pub fn scalar_div(
    data: Vec<Expression>,
    rows: usize,
    cols: usize,
    scalar: Expression,
) -> Result<Expression, EvaluationError> {
    if scalar.is_zero() {
        return Err(EvaluationError::DivisionByZero);
    }

    let result: Result<Vec<Expression>, _> =
        data.iter().map(|x| x.clone() / scalar.clone()).collect();

    Ok(Expression::Matrix(result?, rows, cols))
}

/// Matrix negation
pub fn neg(data: Vec<Expression>, rows: usize, cols: usize) -> Result<Expression, EvaluationError> {
    let result: Result<Vec<Expression>, _> = data.into_iter().map(|x| -x).collect();

    Ok(Expression::Matrix(result?, rows, cols))
}

/// Hadamard (element-wise) product
pub fn hadamard(
    a: Vec<Expression>,
    a_rows: usize,
    a_cols: usize,
    b: Vec<Expression>,
    b_rows: usize,
    b_cols: usize,
) -> Result<Expression, EvaluationError> {
    if a_rows != b_rows || a_cols != b_cols {
        return Err(EvaluationError::InvalidOperation(format!(
            "Hadamard product: matrices must have the same dimensions (got {}×{} and {}×{})",
            a_rows, a_cols, b_rows, b_cols
        )));
    }

    let result: Result<Vec<Expression>, _> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x.clone() * y.clone())?.simplify())
        .collect();

    Ok(Expression::Matrix(result?, a_rows, a_cols))
}

/// Matrix transpose
pub fn transpose(data: Vec<Expression>, rows: usize, cols: usize) -> Expression {
    let mut result = Vec::with_capacity(rows * cols);

    for j in 0..cols {
        for i in 0..rows {
            result.push(data[i * cols + j].clone());
        }
    }

    Expression::Matrix(result, cols, rows)
}

/// Matrix determinant (for 2x2 and 3x3 matrices)
pub fn determinant(
    data: &[Expression],
    rows: usize,
    cols: usize,
) -> Result<Expression, EvaluationError> {
    if rows != cols {
        return Err(EvaluationError::InvalidOperation(
            "Determinant is only defined for square matrices".to_string(),
        ));
    }

    match rows {
        1 => Ok(data[0].clone()),
        2 => {
            // det = a*d - b*c
            let a = data[0].clone();
            let b = data[1].clone();
            let c = data[2].clone();
            let d = data[3].clone();

            (a * d)? - (b * c)?
        }
        3 => {
            // Expansion by first row
            let a = data[0].clone();
            let b = data[1].clone();
            let c = data[2].clone();

            let minor_a = vec![
                data[4].clone(),
                data[5].clone(),
                data[7].clone(),
                data[8].clone(),
            ];
            let minor_b = vec![
                data[3].clone(),
                data[5].clone(),
                data[6].clone(),
                data[8].clone(),
            ];
            let minor_c = vec![
                data[3].clone(),
                data[4].clone(),
                data[6].clone(),
                data[7].clone(),
            ];

            let det_a = determinant(&minor_a, 2, 2)?;
            let det_b = determinant(&minor_b, 2, 2)?;
            let det_c = determinant(&minor_c, 2, 2)?;

            ((a * det_a)? - (b * det_b)?)? + (c * det_c)?
        }
        _ => Err(EvaluationError::InvalidOperation(
            "Determinant calculation is only implemented for 1×1, 2×2, and 3×3 matrices"
                .to_string(),
        )),
    }
}

/// Create identity matrix
pub fn identity(size: usize) -> Expression {
    let mut data = vec![Expression::Real(0.0); size * size];

    for i in 0..size {
        data[i * size + i] = Expression::Real(1.0);
    }

    Expression::Matrix(data, size, size)
}

/// Matrix trace (sum of diagonal elements)
pub fn trace(data: &[Expression], rows: usize, cols: usize) -> Result<Expression, EvaluationError> {
    if rows != cols {
        return Err(EvaluationError::InvalidOperation(
            "Trace is only defined for square matrices".to_string(),
        ));
    }

    let mut sum = Expression::Real(0.0);

    for i in 0..rows {
        sum = (sum + data[i * cols + i].clone())?;
    }

    Ok(sum)
}
