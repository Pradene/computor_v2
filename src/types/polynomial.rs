use {
    crate::{error::EvaluationError, expression::Expression},
    std::{
        collections::HashMap,
        ops::{Add, Neg},
    },
};

/// Collect polynomial coefficients from an expression for a given variable
/// Returns a map of degree -> coefficient expression
pub fn collect_coefficients(
    expr: &Expression,
    var: &str,
) -> Result<HashMap<i32, Expression>, EvaluationError> {
    let mut coefficients = HashMap::new();

    match expr {
        Expression::Real(n) => {
            coefficients.insert(0, Expression::Real(*n));
        }
        Expression::Complex(r, i) => {
            coefficients.insert(0, Expression::Complex(*r, *i));
        }
        Expression::Variable(name) if name == var => {
            coefficients.insert(1, Expression::Real(1.0));
        }
        Expression::Variable(name) => {
            return Err(EvaluationError::UnsupportedOperation(format!(
                "Unexpected variable '{}' in polynomial with variable '{}'",
                name, var
            )));
        }
        Expression::FunctionCall(name, _) => {
            let vars = expr.collect_variables();
            if vars.contains(var) {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Cannot solve equations with function calls containing the variable: {}",
                    name
                )));
            }
            return Err(EvaluationError::UnsupportedOperation(format!(
                "Function call {} should have been evaluated before solving",
                name
            )));
        }
        Expression::Paren(inner) => {
            coefficients = collect_coefficients(inner, var)?;
        }
        Expression::Add(left, right) => {
            let left_coeffs = collect_coefficients(left, var)?;
            let right_coeffs = collect_coefficients(right, var)?;

            merge_coefficients(&mut coefficients, left_coeffs, 1.0);
            merge_coefficients(&mut coefficients, right_coeffs, 1.0);
        }
        Expression::Sub(left, right) => {
            let left_coeffs = collect_coefficients(left, var)?;
            let right_coeffs = collect_coefficients(right, var)?;

            merge_coefficients(&mut coefficients, left_coeffs, 1.0);
            merge_coefficients(&mut coefficients, right_coeffs, -1.0);
        }
        Expression::Neg(inner) => {
            let inner_coeffs = collect_coefficients(inner, var)?;
            merge_coefficients(&mut coefficients, inner_coeffs, -1.0);
        }
        Expression::Mul(..) | Expression::Pow(..) => {
            let (coeff, degree) = extract_term(expr, var)?;
            coefficients.insert(degree, coeff);
        }
        _ => {
            return Err(EvaluationError::UnsupportedOperation(
                "Unsupported expression in polynomial".to_string(),
            ));
        }
    }

    Ok(coefficients)
}

/// Merge source coefficients into destination with a sign multiplier
fn merge_coefficients(
    dest: &mut HashMap<i32, Expression>,
    source: HashMap<i32, Expression>,
    sign: f64,
) {
    for (degree, coeff) in source {
        let adjusted_coeff = apply_sign(coeff, sign);

        dest.entry(degree)
            .and_modify(|existing| {
                if let Ok(sum) = existing.clone().add(adjusted_coeff.clone()) {
                    *existing = sum;
                }
            })
            .or_insert(adjusted_coeff);
    }
}

/// Apply a sign to a coefficient
fn apply_sign(coeff: Expression, sign: f64) -> Expression {
    if (sign - 1.0).abs() < f64::EPSILON {
        coeff
    } else {
        match coeff {
            Expression::Real(n) => Expression::Real(sign * n),
            Expression::Complex(r, i) => Expression::Complex(sign * r, sign * i),
            other => {
                if sign < 0.0 {
                    Expression::Neg(Box::new(other))
                } else {
                    other
                }
            }
        }
    }
}

/// Extract a single polynomial term (coefficient * variable^degree)
pub fn extract_term(expr: &Expression, var: &str) -> Result<(Expression, i32), EvaluationError> {
    match expr {
        Expression::Real(n) => Ok((Expression::Real(*n), 0)),
        Expression::Complex(r, i) => Ok((Expression::Complex(*r, *i), 0)),
        Expression::Variable(name) if name == var => Ok((Expression::Real(1.0), 1)),
        Expression::Paren(inner) => extract_term(inner, var),
        Expression::Neg(inner) => {
            let (coeff, degree) = extract_term(inner, var)?;
            let neg_coeff = coeff.neg()?;
            Ok((neg_coeff, degree))
        }
        Expression::Mul(left, right) => {
            let (left_coeff, left_degree) = extract_term(left, var)?;
            let (right_coeff, right_degree) = extract_term(right, var)?;
            let combined_coeff = left_coeff * right_coeff;
            Ok((combined_coeff?, left_degree + right_degree))
        }
        Expression::Pow(base, exp) => extract_power_term(base, exp, var),
        Expression::FunctionCall(_, _) => {
            let vars = expr.collect_variables();
            if vars.contains(var) {
                Err(EvaluationError::UnsupportedOperation(
                    "Cannot extract term from function containing variable".to_string(),
                ))
            } else {
                Err(EvaluationError::UnsupportedOperation(
                    "Function should have been evaluated".to_string(),
                ))
            }
        }
        _ => Err(EvaluationError::UnsupportedOperation(format!(
            "Cannot extract polynomial term from: {:?}",
            expr
        ))),
    }
}

/// Extract term from power expression
fn extract_power_term(
    base: &Expression,
    exp: &Expression,
    var: &str,
) -> Result<(Expression, i32), EvaluationError> {
    let exp_value = match exp {
        Expression::Real(e) => *e,
        _ => {
            return Err(EvaluationError::UnsupportedOperation(
                "Power exponent must be a real number constant".to_string(),
            ))
        }
    };

    if exp_value < 0.0 {
        return Err(EvaluationError::UnsupportedOperation(
            "Cannot solve equations with negative exponents".to_string(),
        ));
    }

    if exp_value != exp_value.floor() {
        return Err(EvaluationError::UnsupportedOperation(
            "Cannot solve equations with non-integer exponents".to_string(),
        ));
    }

    let exp_i32 = exp_value as i32;
    let base_unwrapped = match base {
        Expression::Paren(inner) => inner.as_ref(),
        other => other,
    };

    match base_unwrapped {
        Expression::Variable(name) if name == var => Ok((Expression::Real(1.0), exp_i32)),
        Expression::Mul(l, r) => extract_power_from_mul(l, r, var, exp_value, exp_i32),
        Expression::Neg(inner) => extract_power_from_neg(inner, var, exp_value, exp_i32),
        _ => Err(EvaluationError::UnsupportedOperation(
            "Invalid power base".to_string(),
        )),
    }
}

fn extract_power_from_mul(
    left: &Expression,
    right: &Expression,
    var: &str,
    exp_value: f64,
    exp_i32: i32,
) -> Result<(Expression, i32), EvaluationError> {
    match (left, right) {
        (Expression::Real(coeff), Expression::Variable(name))
        | (Expression::Variable(name), Expression::Real(coeff))
            if name == var =>
        {
            Ok((Expression::Real(coeff.powf(exp_value)), exp_i32))
        }
        (Expression::Complex(_, _), Expression::Variable(name))
        | (Expression::Variable(name), Expression::Complex(_, _))
            if name == var =>
        {
            Err(EvaluationError::UnsupportedOperation(
                "Cannot solve equations with complex coefficients raised to powers".to_string(),
            ))
        }
        _ => Err(EvaluationError::UnsupportedOperation(
            "Invalid power expression in multiplication".to_string(),
        )),
    }
}

fn extract_power_from_neg(
    inner: &Expression,
    var: &str,
    exp_value: f64,
    exp_i32: i32,
) -> Result<(Expression, i32), EvaluationError> {
    let inner_unwrapped = match inner {
        Expression::Paren(p) => p.as_ref(),
        other => other,
    };

    let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };

    match inner_unwrapped {
        Expression::Variable(name) if name == var => Ok((Expression::Real(sign), exp_i32)),
        Expression::Mul(l, r) => match (l.as_ref(), r.as_ref()) {
            (Expression::Real(coeff), Expression::Variable(name))
            | (Expression::Variable(name), Expression::Real(coeff))
                if name == var =>
            {
                Ok((Expression::Real(sign * coeff.powf(exp_value)), exp_i32))
            }
            _ => Err(EvaluationError::UnsupportedOperation(
                "Invalid power expression in negation".to_string(),
            )),
        },
        _ => Err(EvaluationError::UnsupportedOperation(
            "Invalid power expression".to_string(),
        )),
    }
}

/// Convert Expression coefficients to real f64 values
/// Returns None if any coefficient is complex
pub fn to_real_coefficients(coeffs: &HashMap<i32, Expression>) -> Option<HashMap<i32, f64>> {
    let mut real_coeffs = HashMap::new();
    for (deg, expr) in coeffs {
        match expr {
            Expression::Real(n) => {
                real_coeffs.insert(*deg, *n);
            }
            Expression::Complex(r, i) if i.abs() < f64::EPSILON => {
                real_coeffs.insert(*deg, *r);
            }
            _ => return None,
        }
    }
    Some(real_coeffs)
}

/// Get the degree of a polynomial (max degree with non-zero coefficient)
pub fn degree(coeffs: &HashMap<i32, Expression>) -> i32 {
    coeffs.keys().copied().max().unwrap_or(0)
}

/// Solve polynomial equation of degree 0 (constant)
pub fn solve_constant() -> Vec<Expression> {
    vec![]
}

/// Solve linear equation: ax + b = 0
pub fn solve_linear(coeffs: &HashMap<i32, f64>) -> Result<Vec<Expression>, EvaluationError> {
    let a = coeffs.get(&1).copied().unwrap_or(0.0);
    let b = coeffs.get(&0).copied().unwrap_or(0.0);

    if a.abs() < f64::EPSILON {
        return Ok(solve_constant());
    }

    Ok(vec![Expression::Real(-b / a)])
}

/// Solve quadratic equation: ax^2 + bx + c = 0
pub fn solve_quadratic(coeffs: &HashMap<i32, f64>) -> Result<Vec<Expression>, EvaluationError> {
    let a = coeffs.get(&2).copied().unwrap_or(0.0);
    let b = coeffs.get(&1).copied().unwrap_or(0.0);
    let c = coeffs.get(&0).copied().unwrap_or(0.0);

    if a.abs() < f64::EPSILON {
        return solve_linear(coeffs);
    }

    let discriminant = b * b - 4.0 * a * c;

    if discriminant > 0.0 {
        let sqrt_d = discriminant.sqrt();
        // Use numerically stable formula
        let x1 = if b >= 0.0 {
            (-b - sqrt_d) / (2.0 * a)
        } else {
            (-b + sqrt_d) / (2.0 * a)
        };
        let x2 = c / (a * x1);
        Ok(vec![Expression::Real(x1), Expression::Real(x2)])
    } else if discriminant.abs() < f64::EPSILON {
        let x = -b / (2.0 * a);
        Ok(vec![Expression::Real(x), Expression::Real(x)])
    } else {
        let real = -b / (2.0 * a);
        let imag = (-discriminant).sqrt() / (2.0 * a);
        Ok(vec![
            Expression::Complex(real, imag),
            Expression::Complex(real, -imag),
        ])
    }
}
