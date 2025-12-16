use crate::error::EvaluationError;
use crate::expression::Expression;

pub fn add(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
    Expression::Complex(a_r + b_r, a_i + b_i)
}

pub fn sub(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
    Expression::Complex(a_r - b_r, a_i - b_i)
}

pub fn mul(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
    Expression::Complex(a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r)
}

pub fn div(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Result<Expression, EvaluationError> {
    if b_r.abs() < f64::EPSILON && b_i.abs() < f64::EPSILON {
        return Err(EvaluationError::DivisionByZero);
    }

    let denominator = b_r * b_r + b_i * b_i;
    let real = (a_r * b_r + a_i * b_i) / denominator;
    let imag = (a_i * b_r - a_r * b_i) / denominator;

    Ok(Expression::Complex(real, imag))
}

pub fn pow(a_r: f64, a_i: f64, b_r: f64, b_i: f64) -> Expression {
    let a_is_zero = a_r.abs() < f64::EPSILON && a_i.abs() < f64::EPSILON;
    let b_is_zero = b_r.abs() < f64::EPSILON && b_i.abs() < f64::EPSILON;
    let b_is_one = (b_r - 1.0).abs() < f64::EPSILON && b_i.abs() < f64::EPSILON;

    if a_is_zero {
        if b_r > 0.0 || (b_i.abs() >= f64::EPSILON && b_r < f64::EPSILON) {
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

    let real = if real.abs() < f64::EPSILON { 0.0 } else { real };
    let imag = if imag.abs() < f64::EPSILON { 0.0 } else { imag };

    Expression::Complex(real, imag)
}

pub fn sqrt(r: f64, i: f64) -> Expression {
    if r.abs() < f64::EPSILON && i.abs() < f64::EPSILON {
        return Expression::Complex(0.0, 0.0);
    }

    let magnitude = (r * r + i * i).sqrt();
    let phase = i.atan2(r);

    let sqrt_r = magnitude.sqrt();
    let half_theta = phase / 2.0;

    Expression::Complex(sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin())
}

pub fn abs(r: f64, i: f64) -> f64 {
    (r * r + i * i).sqrt()
}

pub fn exp(r: f64, i: f64) -> Expression {
    let exp_real = r.exp();
    Expression::Complex(exp_real * i.cos(), exp_real * i.sin())
}
