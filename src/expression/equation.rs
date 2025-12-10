use {
    crate::{constant::EPSILON, error::EvaluationError, expression::Expression},
    std::collections::HashMap,
    std::ops::{Add, Mul, Neg},
};

impl Expression {
    pub fn find_roots(&self) -> Result<Vec<Expression>, EvaluationError> {
        let variables = self.collect_variables();

        if variables.is_empty() {
            return self.solve_degree_0();
        }

        if variables.len() > 1 {
            return Err(EvaluationError::UnsupportedOperation(format!(
                "Cannot solve equations with multiple variables: {:?}",
                variables
            )));
        }

        let variable = variables.iter().next().unwrap().clone();
        let coefficients = self.collect_polynomial_coefficients(&variable)?;

        // Check if all coefficients are real
        let all_real = coefficients.values().all(|c| c.is_real());

        if !all_real {
            return Err(EvaluationError::UnsupportedOperation(
                "Cannot solve polynomial equations with complex coefficients".to_string(),
            ));
        }

        // Convert to HashMap<i32, f64>
        let coefficients: HashMap<i32, f64> = coefficients
            .iter()
            .filter_map(|(deg, expr)| match expr {
                Expression::Real(n) => Some((*deg, *n)),
                Expression::Complex(r, i) if i.abs() < EPSILON => Some((*deg, *r)),
                _ => None,
            })
            .collect();

        let degree = coefficients.keys().max().copied().unwrap_or(0);

        if degree < 0 {
            return Err(EvaluationError::InvalidOperation(
                "Invalid polynomial degree".to_string(),
            ));
        }

        match degree {
            0 => self.solve_degree_0(),
            1 => self.solve_degree_1(&coefficients),
            2 => self.solve_degree_2(&coefficients),
            _ => Err(EvaluationError::UnsupportedOperation(format!(
                "Cannot solve polynomial equations of degree {}",
                degree
            ))),
        }
    }

    fn solve_degree_0(&self) -> Result<Vec<Expression>, EvaluationError> {
        Ok(vec![])
    }

    fn solve_degree_1(
        &self,
        coefficients: &HashMap<i32, f64>,
    ) -> Result<Vec<Expression>, EvaluationError> {
        let a = coefficients.get(&1).copied().unwrap_or(0.0);
        let b = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < EPSILON {
            return self.solve_degree_0();
        }

        let solution = -b / a;
        Ok(vec![Expression::Real(solution)])
    }

    fn solve_degree_2(
        &self,
        coefficients: &HashMap<i32, f64>,
    ) -> Result<Vec<Expression>, EvaluationError> {
        let a = coefficients.get(&2).copied().unwrap_or(0.0);
        let b = coefficients.get(&1).copied().unwrap_or(0.0);
        let c = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < EPSILON {
            return self.solve_degree_1(coefficients);
        }

        let discriminant = b * b - 4.0 * a * c;

        Ok(if discriminant > EPSILON {
            let sqrt_discriminant = discriminant.sqrt();
            let x1 = (-b + sqrt_discriminant) / (2.0 * a);
            let x2 = (-b - sqrt_discriminant) / (2.0 * a);

            let x1 = if x1.abs() < EPSILON { 0.0 } else { x1 };
            let x2 = if x2.abs() < EPSILON { 0.0 } else { x2 };

            vec![Expression::Real(x1), Expression::Real(x2)]
        } else if discriminant.abs() < EPSILON {
            let x = -b / (2.0 * a);
            let x = if x.abs() < EPSILON { 0.0 } else { x };

            vec![Expression::Real(x), Expression::Real(x)]
        } else {
            let real = -b / (2.0 * a);
            let imag = (-discriminant).sqrt() / (2.0 * a);

            let real = if real.abs() < EPSILON { 0.0 } else { real };
            let imag = if imag.abs() < EPSILON { 0.0 } else { imag };

            vec![
                Expression::Complex(real, imag),
                Expression::Complex(real, -imag),
            ]
        })
    }

    fn collect_polynomial_coefficients(
        &self,
        variable: &str,
    ) -> Result<HashMap<i32, Expression>, EvaluationError> {
        let mut coefficients = HashMap::new();

        match self {
            Expression::Real(n) => {
                coefficients.insert(0, Expression::Real(*n));
            }
            Expression::Complex(r, i) => {
                coefficients.insert(0, Expression::Complex(*r, *i));
            }
            Expression::Variable(name) if name == variable => {
                coefficients.insert(1, Expression::Real(1.0));
            }
            Expression::Variable(name) => {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Unexpected variable: {}",
                    name
                )));
            }
            Expression::FunctionCall(name, _) => {
                let vars = self.collect_variables();
                if vars.contains(&variable.to_string()) {
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
                coefficients = inner.collect_polynomial_coefficients(variable)?;
            }
            Expression::Add(left, right) => {
                let left_coeffs = left.collect_polynomial_coefficients(variable)?;
                let right_coeffs = right.collect_polynomial_coefficients(variable)?;

                for (degree, coeff) in left_coeffs {
                    coefficients.insert(degree, coeff);
                }
                for (degree, coeff) in right_coeffs {
                    coefficients
                        .entry(degree)
                        .and_modify(|c| {
                            if let Ok(result) = c.clone().add(coeff.clone()) {
                                *c = result;
                            }
                        })
                        .or_insert(coeff);
                }
            }
            Expression::Sub(left, right) => {
                let left_coeffs = left.collect_polynomial_coefficients(variable)?;
                let right_coeffs = right.collect_polynomial_coefficients(variable)?;

                for (degree, coeff) in left_coeffs {
                    coefficients.insert(degree, coeff);
                }
                for (degree, coeff) in right_coeffs {
                    let neg_coeff = coeff.neg()?;
                    coefficients
                        .entry(degree)
                        .and_modify(|c| {
                            if let Ok(result) = c.clone().add(neg_coeff.clone()) {
                                *c = result;
                            }
                        })
                        .or_insert(neg_coeff);
                }
            }
            Expression::Mul(..) => {
                let (coeff, degree) = self.extract_polynomial_term(variable)?;
                coefficients.insert(degree, coeff);
            }
            Expression::Pow(..) => {
                let (coeff, degree) = self.extract_polynomial_term(variable)?;
                coefficients.insert(degree, coeff);
            }
            Expression::Neg(inner) => {
                let inner_coeffs = inner.collect_polynomial_coefficients(variable)?;
                for (degree, coeff) in inner_coeffs {
                    coefficients.insert(degree, coeff.neg()?);
                }
            }
            _ => {
                return Err(EvaluationError::UnsupportedOperation(
                    "Unsupported expression in polynomial".to_string(),
                ));
            }
        }

        Ok(coefficients)
    }

    fn extract_polynomial_term(
        &self,
        variable: &str,
    ) -> Result<(Expression, i32), EvaluationError> {
        match self {
            Expression::Real(n) => Ok((Expression::Real(*n), 0)),
            Expression::Complex(r, i) => Ok((Expression::Complex(*r, *i), 0)),
            Expression::Variable(name) if name == variable => Ok((Expression::Real(1.0), 1)),
            Expression::Paren(inner) => inner.extract_polynomial_term(variable),
            Expression::Neg(inner) => {
                let (coeff, degree) = inner.extract_polynomial_term(variable)?;
                let neg_coeff = coeff.neg()?;
                Ok((neg_coeff, degree))
            }
            Expression::Mul(left, right) => {
                let (left_coeff, left_degree) = left.extract_polynomial_term(variable)?;
                let (right_coeff, right_degree) = right.extract_polynomial_term(variable)?;
                let combined_coeff = left_coeff.mul(right_coeff)?;
                Ok((combined_coeff, left_degree + right_degree))
            }
            Expression::Pow(left, right) => {
                let exp_value = match right.as_ref() {
                    Expression::Real(exp) => *exp,
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
                let unwrapped_left = match left.as_ref() {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };

                match unwrapped_left {
                    Expression::Variable(name) if name == variable => {
                        Ok((Expression::Real(1.0), exp_i32))
                    }
                    Expression::Mul(l, r) => {
                        if let (Expression::Real(coeff), Expression::Variable(name)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Ok((Expression::Real(coeff.powf(exp_value)), exp_i32));
                            }
                        }
                        if let (Expression::Variable(name), Expression::Real(coeff)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Ok((Expression::Real(coeff.powf(exp_value)), exp_i32));
                            }
                        }
                        if let (Expression::Complex(_, _), Expression::Variable(name)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Err(EvaluationError::UnsupportedOperation(
                                    "Cannot solve equations with complex coefficients raised to powers"
                                        .to_string(),
                                ));
                            }
                        }
                        if let (Expression::Variable(name), Expression::Complex(_, _)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Err(EvaluationError::UnsupportedOperation(
                                    "Cannot solve equations with complex coefficients raised to powers"
                                        .to_string(),
                                ));
                            }
                        }
                        Err(EvaluationError::UnsupportedOperation(
                            "Invalid power expression in multiplication".to_string(),
                        ))
                    }
                    Expression::Neg(inner) => {
                        let unwrapped_inner = match inner.as_ref() {
                            Expression::Paren(p) => p.as_ref(),
                            other => other,
                        };

                        match unwrapped_inner {
                            Expression::Variable(name) if name == variable => {
                                let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                Ok((Expression::Real(sign), exp_i32))
                            }
                            Expression::Mul(l, r) => {
                                if let (Expression::Real(coeff), Expression::Variable(name)) =
                                    (l.as_ref(), r.as_ref())
                                {
                                    if name == variable {
                                        let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                        return Ok((
                                            Expression::Real(sign * coeff.powf(exp_value)),
                                            exp_i32,
                                        ));
                                    }
                                }
                                if let (Expression::Variable(name), Expression::Real(coeff)) =
                                    (l.as_ref(), r.as_ref())
                                {
                                    if name == variable {
                                        let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                        return Ok((
                                            Expression::Real(sign * coeff.powf(exp_value)),
                                            exp_i32,
                                        ));
                                    }
                                }
                                Err(EvaluationError::UnsupportedOperation(
                                    "Invalid power expression in negation".to_string(),
                                ))
                            }
                            _ => Err(EvaluationError::UnsupportedOperation(
                                "Invalid power expression".to_string(),
                            )),
                        }
                    }
                    _ => Err(EvaluationError::UnsupportedOperation(
                        "Invalid power base".to_string(),
                    )),
                }
            }
            Expression::FunctionCall(_, _) => {
                let vars = self.collect_variables();
                if vars.contains(&variable.to_string()) {
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
                "Cannot extract term from: {:?}",
                self
            ))),
        }
    }
}
