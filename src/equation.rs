use {
    crate::{error::EvaluationError, expression::Expression, EPSILON},
    std::collections::HashMap,
    std::fmt,
};

#[derive(Debug, Clone, PartialEq)]
pub enum EquationSolution {
    /// No solution exists (e.g., x + 1 = x + 2)
    NoSolution,

    /// Infinitely many solutions (e.g., 5 = 5, x + 1 = x + 1)
    Infinite,

    /// Finite number of solutions
    Finite {
        variable: String,
        solutions: Vec<Expression>,
    },
}

impl fmt::Display for EquationSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EquationSolution::NoSolution => {
                write!(f, "No solution")?;
            }
            EquationSolution::Infinite => {
                write!(f, "Infinite solutions (identity)")?;
            }
            EquationSolution::Finite {
                variable,
                solutions,
            } => {
                if solutions.is_empty() {
                    write!(f, "No solution")?;
                } else if solutions.len() == 1 {
                    write!(f, "{} = {}", variable, solutions[0])?;
                } else {
                    write!(
                        f,
                        "{} = {}\n{} = {}",
                        variable, solutions[0], variable, solutions[1]
                    )?;
                }
            }
        };

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Equation {
    expression: Expression,
}

impl Equation {
    pub fn new(expression: Expression) -> Self {
        Equation { expression }
    }

    pub fn solve(&self) -> Result<EquationSolution, EvaluationError> {
        // Extract variables from the equation
        let variables = self.extract_variables();

        if variables.is_empty() {
            // No variables - check if equation is satisfied
            return Ok(self.solve_constant_equation());
        }

        if variables.len() > 1 {
            return Err(EvaluationError::UnsupportedOperation(format!(
                "Cannot solve equations with multiple variables: {:?}",
                variables
            )));
        }

        let variable = variables[0].clone();

        // Extract polynomial coefficients
        let coefficients = self.extract_polynomial_coefficients(&variable)?;

        // Find the highest degree
        let degree = coefficients.keys().max().copied().unwrap_or(0);

        if degree < 0 {
            return Err(EvaluationError::InvalidOperation(
                "Invalid polynomial degree".to_string(),
            ));
        }

        // Solve based on degree
        let solution = match degree {
            0 => self.solve_degree_0(&coefficients),
            1 => self.solve_degree_1(variable, &coefficients),
            2 => self.solve_degree_2(variable, &coefficients),
            _ => {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Cannot solve polynomial equations of degree {}",
                    degree
                )))
            }
        };

        Ok(solution)
    }

    fn solve_constant_equation(&self) -> EquationSolution {
        // Check if the constant expression is zero
        let is_zero = match &self.expression {
            Expression::Real(n) => n.abs() < EPSILON,
            Expression::Complex(r, i) => r.abs() < EPSILON && i.abs() < EPSILON,
            _ => false,
        };

        if is_zero {
            EquationSolution::Infinite
        } else {
            EquationSolution::NoSolution
        }
    }

    fn solve_degree_0(&self, coefficients: &HashMap<i32, f64>) -> EquationSolution {
        let c = coefficients.get(&0).copied().unwrap_or(0.0);

        if c.abs() < EPSILON {
            // 0 = 0, infinite solutions
            EquationSolution::Infinite
        } else {
            // c = 0 where c ≠ 0, no solution
            EquationSolution::NoSolution
        }
    }

    fn solve_degree_1(
        &self,
        variable: String,
        coefficients: &HashMap<i32, f64>,
    ) -> EquationSolution {
        // ax + b = 0
        // x = -b/a
        let a = coefficients.get(&1).copied().unwrap_or(0.0);
        let b = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < EPSILON {
            return self.solve_degree_0(coefficients);
        }

        let solution = -b / a;
        EquationSolution::Finite {
            variable,
            solutions: vec![Expression::Real(solution)],
        }
    }

    fn solve_degree_2(
        &self,
        variable: String,
        coefficients: &HashMap<i32, f64>,
    ) -> EquationSolution {
        // ax² + bx + c = 0
        let a = coefficients.get(&2).copied().unwrap_or(0.0);
        let b = coefficients.get(&1).copied().unwrap_or(0.0);
        let c = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < EPSILON {
            return self.solve_degree_1(variable, coefficients);
        }

        // Calculate discriminant: Δ = b² - 4ac
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > EPSILON {
            // Two real solutions
            let sqrt_discriminant = discriminant.sqrt();
            let x1 = (-b + sqrt_discriminant) / (2.0 * a);
            let x2 = (-b - sqrt_discriminant) / (2.0 * a);

            let x1 = if x1.abs() < EPSILON { 0.0 } else { x1 };
            let x2 = if x2.abs() < EPSILON { 0.0 } else { x2 };

            EquationSolution::Finite {
                variable,
                solutions: vec![Expression::Real(x1), Expression::Real(x2)],
            }
        } else if discriminant.abs() < EPSILON {
            // One real solution (double root)
            let x = -b / (2.0 * a);
            let x = if x.abs() < EPSILON { 0.0 } else { x };

            EquationSolution::Finite {
                variable,
                solutions: vec![Expression::Real(x)],
            }
        } else {
            // Two complex solutions
            let real = -b / (2.0 * a);
            let imag = (-discriminant).sqrt() / (2.0 * a);

            // Normalize the real part to avoid -0.0
            let real = if real.abs() < EPSILON { 0.0 } else { real };
            let imag = if imag.abs() < EPSILON { 0.0 } else { imag };

            EquationSolution::Finite {
                variable,
                solutions: vec![
                    Expression::Complex(real, imag),
                    Expression::Complex(real, -imag),
                ],
            }
        }
    }

    fn extract_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();
        Self::collect_variables(&self.expression, &mut variables);
        variables
    }

    fn collect_variables(expression: &Expression, variables: &mut Vec<String>) {
        match expression {
            Expression::Variable(name) => {
                // Only add if not already in the list
                if !variables.contains(name) {
                    variables.push(name.clone());
                }
            }
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::MatMul(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                Self::collect_variables(left, variables);
                Self::collect_variables(right, variables);
            }
            Expression::Neg(inner) => {
                Self::collect_variables(inner, variables);
            }
            Expression::FunctionCall(fc) => {
                // Collect variables from function arguments
                for arg in &fc.args {
                    Self::collect_variables(arg, variables);
                }
            }
            // Don't collect anything from Real, Complex, or other literal types
            _ => {}
        }
    }

    fn extract_polynomial_coefficients(
        &self,
        variable: &str,
    ) -> Result<HashMap<i32, f64>, EvaluationError> {
        let mut coefficients = HashMap::new();
        Self::collect_polynomial_terms(&self.expression, variable, &mut coefficients, 1.0)?;
        Ok(coefficients)
    }

    fn collect_polynomial_terms(
        expression: &Expression,
        variable: &str,
        coefficients: &mut HashMap<i32, f64>,
        sign: f64,
    ) -> Result<(), EvaluationError> {
        match expression {
            Expression::Real(n) => {
                *coefficients.entry(0).or_insert(0.0) += sign * n;
            }
            Expression::Complex(r, i) if i.abs() < EPSILON => {
                *coefficients.entry(0).or_insert(0.0) += sign * r;
            }
            Expression::Variable(name) if name == variable => {
                *coefficients.entry(1).or_insert(0.0) += sign;
            }
            Expression::Variable(name) => {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Unexpected variable: {}",
                    name
                )));
            }
            Expression::FunctionCall(fc) => {
                // Check if function call contains the solving variable
                let mut vars = Vec::new();
                Self::collect_variables(expression, &mut vars);

                if vars.contains(&variable.to_string()) {
                    return Err(EvaluationError::UnsupportedOperation(format!(
                        "Cannot solve equations with function calls containing the variable: {}",
                        fc.name
                    )));
                }

                // If function doesn't contain variable, treat as constant
                // But we can't evaluate it without context, so error
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Function call {} should have been evaluated before solving",
                    fc.name
                )));
            }
            Expression::Paren(inner) => {
                Self::collect_polynomial_terms(inner, variable, coefficients, sign)?;
            }
            Expression::Add(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign)?;
                Self::collect_polynomial_terms(right, variable, coefficients, sign)?;
            }
            Expression::Sub(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign)?;
                Self::collect_polynomial_terms(right, variable, coefficients, -sign)?;
            }
            Expression::Mul(..) => {
                let (coeff, degree) = Self::extract_term(expression, variable)?;
                *coefficients.entry(degree).or_insert(0.0) += sign * coeff;
            }
            Expression::Pow(..) => {
                let (coeff, degree) = Self::extract_term(expression, variable)?;
                *coefficients.entry(degree).or_insert(0.0) += sign * coeff;
            }
            Expression::Neg(inner) => {
                Self::collect_polynomial_terms(inner, variable, coefficients, -sign)?;
            }
            _ => {
                return Err(EvaluationError::UnsupportedOperation(
                    "Unsupported expression in polynomial".to_string(),
                ));
            }
        }
        Ok(())
    }

    fn extract_term(
        expression: &Expression,
        variable: &str,
    ) -> Result<(f64, i32), EvaluationError> {
        match expression {
            Expression::Real(n) => Ok((*n, 0)),
            Expression::Complex(r, i) if i.abs() < EPSILON => Ok((*r, 0)),
            Expression::Variable(name) if name == variable => Ok((1.0, 1)),
            Expression::Paren(inner) => Self::extract_term(inner, variable),
            Expression::Neg(inner) => {
                let (coeff, degree) = Self::extract_term(inner, variable)?;
                Ok((-coeff, degree))
            }
            Expression::Mul(left, right) => {
                let (left_coeff, left_degree) = Self::extract_term(left, variable)?;
                let (right_coeff, right_degree) = Self::extract_term(right, variable)?;
                Ok((left_coeff * right_coeff, left_degree + right_degree))
            }
            Expression::Pow(left, right) => {
                if let Expression::Real(exp) = right.as_ref() {
                    let exp_i32 = *exp as i32;

                    // First unwrap any parentheses from the base
                    let unwrapped_left = match left.as_ref() {
                        Expression::Paren(inner) => inner.as_ref(),
                        other => other,
                    };

                    match unwrapped_left {
                        // Handle x^n
                        Expression::Variable(name) if name == variable => Ok((1.0, exp_i32)),
                        // Handle (coeff * x)^n
                        Expression::Mul(l, r) => {
                            // Try both orderings: (coeff * x) and (x * coeff)
                            if let (Expression::Real(coeff), Expression::Variable(name)) =
                                (l.as_ref(), r.as_ref())
                            {
                                if name == variable {
                                    return Ok((coeff.powf(*exp), exp_i32));
                                }
                            }
                            if let (Expression::Variable(name), Expression::Real(coeff)) =
                                (l.as_ref(), r.as_ref())
                            {
                                if name == variable {
                                    return Ok((coeff.powf(*exp), exp_i32));
                                }
                            }
                            Err(EvaluationError::UnsupportedOperation(
                                "Invalid power expression in multiplication".to_string(),
                            ))
                        }
                        // Handle (-x)^n or (-(coeff * x))^n or (-coeff * x)^n
                        Expression::Neg(inner) => {
                            // Unwrap any parentheses inside the negation
                            let unwrapped_inner = match inner.as_ref() {
                                Expression::Paren(p) => p.as_ref(),
                                other => other,
                            };

                            match unwrapped_inner {
                                // (-x)^n
                                Expression::Variable(name) if name == variable => {
                                    let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                    Ok((sign, exp_i32))
                                }
                                // (-(coeff * x))^n or (-coeff * x)^n
                                Expression::Mul(l, r) => {
                                    if let (Expression::Real(coeff), Expression::Variable(name)) =
                                        (l.as_ref(), r.as_ref())
                                    {
                                        if name == variable {
                                            let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                            return Ok((sign * coeff.powf(*exp), exp_i32));
                                        }
                                    }
                                    if let (Expression::Variable(name), Expression::Real(coeff)) =
                                        (l.as_ref(), r.as_ref())
                                    {
                                        if name == variable {
                                            let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                            return Ok((sign * coeff.powf(*exp), exp_i32));
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
                } else {
                    Err(EvaluationError::UnsupportedOperation(
                        "Power exponent must be a real number".to_string(),
                    ))
                }
            }
            Expression::FunctionCall(_) => {
                let mut vars = Vec::new();
                Self::collect_variables(expression, &mut vars);

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
                expression
            ))),
        }
    }
}
