use {
    crate::{constant::EPSILON, error::EvaluationError, expression::Expression},
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

        // Check if all coefficients are real
        let all_real = coefficients.values().all(|c| c.to_f64().is_some());

        if !all_real {
            return Err(EvaluationError::UnsupportedOperation(
                "Cannot solve polynomial equations with complex coefficients".to_string(),
            ));
        }

        // Convert to HashMap<i32, f64>
        let coefficients: HashMap<i32, f64> = coefficients
            .iter()
            .map(|(deg, coeff)| (*deg, coeff.to_f64().unwrap()))
            .collect();

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
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                Self::collect_variables(left, variables);
                Self::collect_variables(right, variables);
            }
            Expression::Neg(inner) => {
                Self::collect_variables(inner, variables);
            }
            Expression::FunctionCall(_, args) => {
                // Collect variables from function arguments
                for arg in args {
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
    ) -> Result<HashMap<i32, Coefficient>, EvaluationError> {
        let sign = Coefficient::Real(1.0);
        let mut coefficients = HashMap::new();

        Self::collect_polynomial_terms(&self.expression, variable, &mut coefficients, sign)?;

        Ok(coefficients)
    }

    fn collect_polynomial_terms(
        expression: &Expression,
        variable: &str,
        coefficients: &mut HashMap<i32, Coefficient>,
        sign: Coefficient,
    ) -> Result<(), EvaluationError> {
        match expression {
            Expression::Real(n) => {
                let coeff = match sign {
                    Coefficient::Real(s) => Coefficient::Real(s * n),
                    Coefficient::Complex(s_r, s_i) => Coefficient::Complex(s_r * n, s_i * n),
                };
                coefficients
                    .entry(0)
                    .and_modify(|c| *c = c.add(&coeff))
                    .or_insert(coeff);
            }
            Expression::Complex(r, i) => {
                let coeff = match sign {
                    Coefficient::Real(s) => Coefficient::Complex(s * r, s * i),
                    Coefficient::Complex(s_r, s_i) => {
                        // (s_r + s_i*i) * (r + i*i) = (s_r*r - s_i*i) + (s_r*i + s_i*r)*i
                        Coefficient::Complex(s_r * r - s_i * i, s_r * i + s_i * r)
                    }
                };
                coefficients
                    .entry(0)
                    .and_modify(|c| *c = c.add(&coeff))
                    .or_insert(coeff);
            }
            Expression::Variable(name) if name == variable => {
                coefficients
                    .entry(1)
                    .and_modify(|c| *c = c.add(&sign))
                    .or_insert(sign);
            }
            Expression::Variable(name) => {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Unexpected variable: {}",
                    name
                )));
            }
            Expression::FunctionCall(name, _) => {
                // Check if function call contains the solving variable
                let mut vars = Vec::new();
                Self::collect_variables(expression, &mut vars);

                if vars.contains(&variable.to_string()) {
                    return Err(EvaluationError::UnsupportedOperation(format!(
                        "Cannot solve equations with function calls containing the variable: {}",
                        name
                    )));
                }

                // If function doesn't contain variable, treat as constant
                // But we can't evaluate it without context, so error
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Function call {} should have been evaluated before solving",
                    name
                )));
            }
            Expression::Paren(inner) => {
                Self::collect_polynomial_terms(inner, variable, coefficients, sign)?;
            }
            Expression::Add(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign.clone())?;
                Self::collect_polynomial_terms(right, variable, coefficients, sign)?;
            }
            Expression::Sub(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign.clone())?;
                let neg_sign = match sign {
                    Coefficient::Real(s) => Coefficient::Real(-s),
                    Coefficient::Complex(r, i) => Coefficient::Complex(-r, -i),
                };
                Self::collect_polynomial_terms(right, variable, coefficients, neg_sign)?;
            }
            Expression::Mul(..) => {
                let (coeff, degree) = Self::extract_term(expression, variable)?;
                coefficients
                    .entry(degree)
                    .and_modify(|c| {
                        let combined = match (&sign, &coeff) {
                            (Coefficient::Real(s), Coefficient::Real(t)) => {
                                Coefficient::Real(s * t)
                            }
                            (Coefficient::Real(s), Coefficient::Complex(t_r, t_i)) => {
                                Coefficient::Complex(s * t_r, s * t_i)
                            }
                            (Coefficient::Complex(s_r, s_i), Coefficient::Real(t)) => {
                                Coefficient::Complex(s_r * t, s_i * t)
                            }
                            (Coefficient::Complex(s_r, s_i), Coefficient::Complex(t_r, t_i)) => {
                                Coefficient::Complex(s_r * t_r - s_i * t_i, s_r * t_i + s_i * t_r)
                            }
                        };
                        *c = c.add(&combined);
                    })
                    .or_insert_with(|| match (&sign, &coeff) {
                        (Coefficient::Real(s), Coefficient::Real(t)) => Coefficient::Real(s * t),
                        (Coefficient::Real(s), Coefficient::Complex(t_r, t_i)) => {
                            Coefficient::Complex(s * t_r, s * t_i)
                        }
                        (Coefficient::Complex(s_r, s_i), Coefficient::Real(t)) => {
                            Coefficient::Complex(s_r * t, s_i * t)
                        }
                        (Coefficient::Complex(s_r, s_i), Coefficient::Complex(t_r, t_i)) => {
                            Coefficient::Complex(s_r * t_r - s_i * t_i, s_r * t_i + s_i * t_r)
                        }
                    });
            }
            Expression::Pow(..) => {
                let (coeff, degree) = Self::extract_term(expression, variable)?;
                coefficients
                    .entry(degree)
                    .and_modify(|c| {
                        let combined = match (&sign, &coeff) {
                            (Coefficient::Real(s), Coefficient::Real(t)) => {
                                Coefficient::Real(s * t)
                            }
                            (Coefficient::Real(s), Coefficient::Complex(t_r, t_i)) => {
                                Coefficient::Complex(s * t_r, s * t_i)
                            }
                            (Coefficient::Complex(s_r, s_i), Coefficient::Real(t)) => {
                                Coefficient::Complex(s_r * t, s_i * t)
                            }
                            (Coefficient::Complex(s_r, s_i), Coefficient::Complex(t_r, t_i)) => {
                                Coefficient::Complex(s_r * t_r - s_i * t_i, s_r * t_i + s_i * t_r)
                            }
                        };
                        *c = c.add(&combined);
                    })
                    .or_insert_with(|| match (&sign, &coeff) {
                        (Coefficient::Real(s), Coefficient::Real(t)) => Coefficient::Real(s * t),
                        (Coefficient::Real(s), Coefficient::Complex(t_r, t_i)) => {
                            Coefficient::Complex(s * t_r, s * t_i)
                        }
                        (Coefficient::Complex(s_r, s_i), Coefficient::Real(t)) => {
                            Coefficient::Complex(s_r * t, s_i * t)
                        }
                        (Coefficient::Complex(s_r, s_i), Coefficient::Complex(t_r, t_i)) => {
                            Coefficient::Complex(s_r * t_r - s_i * t_i, s_r * t_i + s_i * t_r)
                        }
                    });
            }
            Expression::Neg(inner) => {
                let neg_sign = match sign {
                    Coefficient::Real(s) => Coefficient::Real(-s),
                    Coefficient::Complex(r, i) => Coefficient::Complex(-r, -i),
                };
                Self::collect_polynomial_terms(inner, variable, coefficients, neg_sign)?;
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
    ) -> Result<(Coefficient, i32), EvaluationError> {
        match expression {
            Expression::Real(n) => Ok((Coefficient::Real(*n), 0)),
            Expression::Complex(r, i) => Ok((Coefficient::Complex(*r, *i), 0)),
            Expression::Variable(name) if name == variable => Ok((Coefficient::Real(1.0), 1)),
            Expression::Paren(inner) => Self::extract_term(inner, variable),
            Expression::Neg(inner) => {
                let (coeff, degree) = Self::extract_term(inner, variable)?;
                let neg_coeff = match coeff {
                    Coefficient::Real(c) => Coefficient::Real(-c),
                    Coefficient::Complex(r, i) => Coefficient::Complex(-r, -i),
                };
                Ok((neg_coeff, degree))
            }
            Expression::Mul(left, right) => {
                let (left_coeff, left_degree) = Self::extract_term(left, variable)?;
                let (right_coeff, right_degree) = Self::extract_term(right, variable)?;

                let combined_coeff = match (left_coeff, right_coeff) {
                    (Coefficient::Real(l), Coefficient::Real(r)) => Coefficient::Real(l * r),
                    (Coefficient::Real(l), Coefficient::Complex(r_r, r_i)) => {
                        Coefficient::Complex(l * r_r, l * r_i)
                    }
                    (Coefficient::Complex(l_r, l_i), Coefficient::Real(r)) => {
                        Coefficient::Complex(l_r * r, l_i * r)
                    }
                    (Coefficient::Complex(l_r, l_i), Coefficient::Complex(r_r, r_i)) => {
                        Coefficient::Complex(l_r * r_r - l_i * r_i, l_r * r_i + l_i * r_r)
                    }
                };

                Ok((combined_coeff, left_degree + right_degree))
            }
            Expression::Pow(left, right) => {
                // Check if exponent is a non-negative integer
                let exp_value = match right.as_ref() {
                    Expression::Real(exp) => *exp,
                    _ => {
                        return Err(EvaluationError::UnsupportedOperation(
                            "Power exponent must be a real number constant".to_string(),
                        ))
                    }
                };

                // Check if it's a non-negative integer
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

                // First unwrap any parentheses from the base
                let unwrapped_left = match left.as_ref() {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };

                match unwrapped_left {
                    // Handle x^n
                    Expression::Variable(name) if name == variable => {
                        Ok((Coefficient::Real(1.0), exp_i32))
                    }
                    // Handle (coeff * x)^n
                    Expression::Mul(l, r) => {
                        // Try both orderings: (coeff * x) and (x * coeff)
                        if let (Expression::Real(coeff), Expression::Variable(name)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Ok((Coefficient::Real(coeff.powf(exp_value)), exp_i32));
                            }
                        }
                        if let (Expression::Variable(name), Expression::Real(coeff)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Ok((Coefficient::Real(coeff.powf(exp_value)), exp_i32));
                            }
                        }
                        // Handle complex coefficient cases
                        if let (Expression::Complex(_, _), Expression::Variable(name)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                // For complex^n, we need to compute it properly
                                // This is complex, so for now we'll return an error
                                return Err(EvaluationError::UnsupportedOperation(
                                    "Cannot solve equations with complex coefficients raised to powers".to_string(),
                                ));
                            }
                        }
                        if let (Expression::Variable(name), Expression::Complex(_, _)) =
                            (l.as_ref(), r.as_ref())
                        {
                            if name == variable {
                                return Err(EvaluationError::UnsupportedOperation(
                                    "Cannot solve equations with complex coefficients raised to powers".to_string(),
                                ));
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
                                Ok((Coefficient::Real(sign), exp_i32))
                            }
                            // (-(coeff * x))^n or (-coeff * x)^n
                            Expression::Mul(l, r) => {
                                if let (Expression::Real(coeff), Expression::Variable(name)) =
                                    (l.as_ref(), r.as_ref())
                                {
                                    if name == variable {
                                        let sign = if exp_i32 % 2 == 0 { 1.0 } else { -1.0 };
                                        return Ok((
                                            Coefficient::Real(sign * coeff.powf(exp_value)),
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
                                            Coefficient::Real(sign * coeff.powf(exp_value)),
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

#[derive(Debug, Clone)]
enum Coefficient {
    Real(f64),
    Complex(f64, f64),
}

impl Coefficient {
    fn to_f64(&self) -> Option<f64> {
        match self {
            Coefficient::Real(r) => Some(*r),
            Coefficient::Complex(r, i) if i.abs() < EPSILON => Some(*r),
            Coefficient::Complex(_, _) => None,
        }
    }

    fn add(&self, other: &Coefficient) -> Coefficient {
        match (self, other) {
            (Coefficient::Real(a), Coefficient::Real(b)) => Coefficient::Real(a + b),
            (Coefficient::Real(a), Coefficient::Complex(b_r, b_i)) => {
                Coefficient::Complex(a + b_r, *b_i)
            }
            (Coefficient::Complex(a_r, a_i), Coefficient::Real(b)) => {
                Coefficient::Complex(a_r + b, *a_i)
            }
            (Coefficient::Complex(a_r, a_i), Coefficient::Complex(b_r, b_i)) => {
                Coefficient::Complex(a_r + b_r, a_i + b_i)
            }
        }
    }
}
