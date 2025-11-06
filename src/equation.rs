use crate::error::EvaluationError;
use crate::expression::Expression;
use crate::types::complex::Complex;
use crate::types::value::Value;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EquationSolution {
    pub variable: String,
    pub solutions: Vec<Expression>,
}

impl std::fmt::Display for EquationSolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.solutions.is_empty() {
            write!(f, "No solution exists")?;
        } else if self.solutions.len() == 1 {
            writeln!(f, "The solution is:")?;
            write!(f, "{} = {}", self.variable, self.solutions[0])?;
        } else {
            writeln!(f, "The solutions are:")?;
            for (i, sol) in self.solutions.iter().enumerate() {
                if i < self.solutions.len() - 1 {
                    writeln!(f, "{} = {}", self.variable, sol)?;
                } else {
                    write!(f, "{} = {}", self.variable, sol)?;
                }
            }
        }

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
            return self.solve_constant_equation();
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
        let solutions = match degree {
            0 => self.solve_degree_0(&coefficients)?,
            1 => self.solve_degree_1(&coefficients)?,
            2 => self.solve_degree_2(&coefficients)?,
            _ => {
                return Err(EvaluationError::UnsupportedOperation(format!(
                    "Cannot solve polynomial equations of degree {}",
                    degree
                )))
            }
        };

        Ok(EquationSolution {
            variable,
            solutions,
        })
    }

    fn solve_constant_equation(&self) -> Result<EquationSolution, EvaluationError> {
        // Check if the constant expression is zero
        let is_zero = match &self.expression {
            Expression::Value(Value::Real(n)) => n.abs() < f64::EPSILON,
            Expression::Value(Value::Complex(c)) => c.is_zero(),
            _ => false,
        };

        if is_zero {
            Ok(EquationSolution {
                variable: String::new(),
                solutions: vec![],
            })
        } else {
            Err(EvaluationError::InvalidOperation(
                "Equation has no solution (contradiction)".to_string(),
            ))
        }
    }

    fn solve_degree_0(
        &self,
        coefficients: &HashMap<i32, f64>,
    ) -> Result<Vec<Expression>, EvaluationError> {
        let c = coefficients.get(&0).copied().unwrap_or(0.0);

        if c.abs() < f64::EPSILON {
            // 0 = 0, infinite solutions
            Err(EvaluationError::InvalidOperation(
                "Infinite solutions (identity equation)".to_string(),
            ))
        } else {
            // c = 0 where c ≠ 0, no solution
            Ok(vec![])
        }
    }

    fn solve_degree_1(
        &self,
        coefficients: &HashMap<i32, f64>,
    ) -> Result<Vec<Expression>, EvaluationError> {
        // ax + b = 0
        // x = -b/a
        let a = coefficients.get(&1).copied().unwrap_or(0.0);
        let b = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < f64::EPSILON {
            return self.solve_degree_0(coefficients);
        }

        let solution = -b / a;
        Ok(vec![Expression::Value(Value::Real(solution))])
    }

    fn solve_degree_2(
        &self,
        coefficients: &HashMap<i32, f64>,
    ) -> Result<Vec<Expression>, EvaluationError> {
        // ax² + bx + c = 0
        let a = coefficients.get(&2).copied().unwrap_or(0.0);
        let b = coefficients.get(&1).copied().unwrap_or(0.0);
        let c = coefficients.get(&0).copied().unwrap_or(0.0);

        if a.abs() < f64::EPSILON {
            return self.solve_degree_1(coefficients);
        }

        // Calculate discriminant: Δ = b² - 4ac
        let discriminant = b * b - 4.0 * a * c;

        if discriminant > f64::EPSILON {
            // Two real solutions
            let sqrt_discriminant = discriminant.sqrt();
            let x1 = (-b + sqrt_discriminant) / (2.0 * a);
            let x2 = (-b - sqrt_discriminant) / (2.0 * a);
            Ok(vec![
                Expression::Value(Value::Real(x1)),
                Expression::Value(Value::Real(x2)),
            ])
        } else if discriminant.abs() < f64::EPSILON {
            // One real solution (double root)
            let x = -b / (2.0 * a);
            Ok(vec![Expression::Value(Value::Real(x))])
        } else {
            // Two complex solutions
            let real_part = -b / (2.0 * a);
            let imag_part = (-discriminant).sqrt() / (2.0 * a);

            let x1 = Complex::new(real_part, imag_part);
            let x2 = Complex::new(real_part, -imag_part);

            Ok(vec![
                Expression::Value(Value::Complex(x1)),
                Expression::Value(Value::Complex(x2)),
            ])
        }
    }

    fn extract_variables(&self) -> Vec<String> {
        let mut variables = Vec::new();
        Self::collect_variables(&self.expression, &mut variables);
        variables
    }

    fn collect_variables(expr: &Expression, variables: &mut Vec<String>) {
        match expr {
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
            Expression::Value(_) => {}
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
        expr: &Expression,
        variable: &str,
        coefficients: &mut HashMap<i32, f64>,
        sign: f64,
    ) -> Result<(), EvaluationError> {
        match expr {
            Expression::Value(Value::Real(n)) => {
                *coefficients.entry(0).or_insert(0.0) += sign * n;
            }
            Expression::Value(Value::Complex(c)) if c.is_real() => {
                *coefficients.entry(0).or_insert(0.0) += sign * c.real;
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
                Self::collect_variables(expr, &mut vars);

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
            Expression::Add(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign)?;
                Self::collect_polynomial_terms(right, variable, coefficients, sign)?;
            }
            Expression::Sub(left, right) => {
                Self::collect_polynomial_terms(left, variable, coefficients, sign)?;
                Self::collect_polynomial_terms(right, variable, coefficients, -sign)?;
            }
            Expression::Mul(..) => {
                let (coeff, degree) = Self::extract_term(expr, variable)?;
                *coefficients.entry(degree).or_insert(0.0) += sign * coeff;
            }
            Expression::Pow(..) => {
                let (coeff, degree) = Self::extract_term(expr, variable)?;
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
        expr: &Expression,
        variable: &str,
    ) -> Result<(f64, i32), EvaluationError> {
        match expr {
            Expression::Value(Value::Real(n)) => Ok((*n, 0)),
            Expression::Value(Value::Complex(c)) if c.is_real() => Ok((c.real, 0)),
            Expression::Variable(name) if name == variable => Ok((1.0, 1)),
            Expression::Mul(left, right) => {
                let (left_coeff, left_degree) = Self::extract_term(left, variable)?;
                let (right_coeff, right_degree) = Self::extract_term(right, variable)?;
                Ok((left_coeff * right_coeff, left_degree + right_degree))
            }
            Expression::Pow(left, right) => {
                if let Expression::Variable(name) = left.as_ref() {
                    if name == variable {
                        if let Expression::Value(Value::Real(exp)) = right.as_ref() {
                            return Ok((1.0, *exp as i32));
                        }
                    }
                }
                Err(EvaluationError::UnsupportedOperation(
                    "Invalid power expression".to_string(),
                ))
            }
            Expression::FunctionCall(_) => {
                let mut vars = Vec::new();
                Self::collect_variables(expr, &mut vars);

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
                expr
            ))),
        }
    }
}
