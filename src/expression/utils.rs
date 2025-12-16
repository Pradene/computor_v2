use std::collections::HashSet;

use crate::{error::EvaluationError, expression::Expression, types::polynomial};

impl Expression {
    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => n.abs() < f64::EPSILON,
            Expression::Complex(r, i) => r.abs() < f64::EPSILON && i.abs() < f64::EPSILON,
            _ => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        matches!(
            self,
            Expression::Real(_)
                | Expression::Complex(_, _)
                | Expression::Vector(_)
                | Expression::Matrix(_, _, _)
        )
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Real(_) => true,
            Expression::Complex(_, i) => i.abs() < f64::EPSILON,
            _ => false,
        }
    }

    pub fn contains_variable(&self, variable: &str) -> bool {
        match self {
            Expression::Real(_) | Expression::Complex(_, _) => false,
            Expression::Variable(name) => name == variable,
            Expression::Vector(v) => v.iter().any(|e| e.contains_variable(variable)),
            Expression::Matrix(data, _, _) => data.iter().any(|e| e.contains_variable(variable)),
            Expression::FunctionCall(_, args) => args.iter().any(|e| e.contains_variable(variable)),
            Expression::Paren(inner) => inner.contains_variable(variable),
            Expression::Neg(inner) => inner.contains_variable(variable),
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                left.contains_variable(variable) || right.contains_variable(variable)
            }
        }
    }

    pub fn collect_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();

        match self {
            Expression::Variable(name) => {
                variables.insert(name.clone());
            }
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                variables.extend(left.collect_variables());
                variables.extend(right.collect_variables());
            }
            Expression::Neg(inner) => {
                variables.extend(inner.collect_variables());
            }
            Expression::Paren(inner) => {
                variables.extend(inner.collect_variables());
            }
            Expression::FunctionCall(_, args) => {
                // Collect variables from function arguments
                for arg in args {
                    variables.extend(arg.collect_variables());
                }
            }
            // Don't collect anything from Real, Complex, or other literal types
            _ => {}
        };

        variables
    }

    pub fn find_roots(&self) -> Result<Vec<Expression>, EvaluationError> {
        let variables = self.collect_variables();

        if variables.is_empty() {
            return Ok(polynomial::solve_constant());
        }

        if variables.len() > 1 {
            return Err(EvaluationError::UnsupportedOperation(format!(
                "Cannot solve equations with multiple variables: {:?}",
                variables
            )));
        }

        let variable = variables.iter().next().unwrap().clone();
        let coefficients = polynomial::collect_coefficients(self, &variable)?;

        // Check if all coefficients are real
        let real_coeffs = polynomial::to_real_coefficients(&coefficients).ok_or_else(|| {
            EvaluationError::UnsupportedOperation(
                "Cannot solve polynomial equations with complex coefficients".to_string(),
            )
        })?;

        let deg = polynomial::degree(&coefficients);

        match deg {
            0 => Ok(polynomial::solve_constant()),
            1 => polynomial::solve_linear(&real_coeffs),
            2 => polynomial::solve_quadratic(&real_coeffs),
            _ => Err(EvaluationError::UnsupportedOperation(format!(
                "Cannot solve polynomial equations of degree {}",
                deg
            ))),
        }
    }
}
