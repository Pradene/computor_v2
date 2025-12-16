use crate::{error::EvaluationError, expression::Expression, types::polynomial};

impl Expression {
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
