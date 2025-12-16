use crate::{
    error::EvaluationError,
    expression::Expression,
    types::{complex, vector},
};

#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinFunction {
    Rad,
    Norm,
    Abs,
    Sqrt,
    Cos,
    Sin,
    Tan,
    Dot,
    Cross,
    Exp,
}

impl BuiltinFunction {
    pub fn name(&self) -> &str {
        match self {
            BuiltinFunction::Rad => "rad",
            BuiltinFunction::Norm => "norm",
            BuiltinFunction::Abs => "abs",
            BuiltinFunction::Sqrt => "sqrt",
            BuiltinFunction::Cos => "cos",
            BuiltinFunction::Sin => "sin",
            BuiltinFunction::Tan => "tan",
            BuiltinFunction::Dot => "dot",
            BuiltinFunction::Cross => "cross",
            BuiltinFunction::Exp => "exp",
        }
    }

    pub fn arity(&self) -> usize {
        match self {
            BuiltinFunction::Rad => 1,
            BuiltinFunction::Norm => 1,
            BuiltinFunction::Abs => 1,
            BuiltinFunction::Sqrt => 1,
            BuiltinFunction::Cos => 1,
            BuiltinFunction::Sin => 1,
            BuiltinFunction::Tan => 1,
            BuiltinFunction::Dot => 2,
            BuiltinFunction::Cross => 2,
            BuiltinFunction::Exp => 1,
        }
    }

    pub fn call(&self, args: &[Expression]) -> Result<Expression, EvaluationError> {
        match self {
            BuiltinFunction::Rad => args[0].rad(),
            BuiltinFunction::Norm => args[0].norm(),
            BuiltinFunction::Abs => args[0].abs(),
            BuiltinFunction::Sqrt => args[0].sqrt(),
            BuiltinFunction::Cos => args[0].cos(),
            BuiltinFunction::Sin => args[0].sin(),
            BuiltinFunction::Tan => args[0].tan(),
            BuiltinFunction::Dot => args[0].dot(args[1].clone()),
            BuiltinFunction::Cross => args[0].cross(args[1].clone()),
            BuiltinFunction::Exp => args[0].exp(),
        }
    }
}

impl Expression {
    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => {
                if *n < 0.0 {
                    Ok(Expression::Complex(0.0, n.abs().sqrt()))
                } else {
                    Ok(Expression::Real(n.sqrt()))
                }
            }
            Expression::Complex(r, i) => Ok(complex::sqrt(*r, *i)),

            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot sqrt {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "sqrt".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Complex(r, i) => Ok(Expression::Real(complex::abs(*r, *i))),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot abs {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "abs".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.exp())),
            Expression::Complex(r, i) => Ok(complex::exp(*r, *i)),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot exp {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "exp".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn norm(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.abs())),
            Expression::Vector(vector) => vector::magnitude(vector),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot norm {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "norm".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn cos(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.cos())),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot cos {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "cos".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn sin(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.sin())),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot sin {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "sin".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn tan(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.tan())),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot tan {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "tan".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn rad(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(n) => Ok(Expression::Real(n.to_radians())),
            expression if expression.is_concrete() => Err(EvaluationError::InvalidOperation(
                format!("Cannot rad {}: incompatible type", expression),
            )),

            expression => Ok(Expression::FunctionCall(
                "rad".to_string(),
                vec![expression.clone()],
            )),
        }
    }

    pub fn dot(&self, rhs: Expression) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Vector(v1), Expression::Vector(v2)) => vector::dot(v1, &v2),
            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot dot {} and {}: incompatible type",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::FunctionCall(
                "dot".to_string(),
                vec![left.clone(), right.clone()],
            )),
        }
    }

    pub fn cross(&self, rhs: Expression) -> Result<Expression, EvaluationError> {
        match (self, rhs) {
            (Expression::Vector(v1), Expression::Vector(v2)) => vector::cross(v1, &v2),
            (left, right) if left.is_concrete() && right.is_concrete() => {
                Err(EvaluationError::InvalidOperation(format!(
                    "Cannot cross {} and {}: incompatible type",
                    left, right
                )))
            }

            (left, right) => Ok(Expression::FunctionCall(
                "cross".to_string(),
                vec![left.clone(), right.clone()],
            )),
        }
    }
}
