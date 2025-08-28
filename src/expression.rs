use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::error::EvaluationError;
use crate::types::complex::Complex;
use crate::types::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Modulo,
    Multiply,
    Divide,
    Power,
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Subtract => write!(f, "-"),
            BinaryOperator::Multiply => write!(f, "*"),
            BinaryOperator::Divide => write!(f, "/"),
            BinaryOperator::Modulo => write!(f, "%"),
            BinaryOperator::Power => write!(f, "^"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOperator {
    Plus,
    Minus,
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOperator::Plus => write!(f, "+"),
            UnaryOperator::Minus => write!(f, "-"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Complex(Complex),
    Matrix(Matrix),
    Variable(String),
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },
}

impl Expression {
    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => *n == 0.0,
            Expression::Complex(c) => c.is_zero(),
            _ => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Expression::Real(n) => *n == 1.0,
            Expression::Complex(c) => c.is_one(),
            _ => false,
        }
    }

    pub fn is_numeric(&self) -> bool {
        matches!(self, Expression::Real(_) | Expression::Complex(_))
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Real(_) => true,
            Expression::Complex(c) => c.is_real(),
            _ => false,
        }
    }

    pub fn is_matrix(&self) -> bool {
        matches!(self, Expression::Matrix(_))
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Expression::Variable(_))
    }

    pub fn is_symbolic(&self) -> bool {
        matches!(
            self,
            Expression::Variable(_)
                | Expression::FunctionCall { .. }
                | Expression::BinaryOp { .. }
                | Expression::UnaryOp { .. }
        )
    }

    pub fn reduce(&self) -> Result<Expression, EvaluationError> {
        let mut current = self.clone();
        const MAX_ITERATIONS: usize = 64; // Prevent infinite loops

        for _ in 0..MAX_ITERATIONS {
            let collected = current.collect_terms()?;
            if collected == current {
                break;
            }
            current = collected;
        }

        Ok(current)
    }

    /// Collect like terms (e.g., 2x + 3x = 5x, x + x = 2x)
    fn collect_terms(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::BinaryOp {
                op: BinaryOperator::Add,
                ..
            } => {
                let mut terms = HashMap::new();
                self.collect_addition_terms(&mut terms)?;
                self.rebuild_from_terms(terms)
            }
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => {
                // Convert a - b to a + (-b) for easier term collection
                let neg_right = Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(right.as_ref().clone()),
                };
                let as_addition = Expression::BinaryOp {
                    left: left.clone(),
                    op: BinaryOperator::Add,
                    right: Box::new(neg_right),
                };
                as_addition.collect_terms()
            }
            // For other operations, recursively collect terms in sub-expressions
            Expression::BinaryOp { left, op, right } => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left_collected == **left && right_collected == **right {
                    Ok(self.clone()) // No change
                } else {
                    Ok(Expression::BinaryOp {
                        left: Box::new(left_collected),
                        op: op.clone(),
                        right: Box::new(right_collected),
                    })
                }
            }
            Expression::UnaryOp { op, operand } => {
                let operand_collected = operand.collect_terms()?;
                if operand_collected == **operand {
                    Ok(self.clone()) // No change
                } else {
                    Ok(Expression::UnaryOp {
                        op: op.clone(),
                        operand: Box::new(operand_collected),
                    })
                }
            }
            _ => Ok(self.clone()), // Atomic expressions don't need collection
        }
    }

    fn collect_addition_terms(
        &self,
        terms: &mut HashMap<String, f64>,
    ) -> Result<(), EvaluationError> {
        match self {
            // Handle addition: a + b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Add,
                right,
            } => {
                left.collect_addition_terms(terms)?;
                right.collect_addition_terms(terms)?;
            }

            // Handle subtraction as negative addition: a - b -> a + (-1)*b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => {
                left.collect_addition_terms(terms)?;
                right.collect_addition_terms_with_coefficient(terms, -1.0)?;
            }

            // Handle unary minus: -x -> (-1)*x
            Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
            } => {
                operand.collect_addition_terms_with_coefficient(terms, -1.0)?;
            }

            // Handle unary plus: +x -> x
            Expression::UnaryOp {
                op: UnaryOperator::Plus,
                operand,
            } => {
                operand.collect_addition_terms(terms)?;
            }

            // Single term with coefficient 1
            _ => {
                self.collect_addition_terms_with_coefficient(terms, 1.0)?;
            }
        }
        Ok(())
    }

    /// Collect terms with a specific coefficient multiplier
    fn collect_addition_terms_with_coefficient(
        &self,
        terms: &mut HashMap<String, f64>,
        coeff: f64,
    ) -> Result<(), EvaluationError> {
        match self {
            // Constants
            Expression::Real(n) => {
                *terms.entry("__constant__".to_string()).or_insert(0.0) += coeff * n;
            }

            // Variables: x -> 1*x
            Expression::Variable(name) => {
                *terms.entry(name.clone()).or_insert(0.0) += coeff;
            }

            // Multiplication: handle cases like 3*x, x*3, 2*x*y, etc.
            Expression::BinaryOp {
                op: BinaryOperator::Multiply,
                ..
            } => {
                // Extract coefficient and variables from multiplication chain
                let (extracted_coeff, variables) = self.extract_multiplication_parts();
                let total_coeff = coeff * extracted_coeff;

                if variables.is_empty() {
                    // Pure constant multiplication
                    *terms.entry("__constant__".to_string()).or_insert(0.0) += total_coeff;
                } else {
                    // Create a key from the variables
                    let key = if variables.len() == 1 {
                        variables[0].clone()
                    } else {
                        // Multiple variables: create a canonical form like "X * Y"
                        let mut sorted_vars = variables;
                        sorted_vars.sort();
                        sorted_vars.join(" * ")
                    };
                    *terms.entry(key).or_insert(0.0) += total_coeff;
                }
            }

            // Complex expressions: treat as single terms
            _ => {
                let key = format!("{}", self);
                *terms.entry(key).or_insert(0.0) += coeff;
            }
        }
        Ok(())
    }

    /// Extract coefficient and variables from a multiplication expression
    /// Returns (coefficient, vec_of_variable_names)
    fn extract_multiplication_parts(&self) -> (f64, Vec<String>) {
        let mut coefficient = 1.0;
        let mut variables = Vec::new();

        self.collect_multiplication_parts(&mut coefficient, &mut variables);

        (coefficient, variables)
    }

    /// Recursively collect parts of a multiplication expression
    fn collect_multiplication_parts(&self, coefficient: &mut f64, variables: &mut Vec<String>) {
        match self {
            Expression::Real(n) => {
                *coefficient *= n;
            }
            Expression::Variable(name) => {
                variables.push(name.clone());
            }
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Multiply,
                right,
            } => {
                left.collect_multiplication_parts(coefficient, variables);
                right.collect_multiplication_parts(coefficient, variables);
            }
            Expression::Complex(c) if c.is_real() => {
                *coefficient *= c.real;
            }
            _ => {
                // For other expressions, treat as a single variable-like term
                variables.push(format!("{}", self));
            }
        }
    }

    /// Rebuild expression from collected terms
    fn rebuild_from_terms(
        &self,
        terms: HashMap<String, f64>,
    ) -> Result<Expression, EvaluationError> {
        let mut result_terms = Vec::new();

        for (term_str, coeff) in terms {
            // Skip zero coefficients
            if coeff == 0.0 {
                continue;
            }

            let term_expr = if term_str == "__constant__" {
                // Constant term
                Expression::Real(coeff)
            } else if (coeff - 1.0).abs() < f64::EPSILON {
                // Coefficient is 1, just use the term
                if term_str.starts_with('(') && term_str.ends_with(')') {
                    // This was a complex expression - would need proper parsing
                    // For now, create a placeholder variable
                    Expression::Variable(term_str)
                } else {
                    Expression::Variable(term_str)
                }
            } else if (coeff + 1.0).abs() < f64::EPSILON {
                // Coefficient is -1
                let var_expr = Expression::Variable(term_str);
                Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(var_expr),
                }
            } else {
                // General case: coefficient * term
                let var_expr = Expression::Variable(term_str);
                Expression::BinaryOp {
                    left: Box::new(Expression::Real(coeff)),
                    op: BinaryOperator::Multiply,
                    right: Box::new(var_expr),
                }
            };

            result_terms.push(term_expr);
        }

        // Handle empty result
        if result_terms.is_empty() {
            return Ok(Expression::Real(0.0));
        }

        // Build the addition chain using your existing operations
        let mut result = result_terms.clone().into_iter().next().unwrap();
        for term in result_terms.into_iter().skip(1) {
            // Use your existing Add implementation
            result = (result + term)?;
        }

        Ok(result)
    }
}

impl Add for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, &rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a + b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) + *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a + Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a + *b)),

            // Matrix operations
            (Expression::Matrix(a), Expression::Matrix(b)) => {
                (a + b.clone()).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix addition failed".to_string())
                })
            }

            // Invalid mixed operations
            (Expression::Matrix(_), Expression::Real(_))
            | (Expression::Matrix(_), Expression::Complex(_))
            | (Expression::Real(_), Expression::Matrix(_))
            | (Expression::Complex(_), Expression::Matrix(_)) => Err(
                EvaluationError::InvalidOperation("Cannot add scalar and matrix".to_string()),
            ),

            // Algebraic simplifications for numeric types only
            (expr, rhs) if rhs.is_zero() && !expr.is_matrix() => Ok(expr), // x + 0 = x
            (lhs, expr) if lhs.is_zero() && !expr.is_matrix() => Ok(expr.clone()), // 0 + x = x
            (lhs, rhs) if lhs == *rhs && (lhs.is_numeric() || lhs.is_variable()) => {
                // x + x = 2*x (only for numeric types and variables, not matrices)
                Expression::Real(2.0) * lhs
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Add,
                right: Box::new(rhs.clone()),
            }),
        }
    }
}

impl Sub for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, &rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a - b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) - *b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a - Complex::new(*b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a - *b)),

            // Matrix operations
            (Expression::Matrix(a), Expression::Matrix(b)) => {
                (a - b.clone()).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix subtraction failed".to_string())
                })
            }

            // Invalid mixed operations
            (Expression::Matrix(_), Expression::Real(_))
            | (Expression::Matrix(_), Expression::Complex(_))
            | (Expression::Real(_), Expression::Matrix(_))
            | (Expression::Complex(_), Expression::Matrix(_)) => Err(
                EvaluationError::InvalidOperation("Cannot substract scalar and matrix".to_string()),
            ),

            // Algebraic simplifications
            (expr, rhs) if rhs.clone().is_zero() => Ok(expr), // x - 0 = x
            (lhs, _) if lhs.is_zero() => -rhs.clone(),        // 0 - x = -x
            (lhs, rhs) if lhs == *rhs => Ok(Expression::Real(0.0)), // x - x = 0

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Subtract,
                right: Box::new(rhs.clone()),
            }),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a * b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) * b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a * Complex::new(b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a * b)),

            // Matrix operations
            (Expression::Matrix(a), Expression::Matrix(b)) => {
                (a * b).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Matrix multiplication failed".to_string())
                })
            }

            // Scalar-matrix multiplication
            (Expression::Real(scalar), Expression::Matrix(matrix)) => {
                (matrix * scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }
            (Expression::Complex(scalar), Expression::Matrix(matrix)) => {
                (matrix * scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }
            (Expression::Matrix(matrix), Expression::Real(scalar)) => {
                (matrix * scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }
            (Expression::Matrix(matrix), Expression::Complex(scalar)) => {
                (matrix * scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation(
                        "Scalar-matrix multiplication failed".to_string(),
                    )
                })
            }

            // Algebraic simplifications
            (_, expr) | (expr, _) if expr.is_zero() => Ok(Expression::Real(0.0)), // 0 * x = 0
            (expr, rhs) if rhs.is_one() => Ok(expr),                              // x * 1 = x
            (lhs, expr) if lhs.is_one() => Ok(expr),                              // 1 * x = x

            // Distributive property: a * (b + c) = a*b + a*c
            (
                a,
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
            ) => {
                let ab = (a.clone() * *left)?;
                let ac = (a * *right)?;
                ab + ac
            }
            // (a + b) * c = a*c + b*c
            (
                Expression::BinaryOp {
                    left,
                    op: BinaryOperator::Add,
                    right,
                },
                c,
            ) => {
                let ac = (*left * c.clone())?;
                let bc = (*right * c)?;
                ac + bc
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Multiply,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Div for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a / b)),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0) / b))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a / Complex::new(b, 0.0)))
            }
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a / b)),

            // Scalar-matrix division
            (Expression::Real(_), Expression::Matrix(_)) => Err(EvaluationError::InvalidOperation(
                "Cannot divide real number by matrix".to_string(),
            )),
            (Expression::Complex(_), Expression::Matrix(_)) => {
                Err(EvaluationError::InvalidOperation(
                    "Cannot divide complex number by matrix".to_string(),
                ))
            }
            (Expression::Matrix(matrix), Expression::Real(scalar)) => {
                (matrix / scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Scalar-matrix division failed".to_string())
                })
            }
            (Expression::Matrix(matrix), Expression::Complex(scalar)) => {
                (matrix / scalar).map(Expression::Matrix).map_err(|_| {
                    EvaluationError::InvalidOperation("Scalar-matrix division failed".to_string())
                })
            }

            // Algebraic simplifications
            (expr, rhs) if rhs.is_one() => Ok(expr), // x / 1 = x
            (lhs, _) if lhs.is_zero() => Ok(Expression::Real(0.0)), // 0 / x = 0

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Divide,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Rem for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (self, rhs) {
            // Only support real numbers for modulo
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a % b)),
            (Expression::Complex(a), Expression::Complex(b)) if a.is_real() && b.is_real() => {
                Ok(Expression::Real(a.real % b.real))
            }

            // Algebraic simplifications
            (lhs, rhs) if lhs == rhs => Ok(Expression::Real(0.0)), // x % x = 0
            (Expression::Real(0.0), _) => Ok(Expression::Real(0.0)), // 0 % x = 0

            // Unsupported operations for complex numbers
            (Expression::Complex(_), _) | (_, Expression::Complex(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Modulo operation is only supported for real numbers".to_string(),
                ))
            }

            // Matrix modulo is not supported
            (Expression::Matrix(_), _) | (_, Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Modulo operation is not supported for matrices".to_string(),
                ))
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Modulo,
                right: Box::new(rhs),
            }),
        }
    }
}

impl Neg for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn neg(self) -> Self::Output {
        match self {
            Expression::Real(n) => Ok(Expression::Real(-n)),
            Expression::Complex(c) => Ok(Expression::Complex(-c)),
            Expression::Matrix(m) => (-m).map(Expression::Matrix).map_err(|_| {
                EvaluationError::InvalidOperation("Matrix negation failed".to_string())
            }),
            // Double negative: -(-x) = x
            Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand,
            } => Ok(*operand),
            // Distribute minus over addition: -(a + b) = -a - b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Add,
                right,
            } => {
                let neg_left = (-*left)?;
                let neg_right = (-*right)?;
                neg_left - neg_right
            }
            // Distribute minus over subtraction: -(a - b) = b - a
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Subtract,
                right,
            } => *right - *left,
            // Distribute minus over multiplication: -(a * b) = (-a) * b
            Expression::BinaryOp {
                left,
                op: BinaryOperator::Multiply,
                right,
            } => {
                let neg_left = (-*left)?;
                neg_left * *right
            }
            expr => Ok(Expression::UnaryOp {
                op: UnaryOperator::Minus,
                operand: Box::new(expr),
            }),
        }
    }
}

pub trait Power<Rhs = Self> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}

impl Power for Expression {
    type Output = Result<Expression, EvaluationError>;

    fn pow(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            // Numeric operations
            (Expression::Real(a), Expression::Real(b)) => Ok(Expression::Real(a.powf(b))),
            (Expression::Complex(a), Expression::Complex(b)) => Ok(Expression::Complex(a.pow(b))),
            (Expression::Real(a), Expression::Complex(b)) => {
                Ok(Expression::Complex(Complex::new(a, 0.0).pow(b)))
            }
            (Expression::Complex(a), Expression::Real(b)) => {
                Ok(Expression::Complex(a.pow(Complex::new(b, 0.0))))
            }

            // Algebraic simplifications
            (_, Expression::Real(0.0)) => Ok(Expression::Real(1.0)), // x^0 = 1
            (expr, Expression::Real(1.0)) => Ok(expr),               // x^1 = x
            (Expression::Real(1.0), _) => Ok(Expression::Real(1.0)), // 1^x = 1
            (Expression::Real(0.0), _) => Err(EvaluationError::InvalidOperation(
                "Zero to non-positive power is undefined".to_string(),
            )),

            // Matrix powers are not generally supported (except for square matrices and integer powers)
            (Expression::Matrix(_), _) | (_, Expression::Matrix(_)) => {
                Err(EvaluationError::UnsupportedOperation(
                    "Matrix exponentiation is not supported".to_string(),
                ))
            }

            // Convert to symbolic expression
            (lhs, rhs) => Ok(Expression::BinaryOp {
                left: Box::new(lhs),
                op: BinaryOperator::Power,
                right: Box::new(rhs),
            }),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Real(n) => write!(f, "{}", n)?,
            Expression::Complex(n) => write!(f, "{}", n)?,
            Expression::Matrix(matrix) => write!(f, "{}", matrix)?,
            Expression::Variable(name) => write!(f, "{}", name)?,
            Expression::FunctionCall { name, args } => {
                write!(f, "{}", name)?;
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")?;
            }
            Expression::BinaryOp { left, op, right } => {
                let needs_left_parens = match (op, left.as_ref()) {
                    (
                        BinaryOperator::Multiply | BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        },
                    ) => true,
                    (BinaryOperator::Power, Expression::BinaryOp { .. }) => true,
                    _ => false,
                };

                let needs_right_parens = match (op, right.as_ref()) {
                    (
                        BinaryOperator::Subtract,
                        Expression::BinaryOp {
                            op: BinaryOperator::Add | BinaryOperator::Subtract,
                            ..
                        },
                    ) => true,
                    (
                        BinaryOperator::Divide,
                        Expression::BinaryOp {
                            op: BinaryOperator::Multiply | BinaryOperator::Divide,
                            ..
                        },
                    ) => true,
                    (BinaryOperator::Power, Expression::BinaryOp { .. }) => true,
                    _ => false,
                };

                if needs_left_parens {
                    write!(f, "({})", left)?;
                } else {
                    write!(f, "{}", left)?;
                }

                write!(f, " {} ", op)?;

                if needs_right_parens {
                    write!(f, "({})", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::UnaryOp { op, operand } => match operand.as_ref() {
                Expression::BinaryOp { .. } => write!(f, "{}({})", op, operand)?,
                _ => write!(f, "{}{}", op, operand)?,
            },
        }

        Ok(())
    }
}
