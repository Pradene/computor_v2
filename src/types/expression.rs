use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::context::{Context, ContextValue};
use crate::error::EvaluationError;
use crate::types::{complex::Complex, matrix::Matrix, vector::Vector};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Real(f64),
    Complex(Complex),
    Vector(Vector),
    Matrix(Matrix),
}

impl Neg for Value {
    type Output = Result<Self, EvaluationError>;

    fn neg(self) -> Self::Output {
        match self {
            Value::Real(n) => Ok(Value::Real(-n)),
            Value::Complex(c) => Ok(Value::Complex(-c)),
            Value::Vector(v) => Ok(Value::Vector((-v).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector negation failed: {}", e))
            })?)),
            Value::Matrix(m) => Ok(Value::Matrix((-m).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix negation failed: {}", e))
            })?)),
        }
    }
}

impl Add for Value {
    type Output = Result<Self, EvaluationError>;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a + b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) + b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a + Complex::new(b, 0.0))),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector addition failed: {}", e))
            })?)),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a + b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix addition failed: {}", e))
            })?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} + {}",
                left, right
            ))),
        }
    }
}

impl Sub for Value {
    type Output = Result<Self, EvaluationError>;

    fn sub(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a - b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) - b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a - Complex::new(b, 0.0))),
            (Value::Vector(a), Value::Vector(b)) => Ok(Value::Vector((a - b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Vector subtraction failed: {}", e))
            })?)),
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a - b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix subtraction failed: {}", e))
            })?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} - {}",
                left, right
            ))),
        }
    }
}

impl Mul for Value {
    type Output = Result<Self, EvaluationError>;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a * b)),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0) * b)),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a * Complex::new(b, 0.0))),

            // Scalar * Vector
            (Value::Real(s), Value::Vector(v)) | (Value::Vector(v), Value::Real(s)) => {
                Ok(Value::Vector((v * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Vector multiplication failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Complex(c), Value::Vector(v)) | (Value::Vector(v), Value::Complex(c)) => {
                Ok(Value::Vector((v * c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Complex-Vector multiplication failed: {}",
                        e
                    ))
                })?))
            }

            // Scalar * Matrix
            (Value::Real(s), Value::Matrix(m)) | (Value::Matrix(m), Value::Real(s)) => {
                Ok(Value::Matrix((m * s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Scalar-Matrix multiplication failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Complex(c), Value::Matrix(m)) | (Value::Matrix(m), Value::Complex(c)) => {
                Ok(Value::Matrix((m * c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Complex-Matrix multiplication failed: {}",
                        e
                    ))
                })?))
            }

            // Matrix * Matrix
            (Value::Matrix(a), Value::Matrix(b)) => Ok(Value::Matrix((a * b).map_err(|e| {
                EvaluationError::InvalidOperation(format!("Matrix multiplication failed: {}", e))
            })?)),

            // Matrix * Vector
            (Value::Matrix(a), Value::Vector(b)) => Ok(Value::Vector((a * b).map_err(|e| {
                EvaluationError::InvalidOperation(format!(
                    "Matrix-Vector multiplication failed: {}",
                    e
                ))
            })?)),

            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} * {}",
                left, right
            ))),
        }
    }
}

impl Div for Value {
    type Output = Result<Self, EvaluationError>;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Real(a / b))
            }
            (Value::Complex(a), Value::Complex(b)) => {
                if b.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(a / b))
            }
            (Value::Real(a), Value::Complex(b)) => {
                if b.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(Complex::new(a, 0.0) / b))
            }
            (Value::Complex(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Complex(a / Complex::new(b, 0.0)))
            }
            (Value::Vector(v), Value::Real(s)) => {
                if s == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Vector((v / s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Vector-Scalar division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Vector(v), Value::Complex(c)) => {
                if c.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Vector((v / c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Vector-Complex division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Matrix(m), Value::Real(s)) => {
                if s == 0.0 {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Matrix((m / s).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Matrix-Scalar division failed: {}",
                        e
                    ))
                })?))
            }
            (Value::Matrix(m), Value::Complex(c)) => {
                if c.is_zero() {
                    return Err(EvaluationError::DivisionByZero);
                }
                Ok(Value::Matrix((m / c).map_err(|e| {
                    EvaluationError::InvalidOperation(format!(
                        "Matrix-Complex division failed: {}",
                        e
                    ))
                })?))
            }
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} / {}",
                left, right
            ))),
        }
    }
}

impl Rem for Value {
    type Output = Result<Self, EvaluationError>;

    fn rem(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => {
                if b == 0.0 {
                    return Err(EvaluationError::InvalidOperation(
                        "Modulo by zero".to_string(),
                    ));
                }
                Ok(Value::Real(a % b))
            }
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} % {}",
                left, right
            ))),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Real(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Complex(c) => write!(f, "{}", c),
            Value::Vector(v) => write!(f, "{}", v),
            Value::Matrix(m) => write!(f, "{}", m),
        }
    }
}

impl Value {
    pub fn pow(self, rhs: Self) -> Result<Self, EvaluationError> {
        match (self, rhs) {
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a.powf(b))),
            (Value::Complex(a), Value::Real(b)) => Ok(Value::Complex(a.pow(Complex::new(b, 0.0)))),
            (Value::Complex(a), Value::Complex(b)) => Ok(Value::Complex(a.pow(b))),
            (Value::Real(a), Value::Complex(b)) => Ok(Value::Complex(Complex::new(a, 0.0).pow(b))),
            (Value::Matrix(a), Value::Real(b)) => Ok(Value::Matrix(a.pow(b as i32)?)),
            (left, right) => Err(EvaluationError::InvalidOperation(format!(
                "{} ^ {}",
                left, right
            ))),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Value::Real(n) => n.abs() < f64::EPSILON,
            Value::Complex(c) => c.is_zero(),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    pub args: Vec<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Value(Value),
    Variable(String),
    FunctionCall(FunctionCall),
    Neg(Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Mod(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Value(value) => write!(f, "{}", value),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::FunctionCall(fc) => {
                write!(f, "{}(", fc.name)?;
                for (i, arg) in fc.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expression::Neg(operand) => write!(f, "-{}", operand),
            Expression::Add(left, right) => write!(f, "({} + {})", left, right),
            Expression::Sub(left, right) => write!(f, "({} - {})", left, right),
            Expression::Mul(left, right) => write!(f, "({} * {})", left, right),
            Expression::Div(left, right) => write!(f, "({} / {})", left, right),
            Expression::Mod(left, right) => write!(f, "({} % {})", left, right),
            Expression::Pow(left, right) => write!(f, "({} ^ {})", left, right),
        }
    }
}

impl Expression {
    pub fn is_value(&self) -> bool {
        matches!(self, Expression::Value(_))
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Value(v) => v.is_zero(),
            _ => false,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Value(Value::Real(_)) => true,
            Expression::Value(Value::Complex(c)) => c.is_real(),
            _ => false,
        }
    }

    pub fn is_variable(&self) -> bool {
        matches!(self, Expression::Variable(_))
    }

    pub fn is_function(&self) -> bool {
        matches!(self, Expression::FunctionCall(_))
    }
}

impl Add for Expression {
    type Output = Result<Self, EvaluationError>;
    fn add(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().add(right.clone())?))
            }
            _ => Ok(Expression::Add(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Sub for Expression {
    type Output = Result<Self, EvaluationError>;
    fn sub(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().sub(right.clone())?))
            }
            _ => Ok(Expression::Sub(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Mul for Expression {
    type Output = Result<Self, EvaluationError>;
    fn mul(self, rhs: Self) -> Self::Output {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().mul(right.clone())?))
            }
            _ => Ok(Expression::Mul(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Div for Expression {
    type Output = Result<Self, EvaluationError>;
    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::DivisionByZero);
        }

        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().div(right.clone())?))
            }
            _ => Ok(Expression::Div(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Rem for Expression {
    type Output = Result<Self, EvaluationError>;
    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Err(EvaluationError::InvalidOperation(
                "Modulo by zero".to_string(),
            ));
        }

        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().rem(right.clone())?))
            }
            _ => Ok(Expression::Mod(Box::new(self), Box::new(rhs))),
        }
    }
}

impl Neg for Expression {
    type Output = Result<Self, EvaluationError>;
    fn neg(self) -> Self::Output {
        match self {
            Expression::Value(n) => Ok(Expression::Value(n.neg()?)),
            // -(a + b) = (-a) + (-b) = -a - b
            Expression::Add(left, right) => {
                Expression::Value(Value::Real(0.0)).sub(*left)?.sub(*right)
            }
            // -(a - b) = -a + b = b - a
            Expression::Sub(left, right) => (*right).sub(*left),
            // -(a * b) = (-a) * b
            Expression::Mul(left, right) => {
                let neg_left = (*left).neg()?;
                neg_left.mul(*right)
            }
            // Double negative: -(-x) = x
            Expression::Neg(inner) => Ok(*inner),
            _ => Ok(Expression::Neg(Box::new(self))),
        }
    }
}

impl Expression {
    pub fn pow(self, rhs: Self) -> Result<Expression, EvaluationError> {
        match (&self, &rhs) {
            (Expression::Value(left), Expression::Value(right)) => {
                Ok(Expression::Value(left.clone().pow(right.clone())?))
            }
            _ => Ok(Expression::Pow(Box::new(self), Box::new(rhs))),
        }
    }

    pub fn sqrt(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => {
                if *n < 0.0 {
                    Ok(Expression::Value(Value::Complex(Complex::new(
                        0.0,
                        n.abs().sqrt(),
                    ))))
                } else {
                    Ok(Expression::Value(Value::Real(n.sqrt())))
                }
            }
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Complex(c.sqrt()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Sqrt is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn abs(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.abs()))),
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Real(c.abs()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Abs is not implemented for this type".to_string(),
            )),
        }
    }

    pub fn exp(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(n)) => Ok(Expression::Value(Value::Real(n.exp()))),
            Expression::Value(Value::Complex(c)) => Ok(Expression::Value(Value::Complex(c.exp()))),
            _ => Err(EvaluationError::InvalidOperation(
                "Exp is not implemented for this type".to_string(),
            )),
        }
    }
}

#[derive(Debug, Clone)]
struct Term {
    coefficient: f64,
    expression: Expression,
}

impl Term {
    fn new(coefficient: f64, expression: Expression) -> Self {
        Term {
            coefficient,
            expression,
        }
    }

    fn constant(value: f64) -> Self {
        Term {
            coefficient: value,
            expression: Expression::Value(Value::Real(1.0)),
        }
    }

    fn to_expression(self) -> Expression {
        let coeff_abs = self.coefficient.abs();

        if coeff_abs < f64::EPSILON {
            return Expression::Value(Value::Real(0.0));
        }

        // Check if expression is just the constant 1.0
        if matches!(self.expression, Expression::Value(Value::Real(n)) if (n - 1.0).abs() < f64::EPSILON)
        {
            return Expression::Value(Value::Real(self.coefficient));
        }

        if (self.coefficient - 1.0).abs() < f64::EPSILON {
            self.expression
        } else if (self.coefficient + 1.0).abs() < f64::EPSILON {
            Expression::Neg(Box::new(self.expression))
        } else {
            Expression::Mul(
                Box::new(Expression::Value(Value::Real(self.coefficient))),
                Box::new(self.expression),
            )
        }
    }

    // Create a unique key for grouping like terms
    fn key(&self) -> String {
        format!("{}", self.expression)
    }
}

impl Expression {
    pub fn evaluate(&self, context: &Context) -> Result<Expression, EvaluationError> {
        self.evaluate_internal(context, &HashMap::new())
    }

    pub fn evaluate_internal(
        &self,
        context: &Context,
        scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Value(Value::Real(_)) | Expression::Value(Value::Complex(_)) => {
                Ok(self.clone())
            }

            Expression::Value(Value::Vector(vector)) => {
                let mut evaluated_vector = Vec::new();
                for element in vector.iter() {
                    let evaluated_element = element.evaluate_internal(context, scope)?.reduce()?;
                    evaluated_vector.push(evaluated_element);
                }
                let result = Vector::new(evaluated_vector);
                match result {
                    Ok(vector) => Ok(Expression::Value(Value::Vector(vector))),
                    Err(e) => Err(EvaluationError::InvalidOperation(e)),
                }
            }

            Expression::Value(Value::Matrix(matrix)) => {
                let mut evaluated_matrix = Vec::new();
                for row in 0..matrix.rows() {
                    for col in 0..matrix.cols() {
                        let evaluated_element = matrix
                            .get(row, col)
                            .unwrap()
                            .evaluate_internal(context, scope)?
                            .reduce()?;
                        evaluated_matrix.push(evaluated_element);
                    }
                }
                let result = Matrix::new(evaluated_matrix, matrix.rows(), matrix.cols());
                match result {
                    Ok(matrix) => Ok(Expression::Value(Value::Matrix(matrix))),
                    Err(e) => Err(EvaluationError::InvalidOperation(e)),
                }
            }

            Expression::Variable(name) => {
                // First check function parameter scope
                if let Some(expr) = scope.get(name) {
                    return Ok(expr.clone());
                }

                // Then check context variables
                match context.get_variable(name) {
                    Some(ContextValue::Variable(expr)) => {
                        // Recursively evaluate the variable's expression
                        expr.evaluate_internal(context, scope)
                    }
                    Some(ContextValue::Function { .. }) => Err(EvaluationError::InvalidOperation(
                        format!("Cannot use function '{}' as variable", name),
                    )),
                    None => Ok(Expression::Variable(name.to_string())), // Keep symbolic
                }
            }

            Expression::FunctionCall(fc) => {
                match context.get_variable(fc.name.as_str()) {
                    Some(ContextValue::Function(fun)) => {
                        if fc.args.len() != fun.params.len() {
                            return Err(EvaluationError::WrongArgumentCount {
                                expected: fun.params.len(),
                                got: fc.args.len(),
                            });
                        }

                        // Evaluate arguments first
                        let evaluated_args: Result<Vec<_>, _> = fc
                            .args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        let evaluated_args = evaluated_args?;

                        // Create function scope by combining current scope with function parameters
                        let mut function_scope = scope.clone();
                        for (param, arg) in fun.params.iter().zip(evaluated_args.iter()) {
                            function_scope.insert(param.clone(), arg.clone());
                        }

                        // Evaluate function body with new scope
                        fun.body.evaluate_internal(context, &function_scope)
                    }
                    Some(ContextValue::Variable(_)) => Err(EvaluationError::InvalidOperation(
                        format!("'{}' is not a function", fc.name),
                    )),
                    None => {
                        // Evaluate arguments and keep as symbolic function call
                        let evaluated_args: Result<Vec<_>, _> = fc
                            .args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        Ok(Expression::FunctionCall(FunctionCall {
                            name: fc.name.clone(),
                            args: evaluated_args?,
                        }))
                    }
                }
            }

            Expression::Add(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.add(right_eval)
            }
            Expression::Sub(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.sub(right_eval)
            }
            Expression::Mul(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.mul(right_eval)
            }
            Expression::Div(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.div(right_eval)
            }
            Expression::Mod(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.rem(right_eval)
            }
            Expression::Pow(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.pow(right_eval)
            }
            Expression::Neg(inner) => inner.evaluate_internal(context, scope)?.neg(),
        }
    }

    pub fn reduce(&self) -> Result<Expression, EvaluationError> {
        let mut current = self.clone();
        const MAX_ITERATIONS: usize = 64;

        for _ in 0..MAX_ITERATIONS {
            let collected = current.collect_terms()?;
            if collected == current {
                break;
            }
            current = collected;
        }

        Ok(current)
    }

    fn collect_terms(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Add(..) | Expression::Sub(..) => {
                let terms = self.extract_terms(1.0)?;
                self.combine_like_terms(terms)
            }
            Expression::Mul(left, right) => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left_collected.is_value() && right_collected.is_value() {
                    return left_collected.mul(right_collected);
                }

                if left_collected == **left && right_collected == **right {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Mul(
                        Box::new(left_collected),
                        Box::new(right_collected),
                    ))
                }
            }
            Expression::Neg(inner) => {
                let collected = inner.collect_terms()?;

                if let Expression::Neg(double_inner) = &collected {
                    Ok(*double_inner.clone())
                } else if collected == **inner {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Neg(Box::new(collected)))
                }
            }
            _ => Ok(self.clone()),
        }
    }

    // Extract all terms from an addition/subtraction expression
    fn extract_terms(&self, sign: f64) -> Result<Vec<Term>, EvaluationError> {
        match self {
            Expression::Add(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(sign)?);
                Ok(terms)
            }
            Expression::Sub(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(-sign)?);
                Ok(terms)
            }
            Expression::Neg(inner) => inner.extract_terms(-sign),
            Expression::Value(Value::Real(n)) => Ok(vec![Term::constant(sign * n)]),
            Expression::Value(Value::Complex(c)) if c.is_real() => {
                Ok(vec![Term::constant(sign * c.real)])
            }
            Expression::Mul(..) => {
                let (coeff, expr) = self.extract_coefficient();
                Ok(vec![Term::new(sign * coeff, expr)])
            }
            _ => {
                // Variable, Pow, FunctionCall, etc.
                Ok(vec![Term::new(sign, self.clone())])
            }
        }
    }

    // Extract coefficient from multiplication, returning (coefficient, remaining_expression)
    fn extract_coefficient(&self) -> (f64, Expression) {
        let mut coefficient = 1.0;
        let mut non_constant_parts = Vec::new();

        self.collect_mul_parts(&mut coefficient, &mut non_constant_parts);

        if non_constant_parts.is_empty() {
            (coefficient, Expression::Value(Value::Real(1.0)))
        } else if non_constant_parts.len() == 1 {
            (coefficient, non_constant_parts[0].clone())
        } else {
            let mut result = non_constant_parts[0].clone();
            for part in non_constant_parts.iter().skip(1) {
                result = Expression::Mul(Box::new(result), Box::new(part.clone()));
            }
            (coefficient, result)
        }
    }

    fn collect_mul_parts(&self, coefficient: &mut f64, parts: &mut Vec<Expression>) {
        match self {
            Expression::Value(Value::Real(n)) => {
                *coefficient *= n;
            }
            Expression::Value(Value::Complex(c)) if c.is_real() => {
                *coefficient *= c.real;
            }
            Expression::Mul(left, right) => {
                left.collect_mul_parts(coefficient, parts);
                right.collect_mul_parts(coefficient, parts);
            }
            _ => {
                // Variable, Pow, FunctionCall, etc. - keep as-is
                parts.push(self.clone());
            }
        }
    }

    // Combine terms with the same expression part
    fn combine_like_terms(&self, terms: Vec<Term>) -> Result<Expression, EvaluationError> {
        let mut combined: HashMap<String, Term> = HashMap::new();

        for term in terms {
            let key = term.key();
            combined
                .entry(key)
                .and_modify(|existing| existing.coefficient += term.coefficient)
                .or_insert(term);
        }

        // Filter out zero coefficients and convert to expressions
        let mut result_exprs: Vec<Expression> = combined
            .into_values()
            .filter(|term| term.coefficient.abs() >= f64::EPSILON)
            .map(|term| term.to_expression())
            .collect();

        if result_exprs.is_empty() {
            return Ok(Expression::Value(Value::Real(0.0)));
        }

        // Build the final expression with additions
        let mut result = result_exprs.remove(0);
        for expr in result_exprs {
            result = Expression::Add(Box::new(result), Box::new(expr));
        }

        Ok(result)
    }
}
