use std::error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    InvalidSyntax(String),
    UnexpectedToken(String),
    UnexpectedEof,
    DuplicateParameter(String),
    InvalidNumber(String),
    InvalidMatrix(String),
    InvalidVector(String),
    Overflow(String),
    MissingExpression(String),
    UnbalancedParentheses(String),
}

impl error::Error for ParseError {}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::InvalidSyntax(msg) => write!(f, "Syntax error: {}", msg),
            ParseError::UnexpectedToken(token) => write!(f, "Unexpected token: '{}'", token),
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::DuplicateParameter(msg) => write!(f, "Duplicate parameter: {}", msg),
            ParseError::InvalidNumber(num) => write!(f, "Invalid number: {}", num),
            ParseError::InvalidMatrix(msg) => write!(f, "Matrix error: {}", msg),
            ParseError::InvalidVector(msg) => write!(f, "Vector error: {}", msg),
            ParseError::Overflow(msg) => write!(f, "Overflow: {}", msg),
            ParseError::MissingExpression(msg) => write!(f, "Missing expression: {}", msg),
            ParseError::UnbalancedParentheses(msg) => write!(f, "Parentheses: {}", msg),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    WrongArgumentCount {
        name: String,
        expected: usize,
        got: usize,
    },
    DivisionByZero,
    InvalidOperation(String),
    UnsupportedOperation(String),
    CannotOverrideBuiltin(String),
    DimensionMismatch(String),
    TypeMismatch(String),
}

impl error::Error for EvaluationError {}
impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluationError::UndefinedVariable(name) => write!(f, "Undefined variable '{}'", name),
            EvaluationError::UndefinedFunction(name) => write!(f, "Undefined function '{}'", name),
            EvaluationError::WrongArgumentCount {
                name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Function '{}' expects {} arguments, got {}",
                    name, expected, got
                )
            }
            EvaluationError::DivisionByZero => write!(f, "Division by zero"),
            EvaluationError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            EvaluationError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            EvaluationError::CannotOverrideBuiltin(msg) => {
                write!(f, "Cannot override built-in: {}", msg)
            }
            EvaluationError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            EvaluationError::TypeMismatch(msg) => write!(f, "Type mismatch: {}", msg),
        }
    }
}
