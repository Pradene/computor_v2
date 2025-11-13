use std::error;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    InvalidSyntax(String),
    UnexpectedToken(String),
    UnexpectedEof,
    InvalidNumber(String),
    InvalidMatrix(String),
    InvalidVector(String),
}

impl error::Error for ParseError {}
impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
            ParseError::UnexpectedToken(token) => write!(f, "Unexpected token: {}", token),
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::InvalidNumber(num) => write!(f, "Invalid number: {}", num),
            ParseError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
            ParseError::InvalidVector(msg) => write!(f, "Invalid vector: {}", msg),
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
}

impl error::Error for EvaluationError {}
impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluationError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvaluationError::UndefinedFunction(name) => write!(f, "Undefined function: {}", name),
            EvaluationError::WrongArgumentCount {
                name,
                expected,
                got,
            } => {
                write!(f, "{} expect {} arguments, got {}", name, expected, got)
            }
            EvaluationError::DivisionByZero => write!(f, "Division by zero"),
            EvaluationError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            EvaluationError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
            EvaluationError::CannotOverrideBuiltin(msg) => {
                write!(f, "Cannot override builtin function: {}", msg)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ComputorError {
    Parsing(String),
    Evaluation(EvaluationError),
}

impl error::Error for ComputorError {}
impl fmt::Display for ComputorError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ComputorError::Evaluation(e) => write!(f, "{}", e),
            ComputorError::Parsing(e) => write!(f, "{}", e),
        }
    }
}
