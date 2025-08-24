use std::fmt;

#[derive(Debug)]
pub enum ParseError {
    InvalidSyntax(String),
    UnexpectedToken(String),
    UnexpectedEof,
    InvalidNumber(String),
    InvalidMatrix(String),
}

#[derive(Debug)]
pub enum EvaluationError {
    UndefinedVariable(String),
    UndefinedFunction(String),
    WrongArgumentCount { expected: usize, got: usize },
    DivisionByZero,
    InvalidOperation(String),
    UnsupportedOperation(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ParseError::InvalidSyntax(msg) => write!(f, "Invalid syntax: {}", msg),
            ParseError::UnexpectedToken(token) => write!(f, "Unexpected token: {}", token),
            ParseError::UnexpectedEof => write!(f, "Unexpected end of input"),
            ParseError::InvalidNumber(num) => write!(f, "Invalid number: {}", num),
            ParseError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
        }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            EvaluationError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvaluationError::UndefinedFunction(name) => write!(f, "Undefined function: {}", name),
            EvaluationError::WrongArgumentCount { expected, got } => {
                write!(f, "Expected {} arguments, got {}", expected, got)
            }
            EvaluationError::DivisionByZero => write!(f, "Division by zero"),
            EvaluationError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            EvaluationError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for ParseError {}
impl std::error::Error for EvaluationError {}
