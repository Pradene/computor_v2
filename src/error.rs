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
    NotAFunction(String),
    NotAVariable(String),
    WrongArgumentCount { expected: usize, got: usize },
    DivisionByZero,
    InvalidOperation(String),
    UndefinedOperation,
    InvalidMatrix(String),
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
            EvaluationError::NotAFunction(name) => write!(f, "'{}' is not a function", name),
            EvaluationError::NotAVariable(name) => {
                write!(f, "Cannot use function '{}' as variable", name)
            }
            EvaluationError::WrongArgumentCount { expected, got } => {
                write!(f, "Expected {} arguments, got {}", expected, got)
            }
            EvaluationError::DivisionByZero => write!(f, "Division by zero"),
            EvaluationError::UndefinedOperation => write!(f, "Undefined operation"),
            EvaluationError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            EvaluationError::InvalidMatrix(msg) => write!(f, "Invalid matrix: {}", msg),
        }
    }
}

impl std::error::Error for ParseError {}
impl std::error::Error for EvaluationError {}
