pub mod builtin;
pub mod display;
pub mod evaluation;
pub mod operations;
pub mod utils;

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Real(f64),
    Complex(f64, f64),
    Vector(Vec<Expression>),
    Matrix(Vec<Expression>, usize, usize),
    Variable(String),
    FunctionCall(String, Vec<Expression>),
    Paren(Box<Expression>),
    Neg(Box<Expression>),
    Add(Box<Expression>, Box<Expression>),
    Sub(Box<Expression>, Box<Expression>),
    Mul(Box<Expression>, Box<Expression>),
    Hadamard(Box<Expression>, Box<Expression>),
    Div(Box<Expression>, Box<Expression>),
    Mod(Box<Expression>, Box<Expression>),
    Pow(Box<Expression>, Box<Expression>),
}
