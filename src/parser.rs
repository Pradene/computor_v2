use crate::computor::Variable;
use crate::tokenizer::{Token, TokenKind, Tokenizer};

use crate::computor::{FunctionDefinition, Symbol};
use crate::error::ParseError;
use crate::expression::Expression;

pub struct Parser;

#[derive(Debug, Clone)]
pub enum Instruction {
    Command { name: String, args: Vec<String> },
    Assignment { name: String, value: Symbol },
    Equation { left: Expression, right: Expression },
    Query { expression: Expression },
}

impl Parser {
    const COMMANDS: &'static [(&'static str, usize)] =
        &[("quit", 0), ("clear", 0), ("table", 0), ("unset", 1)];

    pub fn parse(line: &str) -> Result<Instruction, ParseError> {
        let tokens = Tokenizer::tokenize(line)?;

        if tokens.is_empty() {
            return Err(ParseError::InvalidSyntax("empty input".to_string()));
        }

        if let Some(eq_pos) = Self::find_equals_position(&tokens) {
            let left_tokens = &tokens[..eq_pos];
            let right_tokens = &tokens[eq_pos + 1..];

            // Check if right side is empty
            if right_tokens.is_empty() {
                return Err(ParseError::MissingExpression("after '='".to_string()));
            }

            // Look for (... = ... ?)
            if let Some(Token {
                kind: TokenKind::Question,
                ..
            }) = right_tokens.last()
            {
                if right_tokens.len() == 1 {
                    let expression = Self::parse_expression_from_tokens(left_tokens)?;
                    Ok(Instruction::Query { expression })
                } else {
                    let left = Self::parse_expression_from_tokens(left_tokens)?;
                    let right = Self::parse_expression_from_tokens(
                        &right_tokens[..right_tokens.len() - 1],
                    )?;
                    Ok(Instruction::Equation { left, right })
                }
            } else {
                // Handle assignment
                Self::parse_assignment(left_tokens, right_tokens)
            }
        } else {
            // Handle command
            Self::parse_command(&tokens)
        }
    }

    fn parse_command(tokens: &[Token]) -> Result<Instruction, ParseError> {
        let cmd_name = if let TokenKind::Identifier(name) = &tokens[0].kind {
            name.as_str()
        } else {
            return Err(ParseError::InvalidSyntax("not a command".to_string()));
        };

        // Find command specification
        let (name, args_count) = Self::COMMANDS
            .iter()
            .find(|(cmd, _)| *cmd == cmd_name)
            .copied()
            .ok_or_else(|| ParseError::InvalidSyntax("unknown command".to_string()))?;

        // Validate argument count
        if args_count != tokens.len() - 1 {
            return Err(ParseError::InvalidSyntax(format!(
                "'{}' expects {} argument(s)",
                name, args_count,
            )));
        }

        let args = tokens[1..].iter().map(|s| s.to_string()).collect();
        Ok(Instruction::Command {
            name: name.to_string(),
            args,
        })
    }

    fn find_equals_position(tokens: &[Token]) -> Option<usize> {
        tokens.iter().position(|t| t.kind == TokenKind::Equal)
    }

    fn parse_assignment(
        left_tokens: &[Token],
        right_tokens: &[Token],
    ) -> Result<Instruction, ParseError> {
        if left_tokens.is_empty() {
            return Err(ParseError::InvalidSyntax(
                "missing variable name before '='".to_string(),
            ));
        }

        // Check if this is a function definition: must have parentheses
        let is_function_definition = left_tokens.len() >= 3
            && left_tokens[1].kind == TokenKind::LeftParen
            && left_tokens.last().unwrap().kind == TokenKind::RightParen;

        if is_function_definition {
            // Function definition: f(params...) = ...
            let name = match &left_tokens[0].kind {
                TokenKind::Identifier(name) => Ok(name.clone()),
                other => Err(ParseError::InvalidSyntax(format!(
                    "expected identifier for function name, got {:?}",
                    other
                ))),
            }?;

            let params = Self::parse_function_parameters(&left_tokens[2..left_tokens.len() - 1])?;
            let body = Self::parse_expression_from_tokens(right_tokens)?;

            let value = Symbol::Function(FunctionDefinition {
                name: name.clone(),
                params,
                body,
            });
            Ok(Instruction::Assignment { name, value })
        } else if left_tokens.len() == 1 {
            // Simple assignment: x = ...
            let name = match &left_tokens[0].kind {
                TokenKind::Identifier(name) => Ok(name.clone()),
                other => Err(ParseError::InvalidSyntax(format!(
                    "expected identifier for variable name, got {:?}",
                    other
                ))),
            }?;

            let expression = Self::parse_expression_from_tokens(right_tokens)?;
            Ok(Instruction::Assignment {
                name: name.clone(),
                value: Symbol::Variable(Variable { name, expression }),
            })
        } else {
            // Invalid: can't assign to expressions like a*b
            Err(ParseError::InvalidSyntax(format!(
                "invalid left-hand side of assignment (use 'f(x)=...' or 'x=...')"
            )))
        }
    }

    fn parse_function_parameters(tokens: &[Token]) -> Result<Vec<String>, ParseError> {
        let mut params = Vec::new();
        let mut expect_identifier = true;

        for token in tokens.iter() {
            match &token.kind {
                TokenKind::Identifier(param) => {
                    if !expect_identifier {
                        return Err(ParseError::InvalidSyntax(format!(
                            "expected comma, got '{}'",
                            param
                        )));
                    } else if params.contains(param) {
                        return Err(ParseError::DuplicateParameter(param.clone()));
                    } else {
                        params.push(param.clone());
                        expect_identifier = false;
                    }
                }
                TokenKind::Comma => {
                    if expect_identifier {
                        return Err(ParseError::InvalidSyntax("unexpected comma".to_string()));
                    }
                    expect_identifier = true;
                }
                other => {
                    return Err(ParseError::InvalidSyntax(format!(
                        "unexpected token {:?} in parameter list",
                        other
                    )))
                }
            }
        }

        // Check if we ended expecting an identifier (trailing comma)
        if expect_identifier && !params.is_empty() {
            return Err(ParseError::InvalidSyntax("trailing comma".to_string()));
        }

        Ok(params)
    }

    fn parse_expression_from_tokens(tokens: &[Token]) -> Result<Expression, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::UnexpectedEof);
        }

        let mut pos = 0;
        let expr = Self::parse_addition(tokens, &mut pos)?;

        // Check if all tokens were consumed
        if pos < tokens.len() {
            return Err(ParseError::InvalidSyntax(format!(
                "unexpected '{}'",
                tokens[pos]
            )));
        }

        Ok(expr)
    }

    fn parse_addition(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = Self::parse_multiplication(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos].kind {
                TokenKind::Plus => {
                    *pos += 1;
                    let right = Self::parse_multiplication(tokens, pos)?;
                    left = Expression::Add(Box::new(left), Box::new(right));
                }
                TokenKind::Minus => {
                    *pos += 1;
                    let right = Self::parse_multiplication(tokens, pos)?;
                    left = Expression::Sub(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplication(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = Self::parse_power(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos].kind {
                TokenKind::Mul => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mul(Box::new(left), Box::new(right));
                }
                TokenKind::Hadamard => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Hadamard(Box::new(left), Box::new(right));
                }
                TokenKind::Divide => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Div(Box::new(left), Box::new(right));
                }
                TokenKind::Modulo => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mod(Box::new(left), Box::new(right));
                }
                TokenKind::Identifier(_) | TokenKind::Imaginary => {
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mul(Box::new(left), Box::new(right));
                }
                TokenKind::Number(_) => {
                    return Err(ParseError::InvalidSyntax(
                        "implicit multiplication not supported (use '*')".to_string(),
                    ));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = Self::parse_unary(tokens, pos)?;

        if *pos < tokens.len() && tokens[*pos].kind == TokenKind::Power {
            *pos += 1;
            let right = Self::parse_power(tokens, pos)?;
            left = Expression::Pow(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match &tokens[*pos].kind {
            TokenKind::Plus => {
                *pos += 1;
                let inner = Self::parse_primary(tokens, pos)?;
                Ok(inner)
            }
            TokenKind::Minus => {
                *pos += 1;
                let inner = Self::parse_primary(tokens, pos)?;
                Ok(Expression::Neg(Box::new(inner)))
            }
            _ => Self::parse_primary(tokens, pos),
        }
    }

    fn parse_primary(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match &tokens[*pos].kind {
            TokenKind::Number(n) => {
                *pos += 1;
                Ok(Expression::Real(*n))
            }
            TokenKind::Imaginary => {
                *pos += 1;
                Ok(Expression::Complex(0.0, 1.0))
            }
            TokenKind::LeftBracket => Self::parse_bracket(tokens, pos),
            TokenKind::Identifier(name) => Self::parse_identifier(tokens, pos, name.clone()),
            TokenKind::LeftParen => Self::parse_parenthesized_expression(tokens, pos),
            other => Err(ParseError::UnexpectedToken(format!("{:?}", other))),
        }
    }

    fn parse_identifier(
        tokens: &[Token],
        pos: &mut usize,
        name: String,
    ) -> Result<Expression, ParseError> {
        *pos += 1;

        // Check for function call
        if *pos < tokens.len() && tokens[*pos].kind == TokenKind::LeftParen {
            Self::parse_function_call(tokens, pos, name)
        } else {
            Ok(Expression::Variable(name))
        }
    }

    fn parse_function_call(
        tokens: &[Token],
        pos: &mut usize,
        name: String,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('

        let mut args = Vec::new();

        // Parse arguments
        while *pos < tokens.len() && tokens[*pos].kind != TokenKind::RightParen {
            let arg = Self::parse_addition(tokens, pos)?;
            args.push(arg);

            if *pos < tokens.len() && tokens[*pos].kind == TokenKind::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::RightParen {
            return Err(ParseError::UnbalancedParentheses(format!(
                "missing ')' for function '{}'",
                name
            )));
        }
        *pos += 1; // consume ')'

        Ok(Expression::FunctionCall(name, args))
    }

    fn parse_parenthesized_expression(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('
        let expression = Self::parse_addition(tokens, pos)?;

        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::RightParen {
            return Err(ParseError::UnbalancedParentheses("missing ')'".to_string()));
        }
        *pos += 1; // consume ')'

        Ok(Expression::Paren(Box::new(expression)))
    }

    fn parse_bracket(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        // Look ahead to determine if this is a matrix or vector
        let is_matrix = if *pos + 1 < tokens.len() {
            tokens[*pos + 1].kind == TokenKind::LeftBracket
        } else {
            false
        };

        if is_matrix {
            Self::parse_matrix(tokens, pos)
        } else {
            Self::parse_vector(tokens, pos)
        }
    }

    fn parse_matrix(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume first '['

        // Verify next token is actually '['
        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::LeftBracket {
            return Err(ParseError::InvalidMatrix(
                "expected '[' for row".to_string(),
            ));
        }

        let mut rows = Vec::new();

        while *pos < tokens.len() && tokens[*pos].kind != TokenKind::RightBracket {
            let row = Self::parse_matrix_row(tokens, pos)?;

            // Validate that row is not empty
            if row.is_empty() {
                return Err(ParseError::InvalidMatrix("empty row".to_string()));
            }

            rows.push(row);

            if *pos < tokens.len() && tokens[*pos].kind == TokenKind::Semicolon {
                *pos += 1; // consume ';'
            } else if *pos < tokens.len() && tokens[*pos].kind != TokenKind::RightBracket {
                return Err(ParseError::InvalidMatrix(format!("expected ';' or ']'")));
            }
        }

        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::RightBracket {
            return Err(ParseError::UnbalancedParentheses(
                "missing ']' for matrix".to_string(),
            ));
        }
        *pos += 1; // consume final ']'

        if rows.is_empty() {
            return Err(ParseError::InvalidMatrix("empty matrix".to_string()));
        }

        let row_length = rows[0].len();
        if rows.iter().any(|r| r.len() != row_length) {
            return Err(ParseError::InvalidMatrix(format!(
                "rows have different lengths"
            )));
        }

        Ok(Expression::Matrix(
            rows.iter().flatten().cloned().collect(),
            rows.len(),
            rows[0].len(),
        ))
    }

    fn parse_matrix_row(tokens: &[Token], pos: &mut usize) -> Result<Vec<Expression>, ParseError> {
        if tokens[*pos].kind != TokenKind::LeftBracket {
            return Err(ParseError::InvalidMatrix(
                "expected '[' for row".to_string(),
            ));
        }

        *pos += 1; // consume '['
        let mut row = Vec::new();

        // Parse row elements
        while *pos < tokens.len() && tokens[*pos].kind != TokenKind::RightBracket {
            let expression = Self::parse_addition(tokens, pos)?;
            row.push(expression);

            if *pos < tokens.len() && tokens[*pos].kind == TokenKind::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::RightBracket {
            return Err(ParseError::UnbalancedParentheses(
                "missing ']' for row".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        Ok(row)
    }

    fn parse_vector(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume '['

        let mut elements = Vec::new();

        while *pos < tokens.len() && tokens[*pos].kind != TokenKind::RightBracket {
            let expression = Self::parse_addition(tokens, pos)?;

            elements.push(expression);

            if *pos < tokens.len() && tokens[*pos].kind == TokenKind::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos].kind != TokenKind::RightBracket {
            return Err(ParseError::UnbalancedParentheses(
                "missing ']' for vector".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        // Check if vector is empty
        if elements.is_empty() {
            return Err(ParseError::InvalidVector("empty vector".to_string()));
        }

        Ok(Expression::Vector(elements))
    }
}
