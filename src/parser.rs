use crate::context::Statement;
use crate::tokenizer::{Token, Tokenizer};

use crate::context::{FunctionDefinition, Symbol};
use crate::error::ParseError;
use crate::expression::{Expression, FunctionCall};

pub struct Parser;

impl Parser {
    pub fn parse(line: &str) -> Result<Statement, ParseError> {
        let line = line.to_lowercase();
        let mut tokenizer = Tokenizer::new(&line);
        let tokens = tokenizer.tokenize()?;

        if tokens.is_empty() || tokens.len() == 1 {
            return Err(ParseError::InvalidSyntax("Empty line".to_string()));
        }

        if let Some(eq_pos) = Self::find_equals_position(&tokens) {
            let left_tokens = &tokens[..eq_pos];
            let right_tokens = &tokens[eq_pos + 1..tokens.len() - 1]; // Exclude EOF

            // Check for query pattern (... = ... ?)
            if right_tokens.iter().last() == Some(&Token::Question) {
                if right_tokens.len() == 1 {
                    let expression = Self::parse_expression_from_tokens(left_tokens)?;
                    Ok(Statement::Query { expression })
                } else {
                    let left = Self::parse_expression_from_tokens(left_tokens)?;
                    let right = Self::parse_expression_from_tokens(right_tokens)?;
                    Ok(Statement::Equation { left, right })
                }
            } else {
                // Handle assignment
                Self::parse_assignment(left_tokens, right_tokens)
            }
        } else {
            Err(ParseError::InvalidSyntax(
                "Expected assignment or query".to_string(),
            ))
        }
    }

    fn find_equals_position(tokens: &[Token]) -> Option<usize> {
        tokens.iter().position(|t| *t == Token::Equal)
    }

    fn parse_assignment(
        left_tokens: &[Token],
        right_tokens: &[Token],
    ) -> Result<Statement, ParseError> {
        if left_tokens.is_empty() {
            return Err(ParseError::InvalidSyntax(
                "Missing variable name".to_string(),
            ));
        }

        let name = match &left_tokens[0] {
            Token::Identifier(name) => Ok(name.clone()),
            _ => Err(ParseError::InvalidSyntax(
                "Expected variable name".to_string(),
            )),
        }?;

        match left_tokens.len() {
            1 => {
                // Simple assignment: x = ...
                let expression = Self::parse_expression_from_tokens(right_tokens)?;
                Ok(Statement::Assignment {
                    name,
                    value: Symbol::Variable(expression),
                })
            }
            _ => {
                // Function definition: f(params...) = ...
                let params = Self::parse_function_parameters(left_tokens)?;
                let body = Self::parse_expression_from_tokens(right_tokens)?;

                let value = Symbol::Function(FunctionDefinition { params, body });
                Ok(Statement::Assignment { name, value })
            }
        }
    }

    fn parse_function_parameters(tokens: &[Token]) -> Result<Vec<String>, ParseError> {
        if tokens.len() < 3 {
            return Err(ParseError::InvalidSyntax(
                "Invalid function definition".to_string(),
            ));
        }

        if tokens[1] != Token::LeftParen {
            return Err(ParseError::InvalidSyntax(
                "Expected '(' after function name".to_string(),
            ));
        }

        // Check for closing parenthesis
        if tokens.iter().last() != Some(&Token::RightParen) {
            return Err(ParseError::InvalidSyntax(
                "Expected ')' at the end of function parameters".to_string(),
            ));
        }

        // Extract parameters between parentheses
        let param_tokens = &tokens[2..tokens.len() - 1];
        let mut params = Vec::new();

        for token in param_tokens {
            match token {
                Token::Identifier(param) => params.push(param.clone()),
                Token::Comma => {} // Skip commas
                _ => return Err(ParseError::InvalidSyntax("Invalid parameter".to_string())),
            }
        }

        Ok(params)
    }

    fn parse_expression_from_tokens(tokens: &[Token]) -> Result<Expression, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::UnexpectedEof);
        }

        Self::parse_addition(tokens, &mut 0)
    }

    fn parse_addition(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = Self::parse_multiplication(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos] {
                Token::Plus => {
                    *pos += 1;
                    let right = Self::parse_multiplication(tokens, pos)?;
                    left = Expression::Add(Box::new(left), Box::new(right));
                }
                Token::Minus => {
                    *pos += 1;
                    let right = Self::parse_multiplication(tokens, pos)?;
                    left = Expression::Sub(Box::new(left), Box::new(right));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplication(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        let mut left = Self::parse_power(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos] {
                Token::Mul => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mul(Box::new(left), Box::new(right));
                }
                Token::MatMul => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::MatMul(Box::new(left), Box::new(right));
                }
                Token::Divide => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Div(Box::new(left), Box::new(right));
                }
                Token::Modulo => {
                    *pos += 1;
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mod(Box::new(left), Box::new(right));
                }
                Token::Identifier(_) | Token::LeftParen | Token::Imaginary => {
                    let right = Self::parse_power(tokens, pos)?;
                    left = Expression::Mul(Box::new(left), Box::new(right));
                }
                Token::Number(_) => {
                    // Disallow number after any expression (ex: a5)
                    return Err(ParseError::InvalidSyntax(
                        "Numbers cannot directly follow expressions".to_string(),
                    ));
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = Self::parse_unary(tokens, pos)?;

        if *pos < tokens.len() && tokens[*pos] == Token::Power {
            *pos += 1;
            let right = Self::parse_power(tokens, pos)?; // Right associative
            left = Expression::Pow(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    fn parse_unary(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match &tokens[*pos] {
            Token::Plus => {
                *pos += 1;
                let inner = Self::parse_primary(tokens, pos)?;
                Ok(inner)
            }
            Token::Minus => {
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

        match &tokens[*pos] {
            Token::Number(n) => {
                *pos += 1;
                Ok(Expression::Real(*n))
            }
            Token::Imaginary => {
                *pos += 1;
                Ok(Expression::Complex(0.0, 1.0))
            }
            Token::LeftBracket => Self::parse_bracket(tokens, pos),
            Token::Identifier(name) => Self::parse_identifier(tokens, pos, name.clone()),
            Token::LeftParen => Self::parse_parenthesized_expression(tokens, pos),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", tokens[*pos]))),
        }
    }

    fn parse_identifier(
        tokens: &[Token],
        pos: &mut usize,
        name: String,
    ) -> Result<Expression, ParseError> {
        *pos += 1;

        // Check for function call
        if *pos < tokens.len() && tokens[*pos] == Token::LeftParen {
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
        while *pos < tokens.len() && tokens[*pos] != Token::RightParen {
            let arg = Self::parse_addition(tokens, pos)?;
            args.push(arg);

            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
            return Err(ParseError::InvalidSyntax(
                "Missing closing parenthesis".to_string(),
            ));
        }
        *pos += 1; // consume ')'

        Ok(Expression::FunctionCall(FunctionCall { name, args }))
    }

    fn parse_parenthesized_expression(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('
        let expr = Self::parse_addition(tokens, pos)?;

        if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
            return Err(ParseError::InvalidSyntax(
                "Missing closing parenthesis".to_string(),
            ));
        }
        *pos += 1; // consume ')'

        Ok(Expression::Paren(Box::new(expr)))
    }

    fn parse_bracket(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos + 1 >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match tokens[*pos + 1] {
            Token::LeftBracket => Self::parse_matrix(tokens, pos),
            _ => Self::parse_vector(tokens, pos),
        }
    }

    fn parse_matrix(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume '['

        let mut rows = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let row = Self::parse_matrix_row(tokens, pos)?;

            // Validate that all elements in the row are not matrices
            for element in &row {
                if let Expression::Matrix(_, _, _) = element {
                    return Err(ParseError::InvalidMatrix(
                        "Matrix elements cannot be matrices".to_string(),
                    ));
                }
            }

            rows.push(row);

            if *pos < tokens.len() && tokens[*pos] == Token::Semicolon {
                *pos += 1; // consume ';'
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing ']' for matrix".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        if rows.is_empty() {
            return Err(ParseError::InvalidMatrix("Empty matrix".to_string()));
        }

        let row_length = rows[0].len();
        if rows.iter().any(|r| r.len() != row_length) {
            return Err(ParseError::InvalidMatrix("Rows are not all the same length".to_string()))
        }

        Ok(Expression::Matrix(            
            rows.iter().flatten().cloned().collect(),
            rows.len(),
            rows[0].len(),
        ))
    }

    fn parse_matrix_row(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Vec<Expression>, ParseError> {
        if tokens[*pos] != Token::LeftBracket {
            return Err(ParseError::InvalidSyntax(
                "Expected '[' for matrix row".to_string(),
            ));
        }

        *pos += 1; // consume '['
        let mut row = Vec::new();

        // Parse row elements
        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let expr = Self::parse_addition(tokens, pos)?;
            row.push(expr);

            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing ']' for matrix row".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        Ok(row)
    }

    fn parse_vector(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume '['

        let mut elements = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let expr = Self::parse_addition(tokens, pos)?;
            elements.push(expr);

            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing ']' for vector".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        Ok(Expression::Vector(
            elements,
        ))
    }
}
