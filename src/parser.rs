use crate::ast::{BinaryOperator, Expression, UnaryOperator, Value};
use crate::error::ParseError;
use crate::matrix::Matrix;
use crate::tokenizer::{Token, Tokenizer};

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Assignment { name: String, value: Value },
    Query { expression: Expression },
}

pub struct LineParser;

impl LineParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(&self, line: &str) -> Result<ParsedLine, ParseError> {
        let mut tokenizer = Tokenizer::new(line);
        let tokens = tokenizer.tokenize()?;

        if tokens.is_empty() || tokens.len() == 1 {
            return Err(ParseError::InvalidSyntax("Empty line".to_string()));
        }

        // Check for query pattern (expression = ?)
        if let Some(eq_pos) = tokens.iter().position(|t| *t == Token::Equal) {
            if eq_pos + 2 < tokens.len() && tokens[eq_pos + 1] == Token::Question {
                let expr_tokens = &tokens[..eq_pos];
                let expression = self.parse_expression_from_tokens(expr_tokens)?;
                return Ok(ParsedLine::Query { expression });
            }
        }

        // Check for assignment pattern
        if let Some(eq_pos) = tokens.iter().position(|t| *t == Token::Equal) {
            if eq_pos == 0 {
                return Err(ParseError::InvalidSyntax(
                    "Missing variable name".to_string(),
                ));
            }

            let name = match &tokens[0] {
                Token::Identifier(name) => name.clone(),
                _ => {
                    return Err(ParseError::InvalidSyntax(
                        "Expected variable name".to_string(),
                    ))
                }
            };

            // Check if it's a function definition
            if tokens.len() > 2 && tokens[1] == Token::LeftParen {
                self.parse_function_definition(&tokens, name)
            } else if eq_pos == 1 {
                // Simple variable assignment
                let expr_tokens = &tokens[eq_pos + 1..tokens.len() - 1]; // exclude EOF
                let expression = self.parse_expression_from_tokens(expr_tokens)?;
                let value = Value::Variable(expression);
                Ok(ParsedLine::Assignment { name, value })
            } else {
                Err(ParseError::InvalidSyntax("Invalid assignment".to_string()))
            }
        } else {
            Err(ParseError::InvalidSyntax(
                "Expected assignment or query".to_string(),
            ))
        }
    }

    fn parse_function_definition(
        &self,
        tokens: &[Token],
        name: String,
    ) -> Result<ParsedLine, ParseError> {
        // Find the closing parenthesis for parameters
        let mut paren_count = 0;
        let mut param_end = None;

        for (i, token) in tokens.iter().enumerate().skip(1) {
            match token {
                Token::LeftParen => paren_count += 1,
                Token::RightParen => {
                    paren_count -= 1;
                    if paren_count == 0 {
                        param_end = Some(i);
                        break;
                    }
                }
                _ => {}
            }
        }

        let param_end = param_end.ok_or(ParseError::InvalidSyntax(
            "Missing closing parenthesis".to_string(),
        ))?;

        // Parse parameters
        let param_tokens = &tokens[2..param_end];
        let mut params = Vec::new();

        for token in param_tokens {
            match token {
                Token::Identifier(param) => params.push(param.clone()),
                Token::Comma => {} // Skip commas
                _ => return Err(ParseError::InvalidSyntax("Invalid parameter".to_string())),
            }
        }

        // Find equals sign
        if param_end + 1 >= tokens.len() || tokens[param_end + 1] != Token::Equal {
            return Err(ParseError::InvalidSyntax(
                "Expected '=' after function parameters".to_string(),
            ));
        }

        // Parse function body
        let body_tokens = &tokens[param_end + 2..tokens.len() - 1]; // exclude EOF
        let body = self.parse_expression_from_tokens(body_tokens)?;

        let value = Value::Function { params, body };
        Ok(ParsedLine::Assignment { name, value })
    }

    fn parse_expression_from_tokens(&self, tokens: &[Token]) -> Result<Expression, ParseError> {
        if tokens.is_empty() {
            return Err(ParseError::UnexpectedEof);
        }

        self.parse_addition(tokens, &mut 0)
    }

    fn parse_addition(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = self.parse_multiplication(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos] {
                Token::Plus => {
                    *pos += 1;
                    let right = self.parse_multiplication(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Add,
                        right: Box::new(right),
                    };
                }
                Token::Minus => {
                    *pos += 1;
                    let right = self.parse_multiplication(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Subtract,
                        right: Box::new(right),
                    };
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_multiplication(
        &self,
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        let mut left = self.parse_power(tokens, pos)?;

        while *pos < tokens.len() {
            match &tokens[*pos] {
                Token::Multiply => {
                    *pos += 1;
                    let right = self.parse_power(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Multiply,
                        right: Box::new(right),
                    };
                }
                Token::Divide => {
                    *pos += 1;
                    let right = self.parse_power(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Divide,
                        right: Box::new(right),
                    };
                }
                // Handle implicit multiplication (like "3b")
                Token::Identifier(_) | Token::LeftParen => {
                    let right = self.parse_power(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Multiply,
                        right: Box::new(right),
                    };
                }
                _ => break,
            }
        }

        Ok(left)
    }

    fn parse_power(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        let mut left = self.parse_unary(tokens, pos)?;

        if *pos < tokens.len() && tokens[*pos] == Token::Power {
            *pos += 1;
            let right = self.parse_power(tokens, pos)?; // Right associative
            left = Expression::BinaryOp {
                left: Box::new(left),
                op: BinaryOperator::Power,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_unary(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match &tokens[*pos] {
            Token::Plus => {
                *pos += 1;
                let operand = self.parse_primary(tokens, pos)?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Plus,
                    operand: Box::new(operand),
                })
            }
            Token::Minus => {
                *pos += 1;
                let operand = self.parse_primary(tokens, pos)?;
                Ok(Expression::UnaryOp {
                    op: UnaryOperator::Minus,
                    operand: Box::new(operand),
                })
            }
            _ => self.parse_primary(tokens, pos),
        }
    }

    fn parse_primary(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        match &tokens[*pos] {
            Token::Number(n) => {
                *pos += 1;
                Ok(Expression::Number(*n))
            }
            Token::LeftBracket => self.parse_matrix(tokens, pos),
            Token::Identifier(name) => {
                *pos += 1;
                // Check for function call
                if *pos < tokens.len() && tokens[*pos] == Token::LeftParen {
                    // consume '('
                    *pos += 1;
                    let mut args = Vec::new();

                    // Parse arguments
                    while *pos < tokens.len() && tokens[*pos] != Token::RightParen {
                        let arg = self.parse_addition(tokens, pos)?;
                        args.push(arg);

                        if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                            // consume ','
                            *pos += 1;
                        }
                    }

                    if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
                        return Err(ParseError::InvalidSyntax(
                            "Missing closing parenthesis".to_string(),
                        ));
                    }
                    // consume ')'
                    *pos += 1;

                    Ok(Expression::FunctionCall {
                        name: name.clone(),
                        args,
                    })
                } else {
                    Ok(Expression::Variable(name.clone()))
                }
            }
            Token::LeftParen => {
                // consume '('
                *pos += 1;
                let expr = self.parse_addition(tokens, pos)?;

                if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
                    return Err(ParseError::InvalidSyntax(
                        "Missing closing parenthesis".to_string(),
                    ));
                }
                // consume ')'
                *pos += 1;

                Ok(expr)
            }
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", tokens[*pos]))),
        }
    }

    fn parse_matrix(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        // consume '['
        *pos += 1;

        let mut rows = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            if tokens[*pos] != Token::LeftBracket {
                return Err(ParseError::InvalidSyntax(
                    "Expected '[' for matrix row".to_string(),
                ));
            }

            // consume '['
            *pos += 1;
            let mut row = Vec::new();

            // Parse row elements
            while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
                let expr = self.parse_addition(tokens, pos)?;

                // For now, only allow numbers in matrices
                if let Expression::Number(n) = expr {
                    row.push(n);
                } else {
                    return Err(ParseError::InvalidSyntax(
                        "Only numbers allowed in matrices".to_string(),
                    ));
                }

                if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                    // consume ','
                    *pos += 1;
                }
            }

            if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
                return Err(ParseError::InvalidSyntax(
                    "Missing ']' for matrix row".to_string(),
                ));
            }
            // consume ']'
            *pos += 1;

            rows.push(row);

            if *pos < tokens.len() && tokens[*pos] == Token::Semicolon {
                // consume ';'
                *pos += 1;
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing ']' for matrix".to_string(),
            ));
        }
        // consume ']'
        *pos += 1;

        let matrix = Matrix::from_numbers(rows).map_err(|e| ParseError::InvalidSyntax(e))?;

        Ok(Expression::Matrix(matrix))
    }
}

impl Default for LineParser {
    fn default() -> Self {
        Self::new()
    }
}
