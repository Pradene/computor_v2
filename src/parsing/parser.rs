use crate::types::complex::Complex;
use crate::types::matrix::Matrix;

use crate::parsing::tokenizer::{Token, Tokenizer};

use crate::context::ContextValue;
use crate::error::ParseError;
use crate::expression::{BinaryOperator, Expression, UnaryOperator};

#[derive(Debug, Clone)]
pub enum ParsedLine {
    Assignment { name: String, value: ContextValue },
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

        if let Some(eq_pos) = self.find_equals_position(&tokens) {
            // Check for query pattern (expression = ?)
            if self.is_query_pattern(&tokens, eq_pos) {
                let expr_tokens = &tokens[..eq_pos];
                let expression = self.parse_expression_from_tokens(expr_tokens)?;
                return Ok(ParsedLine::Query { expression });
            }

            // Handle assignment pattern
            return self.parse_assignment(&tokens, eq_pos);
        }

        Err(ParseError::InvalidSyntax(
            "Expected assignment or query".to_string(),
        ))
    }

    fn find_equals_position(&self, tokens: &[Token]) -> Option<usize> {
        tokens.iter().position(|t| *t == Token::Equal)
    }

    fn is_query_pattern(&self, tokens: &[Token], eq_pos: usize) -> bool {
        eq_pos + 2 < tokens.len() && tokens[eq_pos + 1] == Token::Question
    }

    fn parse_assignment(&self, tokens: &[Token], eq_pos: usize) -> Result<ParsedLine, ParseError> {
        if eq_pos == 0 {
            return Err(ParseError::InvalidSyntax(
                "Missing variable name".to_string(),
            ));
        }

        let name = self.extract_variable_name(&tokens[0])?;

        // Check if it's a function definition
        if self.is_function_definition(&tokens) {
            self.parse_function_definition(tokens, name)
        } else if eq_pos == 1 {
            // Simple variable assignment
            let expr_tokens = &tokens[eq_pos + 1..tokens.len() - 1];
            let expression = self.parse_expression_from_tokens(expr_tokens)?;
            let value = ContextValue::Variable(expression);
            Ok(ParsedLine::Assignment { name, value })
        } else {
            Err(ParseError::InvalidSyntax("Invalid assignment".to_string()))
        }
    }

    fn extract_variable_name(&self, token: &Token) -> Result<String, ParseError> {
        match token {
            Token::Identifier(name) => Ok(name.clone()),
            _ => Err(ParseError::InvalidSyntax(
                "Expected variable name".to_string(),
            )),
        }
    }

    fn is_function_definition(&self, tokens: &[Token]) -> bool {
        tokens.len() > 2 && tokens[1] == Token::LeftParen
    }

    fn parse_function_definition(
        &self,
        tokens: &[Token],
        name: String,
    ) -> Result<ParsedLine, ParseError> {
        let param_end = self.find_function_params_end(tokens)?;
        let params = self.parse_function_parameters(tokens, param_end)?;
        self.validate_function_equals(tokens, param_end)?;

        let body_tokens = &tokens[param_end + 2..tokens.len() - 1]; // exclude EOF
        let body = self.parse_expression_from_tokens(body_tokens)?;

        let value = ContextValue::Function { params, body };
        Ok(ParsedLine::Assignment { name, value })
    }

    fn find_function_params_end(&self, tokens: &[Token]) -> Result<usize, ParseError> {
        let mut paren_count = 0;

        for (i, token) in tokens.iter().enumerate().skip(1) {
            match token {
                Token::LeftParen => paren_count += 1,
                Token::RightParen => {
                    paren_count -= 1;
                    if paren_count == 0 {
                        return Ok(i);
                    }
                }
                _ => {}
            }
        }

        Err(ParseError::InvalidSyntax(
            "Missing closing parenthesis".to_string(),
        ))
    }

    fn parse_function_parameters(
        &self,
        tokens: &[Token],
        param_end: usize,
    ) -> Result<Vec<String>, ParseError> {
        let param_tokens = &tokens[2..param_end];
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

    fn validate_function_equals(
        &self,
        tokens: &[Token],
        param_end: usize,
    ) -> Result<(), ParseError> {
        if param_end + 1 >= tokens.len() || tokens[param_end + 1] != Token::Equal {
            return Err(ParseError::InvalidSyntax(
                "Expected '=' after function parameters".to_string(),
            ));
        }
        Ok(())
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
                Token::Modulo => {
                    *pos += 1;
                    let right = self.parse_power(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Modulo,
                        right: Box::new(right),
                    };
                }
                Token::Identifier(_) | Token::LeftParen | Token::Imaginary => {
                    let right = self.parse_power(tokens, pos)?;
                    left = Expression::BinaryOp {
                        left: Box::new(left),
                        op: BinaryOperator::Multiply,
                        right: Box::new(right),
                    };
                }
                Token::Number(_) => {
                    // Disallow number after any expression (ex: a5)
                    return Err(ParseError::InvalidSyntax(
                        "Numbers cannot directly follow expressions".to_string()
                    ));
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
                Ok(Expression::Complex(Complex::new(*n, 0.0)))
            }
            Token::Imaginary => {
                *pos += 1;
                Ok(Expression::Complex(Complex::new(0.0, 1.0)))
            }
            Token::LeftBracket => self.parse_matrix(tokens, pos),
            Token::Identifier(name) => self.parse_identifier(tokens, pos, name.clone()),
            Token::LeftParen => self.parse_parenthesized_expression(tokens, pos),
            _ => Err(ParseError::UnexpectedToken(format!("{:?}", tokens[*pos]))),
        }
    }

    fn parse_identifier(
        &self,
        tokens: &[Token],
        pos: &mut usize,
        name: String,
    ) -> Result<Expression, ParseError> {
        *pos += 1;

        // Check for function call
        if *pos < tokens.len() && tokens[*pos] == Token::LeftParen {
            self.parse_function_call(tokens, pos, name)
        } else {
            Ok(Expression::Variable(name))
        }
    }

    fn parse_function_call(
        &self,
        tokens: &[Token],
        pos: &mut usize,
        name: String,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('

        let mut args = Vec::new();

        // Parse arguments
        while *pos < tokens.len() && tokens[*pos] != Token::RightParen {
            let arg = self.parse_addition(tokens, pos)?;
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

        Ok(Expression::FunctionCall { name, args })
    }

    fn parse_parenthesized_expression(
        &self,
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('
        let expr = self.parse_addition(tokens, pos)?;

        if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
            return Err(ParseError::InvalidSyntax(
                "Missing closing parenthesis".to_string(),
            ));
        }
        *pos += 1; // consume ')'

        Ok(expr)
    }

    fn parse_matrix(&self, tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume '['

        let mut rows = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let row = self.parse_matrix_row(tokens, pos)?;
            
            // Validate that all elements in the row are not matrices
            for element in &row {
                if let Expression::Matrix(_) = element {
                    return Err(ParseError::InvalidMatrix(
                        "Matrix elements cannot be matrices".to_string()
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

        Ok(Expression::Matrix(
            Matrix::new(rows.iter().flatten().cloned().collect(), rows.len(), rows[0].len()).map_err(|e| ParseError::InvalidMatrix(e))?,
        ))
    }

    fn parse_matrix_row(
        &self,
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
            let expr = self.parse_addition(tokens, pos)?;
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
}

impl Default for LineParser {
    fn default() -> Self {
        Self::new()
    }
}
