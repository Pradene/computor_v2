use crate::context::{Statement, Variable};
use crate::tokenizer::{Token, Tokenizer};

use crate::context::{FunctionDefinition, Symbol};
use crate::error::ParseError;
use crate::expression::{Expression, FunctionCall};

pub struct Parser;

impl Parser {
    pub fn parse(line: &str) -> Result<Statement, ParseError> {
        let tokens = Tokenizer::tokenize(&line)?;

        if tokens.is_empty() {
            return Err(ParseError::InvalidSyntax(
                "Empty line: please enter a valid assignment, equation, or query".to_string(),
            ));
        }

        if let Some(eq_pos) = Self::find_equals_position(&tokens) {
            let left_tokens = &tokens[..eq_pos];
            let right_tokens = &tokens[eq_pos + 1..];

            // Check if right side is empty
            if right_tokens.is_empty() {
                return Err(ParseError::InvalidSyntax(
                    "Missing expression after '=': expected a value or expression on the right side".to_string(),
                ));
            }

            // Look for (... = ... ?)
            if right_tokens.iter().last() == Some(&Token::Question) {
                if right_tokens.len() == 1 {
                    let expression = Self::parse_expression_from_tokens(left_tokens)?;
                    Ok(Statement::Query { expression })
                } else {
                    let left = Self::parse_expression_from_tokens(left_tokens)?;
                    let right = Self::parse_expression_from_tokens(
                        &right_tokens[..right_tokens.len() - 1],
                    )?;
                    Ok(Statement::Equation { left, right })
                }
            } else {
                // Handle assignment
                Self::parse_assignment(left_tokens, right_tokens)
            }
        } else {
            Err(ParseError::InvalidSyntax(
                "No assignment operator found: expected format 'x = value' or 'x = y?'".to_string(),
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
                "Missing variable name: the left side of '=' must start with a variable name"
                    .to_string(),
            ));
        }

        // Check if this is a function definition: must have parentheses
        let is_function_definition = left_tokens.len() >= 3
            && left_tokens[1] == Token::LeftParen
            && left_tokens.last() == Some(&Token::RightParen);

        if is_function_definition {
            // Function definition: f(params...) = ...
            let name = match &left_tokens[0] {
                Token::Identifier(name) => Ok(name.clone()),
                other => Err(ParseError::InvalidSyntax(format!(
                    "Invalid function name: expected an identifier, got {:?}",
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
            Ok(Statement::Assignment { name, value })
        } else if left_tokens.len() == 1 {
            // Simple assignment: x = ...
            let name = match &left_tokens[0] {
                Token::Identifier(name) => Ok(name.clone()),
                other => Err(ParseError::InvalidSyntax(format!(
                    "Invalid variable name: expected an identifier, got {:?}",
                    other
                ))),
            }?;

            let expression = Self::parse_expression_from_tokens(right_tokens)?;
            Ok(Statement::Assignment {
                name: name.clone(),
                value: Symbol::Variable(Variable { name, expression }),
            })
        } else {
            // Invalid: can't assign to expressions like a*b
            Err(ParseError::InvalidSyntax(
                format!(
                    "Invalid left side of assignment: got {} tokens but expected a single variable or function definition. \
                     Did you mean to write a function like 'f(x) = ...' or a simple assignment like 'x = ...'?",
                    left_tokens.len()
                )
            ))
        }
    }

    fn parse_function_parameters(tokens: &[Token]) -> Result<Vec<String>, ParseError> {
        let mut params = Vec::new();
        let mut expect_identifier = true;

        for (i, token) in tokens.iter().enumerate() {
            match token {
                Token::Identifier(param) => {
                    if !expect_identifier {
                        return Err(ParseError::InvalidSyntax(
                            format!(
                                "Unexpected parameter at position {}: expected comma separator, got identifier '{}'",
                                i, param
                            )
                        ));
                    }
                    params.push(param.clone());
                    expect_identifier = false;
                }
                Token::Comma => {
                    if expect_identifier {
                        return Err(ParseError::InvalidSyntax(
                            format!(
                                "Unexpected comma at position {}: no parameter before this comma",
                                i
                            )
                        ));
                    }
                    expect_identifier = true;
                }
                other => {
                    return Err(ParseError::InvalidSyntax(format!(
                        "Invalid token in parameter list at position {}: got {:?}, expected parameter name or comma",
                        i, other
                    )))
                }
            }
        }

        // Check if we ended expecting an identifier (trailing comma)
        if expect_identifier && !params.is_empty() {
            return Err(ParseError::InvalidSyntax(
                "Trailing comma in parameter list: remove the comma before the closing parenthesis"
                    .to_string(),
            ));
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
                "Unexpected token at position {}: got {:?}. All tokens must form a single valid expression",
                pos, tokens[pos]
            )));
        }

        Ok(expr)
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

    fn parse_multiplication(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
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
                    return Err(ParseError::InvalidSyntax(
                        format!(
                            "Number cannot directly follow an expression at token position {}: \
                             numbers must be separated from expressions (e.g., use '5 * x' instead of '5x', or '2 * (a+b)' instead of '2(a+b)')",
                            pos
                        )
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
            let right = Self::parse_power(tokens, pos)?;
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
            other => Err(ParseError::UnexpectedToken(format!(
                "Unexpected token at position {}: got {:?}, expected a number, variable, '(', '[', or '-'",
                pos, other
            ))),
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
            return Err(ParseError::InvalidSyntax(format!(
                "Missing closing parenthesis in function call '{}': expected ')' after arguments",
                name
            )));
        }
        *pos += 1; // consume ')'

        Ok(Expression::FunctionCall(FunctionCall { name, args }))
    }

    fn parse_parenthesized_expression(
        tokens: &[Token],
        pos: &mut usize,
    ) -> Result<Expression, ParseError> {
        *pos += 1; // consume '('
        let expression = Self::parse_addition(tokens, pos)?;

        if *pos >= tokens.len() || tokens[*pos] != Token::RightParen {
            return Err(ParseError::InvalidSyntax(
                "Missing closing parenthesis: expected ')' to match opening '('".to_string(),
            ));
        }
        *pos += 1; // consume ')'

        Ok(Expression::Paren(Box::new(expression)))
    }

    fn parse_bracket(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        if *pos >= tokens.len() {
            return Err(ParseError::UnexpectedEof);
        }

        // Look ahead to determine if this is a matrix or vector
        // A matrix starts with [[, a vector is just [
        let is_matrix = if *pos + 1 < tokens.len() {
            tokens[*pos + 1] == Token::LeftBracket
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
        if *pos >= tokens.len() || tokens[*pos] != Token::LeftBracket {
            return Err(ParseError::InvalidMatrix(
                "Invalid matrix syntax: expected '[' to start first row, got a vector instead. \
                 Format: [[row1]; [row2]; ...] where each row is in brackets"
                    .to_string(),
            ));
        }

        let mut rows = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let row = Self::parse_matrix_row(tokens, pos)?;

            // Validate that all elements in the row are not matrices
            for element in &row {
                if let Expression::Matrix(_, _, _) = element {
                    return Err(ParseError::InvalidMatrix(
                        "Matrix elements cannot be nested matrices: use a single level of matrices only"
                            .to_string(),
                    ));
                }
            }

            rows.push(row);

            if *pos < tokens.len() && tokens[*pos] == Token::Semicolon {
                *pos += 1; // consume ';'
            } else if *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
                return Err(ParseError::InvalidMatrix(
                    format!(
                        "Invalid matrix syntax at position {}: expected ';' to separate rows or ']' to end matrix, got {:?}",
                        pos, tokens[*pos]
                    )
                ));
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing closing bracket ']' for matrix".to_string(),
            ));
        }
        *pos += 1; // consume final ']'

        if rows.is_empty() {
            return Err(ParseError::InvalidMatrix(
                "Empty matrix: matrix must contain at least one row with one element".to_string(),
            ));
        }

        let row_length = rows[0].len();
        if rows.iter().any(|r| r.len() != row_length) {
            let lengths: Vec<usize> = rows.iter().map(|r| r.len()).collect();
            return Err(ParseError::InvalidMatrix(format!(
                "Rows have inconsistent lengths: first row has {} elements, but other rows have {}. All rows must have the same length",
                row_length,
                lengths
                    .iter()
                    .skip(1)
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        Ok(Expression::Matrix(
            rows.iter().flatten().cloned().collect(),
            rows.len(),
            rows[0].len(),
        ))
    }

    fn parse_matrix_row(tokens: &[Token], pos: &mut usize) -> Result<Vec<Expression>, ParseError> {
        if tokens[*pos] != Token::LeftBracket {
            return Err(ParseError::InvalidSyntax(format!(
                "Expected '[' to start matrix row at position {}, got {:?}",
                pos, tokens[*pos]
            )));
        }

        *pos += 1; // consume '['
        let mut row = Vec::new();

        // Parse row elements
        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let expression = Self::parse_addition(tokens, pos)?;
            row.push(expression);

            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing closing bracket ']' for matrix row".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        Ok(row)
    }

    fn parse_vector(tokens: &[Token], pos: &mut usize) -> Result<Expression, ParseError> {
        *pos += 1; // consume '['

        let mut elements = Vec::new();

        while *pos < tokens.len() && tokens[*pos] != Token::RightBracket {
            let expression = Self::parse_addition(tokens, pos)?;
            elements.push(expression);

            if *pos < tokens.len() && tokens[*pos] == Token::Comma {
                *pos += 1; // consume ','
            }
        }

        if *pos >= tokens.len() || tokens[*pos] != Token::RightBracket {
            return Err(ParseError::InvalidSyntax(
                "Missing closing bracket ']' for vector".to_string(),
            ));
        }
        *pos += 1; // consume ']'

        Ok(Expression::Vector(elements))
    }
}
