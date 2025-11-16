use crate::error::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Identifier(String),
    Imaginary,
    Plus,
    Minus,
    MatMul,
    Mul,
    Divide,
    Modulo,
    Power,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Semicolon,
    Equal,
    Question,
    Comma,
    Eof,
}

pub struct Tokenizer;

impl Tokenizer {
    pub fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
        let chars: Vec<char> = input.chars().collect();
        let mut tokens = Vec::new();
        let mut position = 0;

        loop {
            let token = Self::next_token(&chars, &mut position)?;
            let is_eof = token == Token::Eof;
            if is_eof {
                break;
            }
            tokens.push(token);
        }

        Ok(tokens)
    }

    fn current_char(chars: &[char], position: usize) -> Option<char> {
        chars.get(position).copied()
    }

    fn peek_char(chars: &[char], position: usize, offset: usize) -> Option<char> {
        chars.get(position + offset).copied()
    }

    fn skip_whitespace(chars: &[char], position: &mut usize) {
        while let Some(ch) = Self::current_char(chars, *position) {
            if ch.is_whitespace() {
                *position += 1;
            } else {
                break;
            }
        }
    }

    fn read_number(chars: &[char], position: &mut usize) -> Result<f64, ParseError> {
        let start = *position;

        while let Some(ch) = Self::current_char(chars, *position) {
            if ch.is_ascii_digit() || ch == '.' {
                *position += 1;
            } else {
                break;
            }
        }

        let number_str: String = chars[start..*position].iter().collect();
        let num = number_str
            .parse::<f64>()
            .map_err(|_| ParseError::InvalidNumber(number_str.clone()))?;

        // Check for overflow/underflow
        if num.is_infinite() {
            return Err(ParseError::Overflow(format!(
                "Number too large: {}",
                number_str
            )));
        }

        Ok(num)
    }

    fn read_identifier(chars: &[char], position: &mut usize) -> String {
        let start = *position;

        while let Some(ch) = Self::current_char(chars, *position) {
            if ch.is_ascii_alphabetic() {
                *position += 1;
            } else {
                break;
            }
        }

        chars[start..*position].iter().collect()
    }

    fn next_token(chars: &[char], position: &mut usize) -> Result<Token, ParseError> {
        Self::skip_whitespace(chars, position);

        match Self::current_char(chars, *position) {
            None => Ok(Token::Eof),
            Some(ch) => match ch {
                '+' => {
                    *position += 1;
                    Ok(Token::Plus)
                }
                '-' => {
                    *position += 1;
                    Ok(Token::Minus)
                }
                '*' => {
                    *position += 1;
                    if Self::current_char(chars, *position) == Some('*') {
                        *position += 1;
                        Ok(Token::MatMul)
                    } else {
                        Ok(Token::Mul)
                    }
                }
                '/' => {
                    *position += 1;
                    Ok(Token::Divide)
                }
                '%' => {
                    *position += 1;
                    Ok(Token::Modulo)
                }
                '^' => {
                    *position += 1;
                    Ok(Token::Power)
                }
                '(' => {
                    *position += 1;
                    Ok(Token::LeftParen)
                }
                ')' => {
                    *position += 1;
                    Ok(Token::RightParen)
                }
                '=' => {
                    *position += 1;
                    Ok(Token::Equal)
                }
                '[' => {
                    *position += 1;
                    Ok(Token::LeftBracket)
                }
                ']' => {
                    *position += 1;
                    Ok(Token::RightBracket)
                }
                ';' => {
                    *position += 1;
                    Ok(Token::Semicolon)
                }
                '?' => {
                    *position += 1;
                    Ok(Token::Question)
                }
                ',' => {
                    *position += 1;
                    Ok(Token::Comma)
                }
                'i' => {
                    if let Some(next) = Self::peek_char(chars, *position, 1) {
                        if next.is_ascii_alphabetic() {
                            let ident = Self::read_identifier(chars, position);
                            Ok(Token::Identifier(ident))
                        } else {
                            *position += 1;
                            Ok(Token::Imaginary)
                        }
                    } else {
                        *position += 1;
                        Ok(Token::Imaginary)
                    }
                }
                _ if ch.is_ascii_digit() => {
                    let num = Self::read_number(chars, position)?;
                    Ok(Token::Number(num))
                }
                _ if ch.is_ascii_alphabetic() => {
                    let ident = Self::read_identifier(chars, position);
                    Ok(Token::Identifier(ident))
                }
                _ => Err(ParseError::UnexpectedToken(ch.to_string())),
            },
        }
    }
}
