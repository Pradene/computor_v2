use std::fmt;

use crate::error::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(f64),
    Identifier(String),
    Imaginary,
    Plus,
    Minus,
    Hadamard,
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

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenKind::Number(n) => write!(f, "{}", n),
            TokenKind::Identifier(i) => write!(f, "{}", i),
            TokenKind::Imaginary => write!(f, "i"),
            TokenKind::Plus => write!(f, "+"),
            TokenKind::Minus => write!(f, "-"),
            TokenKind::Hadamard => write!(f, "**"),
            TokenKind::Mul => write!(f, "*"),
            TokenKind::Divide => write!(f, "/"),
            TokenKind::Modulo => write!(f, "%"),
            TokenKind::Power => write!(f, "^"),
            TokenKind::LeftParen => write!(f, "("),
            TokenKind::RightParen => write!(f, ")"),
            TokenKind::LeftBracket => write!(f, "["),
            TokenKind::RightBracket => write!(f, "]"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Equal => write!(f, "="),
            TokenKind::Question => write!(f, "?"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Eof => write!(f, ""),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub position: usize,
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.kind)
    }
}

pub struct Tokenizer;

impl Tokenizer {
    pub fn tokenize(input: &str) -> Result<Vec<Token>, ParseError> {
        let chars: Vec<char> = input.chars().collect();
        let mut tokens = Vec::new();
        let mut position = 0;

        loop {
            let token = Self::next_token(&chars, &mut position)?;
            let is_eof = token.kind == TokenKind::Eof;
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
            None => Ok(Token {
                kind: TokenKind::Eof,
                position: *position,
            }),
            Some(ch) => match ch {
                '+' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Plus,
                        position: *position - 1,
                    })
                }
                '-' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Minus,
                        position: *position - 1,
                    })
                }
                '*' => {
                    *position += 1;
                    if Self::current_char(chars, *position) == Some('*') {
                        *position += 1;
                        Ok(Token {
                            kind: TokenKind::Hadamard,
                            position: *position - 2,
                        })
                    } else {
                        Ok(Token {
                            kind: TokenKind::Mul,
                            position: *position - 1,
                        })
                    }
                }
                '/' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Divide,
                        position: *position - 1,
                    })
                }
                '%' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Modulo,
                        position: *position - 1,
                    })
                }
                '^' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Power,
                        position: *position - 1,
                    })
                }
                '(' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::LeftParen,
                        position: *position - 1,
                    })
                }
                ')' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::RightParen,
                        position: *position - 1,
                    })
                }
                '=' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Equal,
                        position: *position - 1,
                    })
                }
                '[' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::LeftBracket,
                        position: *position - 1,
                    })
                }
                ']' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::RightBracket,
                        position: *position - 1,
                    })
                }
                ';' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Semicolon,
                        position: *position - 1,
                    })
                }
                '?' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Question,
                        position: *position - 1,
                    })
                }
                ',' => {
                    *position += 1;
                    Ok(Token {
                        kind: TokenKind::Comma,
                        position: *position - 1,
                    })
                }
                _ if ch == 'i' || ch == 'I' => {
                    let start = *position;
                    if let Some(next) = Self::peek_char(chars, *position, 1) {
                        if next.is_ascii_alphabetic() {
                            let ident = Self::read_identifier(chars, position);
                            Ok(Token {
                                kind: TokenKind::Identifier(ident),
                                position: start,
                            })
                        } else {
                            *position += 1;
                            Ok(Token {
                                kind: TokenKind::Imaginary,
                                position: start,
                            })
                        }
                    } else {
                        *position += 1;
                        Ok(Token {
                            kind: TokenKind::Imaginary,
                            position: start,
                        })
                    }
                }
                _ if ch.is_ascii_digit() => {
                    let start = *position;
                    let num = Self::read_number(chars, position)?;
                    Ok(Token {
                        kind: TokenKind::Number(num),
                        position: start,
                    })
                }
                _ if ch.is_ascii_alphabetic() => {
                    let start = *position;
                    let ident = Self::read_identifier(chars, position);
                    Ok(Token {
                        kind: TokenKind::Identifier(ident),
                        position: start,
                    })
                }
                _ => Err(ParseError::UnexpectedToken(ch.to_string())),
            },
        }
    }
}
