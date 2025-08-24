use crate::error::ParseError;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Number(f64),
    Identifier(String),
    Imaginary,
    Plus,
    Minus,
    Multiply,
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

pub struct Tokenizer {
    input: Vec<char>,
    position: usize,
}

impl Tokenizer {
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
        }
    }

    pub fn tokenize(&mut self) -> Result<Vec<Token>, ParseError> {
        let mut tokens = Vec::new();

        loop {
            let token = self.next_token()?;
            let is_eof = token == Token::Eof;
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        Ok(tokens)
    }

    fn current_char(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    fn peek_char(&self, offset: usize) -> Option<char> {
        self.input.get(self.position + offset).copied()
    }

    fn advance(&mut self) {
        self.position += 1;
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.current_char() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> Result<f64, ParseError> {
        let start = self.position;

        while let Some(ch) = self.current_char() {
            if ch.is_ascii_digit() || ch == '.' {
                self.advance();
            } else {
                break;
            }
        }

        let number_str: String = self.input[start..self.position].iter().collect();
        number_str
            .parse::<f64>()
            .map_err(|_| ParseError::InvalidNumber(number_str))
    }

    fn read_identifier(&mut self) -> String {
        let start = self.position;

        while let Some(ch) = self.current_char() {
            if ch.is_ascii_alphanumeric() {
                self.advance();
            } else {
                break;
            }
        }

        self.input[start..self.position].iter().collect()
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace();

        match self.current_char() {
            None => Ok(Token::Eof),
            Some(ch) => match ch {
                '+' => {
                    self.advance();
                    Ok(Token::Plus)
                }
                '-' => {
                    self.advance();
                    Ok(Token::Minus)
                }
                '*' => {
                    self.advance();
                    Ok(Token::Multiply)
                }
                '/' => {
                    self.advance();
                    Ok(Token::Divide)
                }
                '%' => {
                    self.advance();
                    Ok(Token::Modulo)
                }
                '^' => {
                    self.advance();
                    Ok(Token::Power)
                }
                '(' => {
                    self.advance();
                    Ok(Token::LeftParen)
                }
                ')' => {
                    self.advance();
                    Ok(Token::RightParen)
                }
                '=' => {
                    self.advance();
                    Ok(Token::Equal)
                }
                '[' => {
                    self.advance();
                    Ok(Token::LeftBracket)
                }
                ']' => {
                    self.advance();
                    Ok(Token::RightBracket)
                }
                ';' => {
                    self.advance();
                    Ok(Token::Semicolon)
                }
                '?' => {
                    self.advance();
                    Ok(Token::Question)
                }
                ',' => {
                    self.advance();
                    Ok(Token::Comma)
                }
                'i' => {
                    if let Some(next) = self.peek_char(1) {
                        if next.is_ascii_alphanumeric() {
                            let ident = self.read_identifier();
                            Ok(Token::Identifier(ident))
                        } else {
                            self.advance();
                            Ok(Token::Imaginary)
                        }
                    } else {
                        self.advance();
                        Ok(Token::Imaginary)
                    }
                }
                _ if ch.is_ascii_digit() => {
                    let num = self.read_number()?;
                    Ok(Token::Number(num))
                }
                _ if ch.is_ascii_alphabetic() => {
                    let ident = self.read_identifier();
                    Ok(Token::Identifier(ident))
                }
                _ => Err(ParseError::UnexpectedToken(ch.to_string())),
            },
        }
    }
}
