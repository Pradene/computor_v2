pub mod computor;
pub mod error;
pub mod expression;
pub mod parser;
pub mod tokenizer;
pub mod types;

use {
    crate::{
        computor::Computor,
        parser::{Instruction, Parser},
    },
    rustyline::{error::ReadlineError, Config, Editor, Result as RustylineResult},
    std::process::Command,
};

fn main() -> RustylineResult<()> {
    let config = Config::builder().history_ignore_dups(true)?.build();
    let mut reader = Editor::<(), _>::with_config(config)?;

    let history_file = "history.txt";
    let _ = reader.load_history(history_file);

    let mut computor = Computor::new();

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                }

                reader.add_history_entry(line)?;

                let statement = match Parser::parse(line) {
                    Ok(stmt) => stmt,
                    Err(e) => {
                        eprintln!("{}", e);
                        continue;
                    }
                };

                match statement {
                    Instruction::Assignment { name, value } => {
                        match computor.assign(name, value) {
                            Ok(expression) => println!("{}", expression),
                            Err(error) => eprintln!("{}", error),
                        };
                    }
                    Instruction::Query { expression } => {
                        match computor.evaluate_expression(&expression) {
                            Ok(expression) => println!("{}", expression),
                            Err(error) => eprintln!("{}", error),
                        };
                    }
                    Instruction::Equation { left, right } => {
                        match computor.evaluate_equation(&left, &right) {
                            Ok(solution) => println!("{}", solution),
                            Err(error) => eprintln!("{}", error),
                        };
                    }
                    Instruction::Command { name, args } => match name.as_str() {
                        "quit" => break,
                        "table" => print!("{}", computor),
                        "clear" => {
                            Command::new("clear").status().unwrap();
                        }
                        "unset" => {
                            if let Some(symbol) = computor.unset(args[0].as_str()) {
                                println!("'{}' unset", symbol);
                            }
                        }
                        _ => eprintln!("Not a valid command"),
                    },
                };
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("Ctrl-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                eprintln!("Ctrl-D");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    reader.save_history(history_file)?;

    Ok(())
}
