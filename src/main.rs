use computor_v2::context::{Context, Statement};
use computor_v2::parser::Parser;

use rustyline::{error::ReadlineError, Config, Editor, Result as RustylineResult};

fn main() -> RustylineResult<()> {
    let config = Config::builder().history_ignore_dups(true)?.build();
    let history_file = "history.txt";

    let mut reader = Editor::<(), _>::with_config(config)?;
    let _ = reader.load_history(history_file);
    let mut context = Context::new();

    let parser = Parser::new();

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                if line.as_str() == "quit" {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }

                match parser.parse(&line) {
                    Ok(Statement::Assignment { name, value }) => {
                        match context.assign(name, value) {
                            Ok(result) => println!("{}", result),
                            Err(e) => {
                                eprintln!("Assignment error: {}", e);
                                continue;
                            }
                        }
                    }
                    Ok(Statement::Query { expression }) => {
                        match context.evaluate_expression(&expression) {
                            Ok(result) => println!("{}", result),
                            Err(e) => {
                                eprintln!("Evaluation error: {}", e);
                                continue;
                            }
                        }
                    }
                    Ok(Statement::Equation { left, right }) => {
                        match context.evaluate_equation(&left, &right) {
                            Ok(result) => println!("{}", result),
                            Err(e) => {
                                eprintln!("Evaluation error: {}", e);
                                continue;
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Parse error: {}", e);
                        continue;
                    }
                }

                reader.add_history_entry(line.as_str())?;
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
