use computor_v2::context::Context;
use computor_v2::parser::{LineParser, ParsedLine};

use rustyline::{error::ReadlineError, Config, Editor, Result as RustylineResult};

fn main() -> RustylineResult<()> {
    let config = Config::builder().history_ignore_dups(true)?.build();
    let history_file = "history.txt";

    let mut reader: Editor<(), _> = Editor::with_config(config)?;
    if reader.load_history(history_file).is_err() {
        println!("No history found.");
    }

    let mut context = Context::new();
    let parser = LineParser::new();

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                if line.clone() == format!("quit") {
                    break;
                }
                if line.trim().is_empty() {
                    continue;
                }

                match parser.parse(&line) {
                    Ok(ParsedLine::Assignment { name, value }) => {
                        match context.assign(name, value) {
                            Ok(result) => println!("{}", result),
                            Err(e) => eprintln!("Assignment error: {}", e),
                        }
                    }
                    Ok(ParsedLine::Query { expression }) => {
                        match context.evaluate_expression(&expression) {
                            Ok(result) => println!("{}", result),
                            Err(e) => eprintln!("Evaluation error: {}", e),
                        }
                    }
                    Ok(ParsedLine::Equation { left, right }) => {
                        match context.evaluate_equation(&left, &right) {
                            Ok(result) => println!("{}", result),
                            Err(e) => eprintln!("Evaluation error: {}", e),
                        }
                    }
                    Err(e) => println!("Parse error: {}", e),
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
