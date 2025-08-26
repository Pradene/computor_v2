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

                let line = line.to_lowercase();

                match parser.parse(&line) {
                    Ok(ParsedLine::Assignment { name, value }) => {
                        println!("{}", value);
                        context.assign(name, value);
                    }
                    Ok(ParsedLine::Query { expression }) => {
                        match context.evaluate_expression(&expression) {
                            Ok(result) => println!("{}", result),
                            Err(e) => println!("Evaluation error: {}", e),
                        }
                    }
                    Err(e) => println!("Parse error: {}", e),
                }

                reader.add_history_entry(line.as_str())?;
            }
            Err(ReadlineError::Interrupted) => {
                println!("Ctrl-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("Ctrl-D");
                break;
            }
            Err(err) => {
                println!("Error: {:?}", err);
                break;
            }
        }
    }

    reader.save_history(history_file)?;

    Ok(())
}
