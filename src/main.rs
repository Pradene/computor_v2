use computor_v2::{Context, LineParser, ParsedLine};

use rustyline::{error::ReadlineError, DefaultEditor, Result as RustylineResult};

fn main() -> RustylineResult<()> {
    let mut reader = DefaultEditor::new()?;
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
                        println!("{}", value);
                        context.assign(name, value);
                    }
                    Ok(ParsedLine::Query { expression }) => {
                        match context.evaluate_query(&expression) {
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

    Ok(())
}
