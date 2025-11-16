use rustyline::{error::ReadlineError, Config, Editor, Result as RustylineResult};
use std::process::Command;

use computor_v2::context::Context;

fn main() -> RustylineResult<()> {
    let config = Config::builder().history_ignore_dups(true)?.build();
    let mut reader = Editor::<(), _>::with_config(config)?;

    let history_file = "history.txt";
    let _ = reader.load_history(history_file);

    let mut context = Context::new();

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                let line = line.trim();

                if line.is_empty() {
                    continue;
                } else if line == "quit" {
                    break;
                } else if line == "table" {
                    context.print_table();
                } else if line == "clear" {
                    Command::new("clear").status().unwrap();
                    let words: Vec<&str> = line.split_whitespace().collect();

                    if words.len() != 2 {
                        eprintln!("Usage: unset <name>");
                    } else {
                        match context.unset(words[1]) {
                            Some(symbol) => println!("'{}' unset", symbol),
                            None => eprintln!("Error: '{}' not found", words[1]),
                        }
                    }
                } else {
                    match context.compute(line) {
                        Ok(result) => println!("{}", result),
                        Err(e) => eprintln!("{}", e),
                    }
                }
                reader.add_history_entry(line)?;
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
