use rustyline::error::ReadlineError;
use rustyline::{DefaultEditor, Result};

fn main() -> Result<()> {
    let mut reader = DefaultEditor::new()?;

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                if line.clone() == format!("quit") {
                    break;
                }
                reader.add_history_entry(line.as_str())?;
                println!("Line: {}", line);
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
