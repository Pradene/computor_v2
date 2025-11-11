use computor_v2::context::Context;

use rustyline::{error::ReadlineError, Config, Editor, Result as RustylineResult};

fn main() -> RustylineResult<()> {
    let config = Config::builder().history_ignore_dups(true)?.build();
    let mut reader = Editor::<(), _>::with_config(config)?;

    let history_file = "history.txt";
    let _ = reader.load_history(history_file);

    let mut context = Context::new();

    loop {
        match reader.readline("> ") {
            Ok(line) => {
                if line.as_str() == "quit" {
                    break;
                } else if line.trim().is_empty() {
                    continue;
                } else if line.as_str() == "table" {
                    context.print_table();
                } else {
                    context.compute(line.as_str());
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
