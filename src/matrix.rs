use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Matrix cannot be empty".to_string());
        }

        let rows = data.len();
        let cols = data[0].len();

        // Validate that all rows have the same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != cols {
                return Err(format!(
                    "Row {} has {} elements, expected {}",
                    i,
                    row.len(),
                    cols
                ));
            }
        }

        Ok(Matrix { data, rows, cols })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        self.data.get(row)?.get(col).copied()
    }

    pub fn data(&self) -> &Vec<Vec<f64>> {
        &self.data
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, row) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ";")?;
            }
            write!(f, "[")?;
            for (j, val) in row.iter().enumerate() {
                if j > 0 {
                    write!(f, ",")?;
                }
                if val.fract() == 0.0 {
                    write!(f, "{}", *val as i64)?;
                } else {
                    write!(f, "{}", val)?;
                }
            }
            write!(f, "]")?;
        }
        write!(f, "]")?;

        Ok(())
    }
}
