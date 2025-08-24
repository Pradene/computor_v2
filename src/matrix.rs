use crate::ast::Expression;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    data: Vec<Vec<Expression>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    pub fn new(data: Vec<Vec<Expression>>) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Matrix cannot be empty".to_string());
        }

        let rows = data.len();
        let cols = data[0].len();

        // Check that all rows have the same length
        for row in &data {
            if row.len() != cols {
                return Err("All matrix rows must have the same length".to_string());
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

    pub fn get(&self, row: usize, col: usize) -> Option<&Expression> {
        self.data.get(row)?.get(col)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Vec<Expression>> {
        self.data.iter()
    }
}
