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

    // Create a matrix from f64 values (convenience constructor)
    pub fn from_numbers(data: Vec<Vec<f64>>) -> Result<Self, String> {
        let expr_data: Vec<Vec<Expression>> = data
            .into_iter()
            .map(|row| row.into_iter().map(Expression::Number).collect())
            .collect();
        Self::new(expr_data)
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

    // Try to convert to a numeric matrix if all elements are numbers
    pub fn to_numeric(&self) -> Option<Vec<Vec<f64>>> {
        let mut result = Vec::new();
        for row in &self.data {
            let mut numeric_row = Vec::new();
            for expr in row {
                match expr {
                    Expression::Number(n) => numeric_row.push(*n),
                    _ => return None, // Contains non-numeric expression
                }
            }
            result.push(numeric_row);
        }
        Some(result)
    }
}
