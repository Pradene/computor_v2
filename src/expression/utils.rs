use std::collections::HashSet;

use crate::expression::Expression;

impl Expression {
    pub fn is_zero(&self) -> bool {
        match self {
            Expression::Real(n) => n.abs() < f64::EPSILON,
            Expression::Complex(r, i) => r.abs() < f64::EPSILON && i.abs() < f64::EPSILON,
            _ => false,
        }
    }

    pub fn is_concrete(&self) -> bool {
        matches!(
            self,
            Expression::Real(_)
                | Expression::Complex(_, _)
                | Expression::Vector(_)
                | Expression::Matrix(_, _, _)
        )
    }

    pub fn is_real(&self) -> bool {
        match self {
            Expression::Real(_) => true,
            Expression::Complex(_, i) => i.abs() < f64::EPSILON,
            _ => false,
        }
    }

    pub fn contains_variable(&self, variable: &str) -> bool {
        match self {
            Expression::Real(_) | Expression::Complex(_, _) => false,
            Expression::Variable(name) => name == variable,
            Expression::Vector(v) => v.iter().any(|e| e.contains_variable(variable)),
            Expression::Matrix(data, _, _) => data.iter().any(|e| e.contains_variable(variable)),
            Expression::FunctionCall(_, args) => args.iter().any(|e| e.contains_variable(variable)),
            Expression::Paren(inner) => inner.contains_variable(variable),
            Expression::Neg(inner) => inner.contains_variable(variable),
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                left.contains_variable(variable) || right.contains_variable(variable)
            }
        }
    }

    pub fn collect_variables(&self) -> HashSet<String> {
        let mut variables = HashSet::new();

        match self {
            Expression::Variable(name) => {
                variables.insert(name.clone());
            }
            Expression::Add(left, right)
            | Expression::Sub(left, right)
            | Expression::Mul(left, right)
            | Expression::Hadamard(left, right)
            | Expression::Div(left, right)
            | Expression::Mod(left, right)
            | Expression::Pow(left, right) => {
                variables.extend(left.collect_variables());
                variables.extend(right.collect_variables());
            }
            Expression::Neg(inner) => {
                variables.extend(inner.collect_variables());
            }
            Expression::Paren(inner) => {
                variables.extend(inner.collect_variables());
            }
            Expression::FunctionCall(_, args) => {
                // Collect variables from function arguments
                for arg in args {
                    variables.extend(arg.collect_variables());
                }
            }
            // Don't collect anything from Real, Complex, or other literal types
            _ => {}
        };

        variables
    }
}
