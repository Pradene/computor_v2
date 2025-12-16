use {
    crate::{error::EvaluationError, expression::Expression, types::polynomial},
    std::{
        collections::{HashMap, HashSet},
        ops::{Add, Div, Mul, Neg, Rem, Sub},
    },
};

impl Expression {
    pub fn simplify(&self) -> Result<Expression, EvaluationError> {
        let mut current = self.clone();
        let mut previous = String::new();
        const MAX_ITERATIONS: usize = 32;

        for iteration in 0..MAX_ITERATIONS {
            let current_str = format!("{:?}", current);
            if iteration > 0 && current_str == previous {
                break;
            }
            previous = current_str;

            current = current
                .expand_powers()?
                .distribute_multiplication()?
                .collect_terms()?
                .reduce_rational()?;
        }

        Ok(current)
    }

    /// Expand powers like (a + b)^2 into (a + b) * (a + b)
    fn expand_powers(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Pow(base, right) => {
                // First recursively expand the base and exponent
                let base_expanded = base.expand_powers()?;
                let right_expanded = right.expand_powers()?;

                // Check if we should expand this power
                if let Expression::Real(n) = right_expanded {
                    if n >= 2.0 && n == n.floor() {
                        let n = n as usize;
                        // Expand (a ± b)^n, (a * b)^n, etc.
                        match &base_expanded {
                            Expression::Add(..) | Expression::Sub(..) | Expression::Mul(..) => {
                                // Start with the base
                                let mut result = base_expanded.clone();

                                // Multiply by itself (n-1) times
                                for _ in 1..n {
                                    result = result.mul(base_expanded.clone())?;
                                }
                                return Ok(result);
                            }
                            _ => {}
                        }
                    }
                }

                // If no expansion, recursively expand children
                Ok(Expression::Pow(
                    Box::new(base_expanded),
                    Box::new(right_expanded),
                ))
            }

            Expression::Add(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.add(right_expanded)
            }

            Expression::Sub(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.sub(right_expanded)
            }

            Expression::Mul(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.mul(right_expanded)
            }

            Expression::Hadamard(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.hadamard(right_expanded)
            }

            Expression::Div(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.div(right_expanded)
            }

            Expression::Mod(left, right) => {
                let left_expanded = left.expand_powers()?;
                let right_expanded = right.expand_powers()?;
                left_expanded.rem(right_expanded)
            }

            Expression::Neg(inner) => {
                let inner_expanded = inner.expand_powers()?;
                inner_expanded.neg()
            }

            Expression::Paren(inner) => {
                let inner_expanded = inner.expand_powers()?;
                Ok(Expression::Paren(Box::new(inner_expanded)))
            }

            Expression::Vector(v) => {
                let expanded: Result<Vec<_>, _> = v.iter().map(|e| e.expand_powers()).collect();
                Ok(Expression::Vector(expanded?))
            }

            Expression::Matrix(data, rows, cols) => {
                let expanded: Result<Vec<_>, _> = data.iter().map(|e| e.expand_powers()).collect();
                Ok(Expression::Matrix(expanded?, *rows, *cols))
            }

            Expression::FunctionCall(name, args) => {
                let expanded_args: Result<Vec<_>, _> =
                    args.iter().map(|e| e.expand_powers()).collect();
                Ok(Expression::FunctionCall(name.clone(), expanded_args?))
            }

            // Base cases: Real, Complex, Variable - no expansion needed
            _ => Ok(self.clone()),
        }
    }

    /// Distribute multiplication over addition: a * (b + c) = a*b + a*c
    fn distribute_multiplication(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Mul(left, right) => {
                let left_dist = left.distribute_multiplication()?;
                let right_dist = right.distribute_multiplication()?;

                // Unwrap parentheses to check what's inside
                let left_unwrapped = match &left_dist {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };
                let right_unwrapped = match &right_dist {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };

                match (left_unwrapped, right_unwrapped) {
                    // a * (b + c) = a*b + a*c
                    (_, Expression::Add(b, c)) => {
                        let left_times_b = left_dist.clone().mul((**b).clone())?;
                        let left_times_c = left_dist.clone().mul((**c).clone())?;
                        left_times_b.add(left_times_c)
                    }
                    // a * (b - c) = a*b - a*c
                    (_, Expression::Sub(b, c)) => {
                        let left_times_b = left_dist.clone().mul((**b).clone())?;
                        let left_times_c = left_dist.clone().mul((**c).clone())?;
                        left_times_b.sub(left_times_c)
                    }
                    // (a + b) * c = a*c + b*c
                    (Expression::Add(a, b), _) => {
                        let a_times_right = (**a).clone().mul(right_dist.clone())?;
                        let b_times_right = (**b).clone().mul(right_dist.clone())?;
                        a_times_right.add(b_times_right)
                    }
                    // (a - b) * c = a*c - b*c
                    (Expression::Sub(a, b), _) => {
                        let a_times_right = (**a).clone().mul(right_dist.clone())?;
                        let b_times_right = (**b).clone().mul(right_dist.clone())?;
                        a_times_right.sub(b_times_right)
                    }
                    _ => Ok(Expression::Mul(Box::new(left_dist), Box::new(right_dist))),
                }
            }
            Expression::Add(left, right) => {
                let left_dist = left.distribute_multiplication()?;
                let right_dist = right.distribute_multiplication()?;
                left_dist.add(right_dist)
            }
            Expression::Sub(left, right) => {
                let left_dist = left.distribute_multiplication()?;
                let right_dist = right.distribute_multiplication()?;
                left_dist.sub(right_dist)
            }
            Expression::Neg(inner) => {
                let inner_dist = inner.distribute_multiplication()?;
                inner_dist.neg()
            }
            Expression::Pow(left, right) => {
                let left_dist = left.distribute_multiplication()?;
                let right_dist = right.distribute_multiplication()?;
                left_dist.pow(right_dist)
            }
            _ => Ok(self.clone()),
        }
    }

    fn collect_terms(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Add(..) | Expression::Sub(..) => {
                let terms = self.extract_terms(1.0)?;
                self.combine_like_terms(terms)
            }
            Expression::Mul(left, right) => {
                let left_collected = left.collect_terms()?;
                let right_collected = right.collect_terms()?;

                if left_collected.is_concrete() && right_collected.is_concrete() {
                    return left_collected.mul(right_collected);
                }

                if left_collected == **left && right_collected == **right {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Mul(
                        Box::new(left_collected),
                        Box::new(right_collected),
                    ))
                }
            }
            Expression::Neg(inner) => {
                let collected = inner.collect_terms()?;

                if let Expression::Neg(double_inner) = &collected {
                    Ok(*double_inner.clone())
                } else if collected == **inner {
                    Ok(self.clone())
                } else {
                    Ok(Expression::Neg(Box::new(collected)))
                }
            }
            _ => Ok(self.clone()),
        }
    }

    // Extract all terms from an addition/subtraction expression
    fn extract_terms(&self, sign: f64) -> Result<Vec<Term>, EvaluationError> {
        match self {
            Expression::Add(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(sign)?);
                Ok(terms)
            }
            Expression::Sub(left, right) => {
                let mut terms = left.extract_terms(sign)?;
                terms.extend(right.extract_terms(-sign)?);
                Ok(terms)
            }
            Expression::Neg(inner) => inner.extract_terms(-sign),
            Expression::Real(n) => Ok(vec![Term::constant(sign * n)]),
            Expression::Mul(..) => {
                let (coeff, expression) = self.extract_coefficient();
                Ok(vec![Term::new(sign * coeff, expression)])
            }
            _ => {
                // Variable, Pow, FunctionCall, etc.
                Ok(vec![Term::new(sign, self.clone())])
            }
        }
    }

    // Extract coefficient from multiplication, returning (coefficient, remaining_expression)
    fn extract_coefficient(&self) -> (f64, Expression) {
        let mut coefficient = 1.0;
        let mut non_constant_parts = Vec::new();

        self.collect_multiplication_parts(&mut coefficient, &mut non_constant_parts);

        if non_constant_parts.is_empty() {
            (coefficient, Expression::Real(1.0))
        } else if non_constant_parts.len() == 1 {
            (coefficient, non_constant_parts[0].clone())
        } else {
            let mut result = non_constant_parts[0].clone();
            for part in non_constant_parts.iter().skip(1) {
                result = Expression::Mul(Box::new(result), Box::new(part.clone()));
            }
            (coefficient, result)
        }
    }

    fn collect_multiplication_parts(&self, coefficient: &mut f64, parts: &mut Vec<Expression>) {
        match self {
            Expression::Real(n) => {
                *coefficient *= n;
            }
            Expression::Mul(left, right) => {
                left.collect_multiplication_parts(coefficient, parts);
                right.collect_multiplication_parts(coefficient, parts);
            }
            _ => {
                // Variable, Pow, FunctionCall, etc. - keep as-is
                parts.push(self.clone());
            }
        }
    }

    // Combine terms with the same expression part
    fn combine_like_terms(&self, terms: Vec<Term>) -> Result<Expression, EvaluationError> {
        let mut combined: HashMap<String, Term> = HashMap::new();

        for term in terms {
            let key = term.key();
            combined
                .entry(key)
                .and_modify(|existing| existing.coefficient += term.coefficient)
                .or_insert(term);
        }

        let mut positive_terms: Vec<Term> = Vec::new();
        let mut negative_terms: Vec<Term> = Vec::new();

        for term in combined.into_values() {
            if term.coefficient.abs() < f64::EPSILON {
                continue; // Skip zero terms
            }

            if term.coefficient > 0.0 {
                positive_terms.push(term);
            } else {
                // Store as positive coefficient, we'll subtract later
                negative_terms.push(Term::new(-term.coefficient, term.expression));
            }
        }

        if positive_terms.is_empty() && negative_terms.is_empty() {
            return Ok(Expression::Real(0.0));
        }

        // Build expression starting with positive terms (or first negative if no positive)
        let mut result = if !positive_terms.is_empty() {
            let mut pos_iter = positive_terms.into_iter();
            let mut result = pos_iter.next().unwrap().to_expression();

            for term in pos_iter {
                result = Expression::Add(Box::new(result), Box::new(term.to_expression()));
            }
            result
        } else {
            // All terms are negative, start with negation of first term
            let first = negative_terms.remove(0);
            Expression::Neg(Box::new(first.to_expression()))
        };

        // Subtract all negative terms
        for term in negative_terms {
            result = Expression::Sub(Box::new(result), Box::new(term.to_expression()));
        }

        Ok(result)
    }

    pub fn reduce_rational(&self) -> Result<Expression, EvaluationError> {
        let Expression::Div(numerator, denominator) = self else {
            return Ok(self.clone());
        };

        let var_num = numerator.collect_variables();
        let var_den = denominator.collect_variables();

        let variables: HashSet<String> = var_num.union(&var_den).cloned().collect();

        if variables.len() != 1 {
            return Ok(self.clone());
        }

        let var = variables.iter().next().unwrap().clone();

        let num_coeffs = polynomial::collect_coefficients(numerator, &var)?;
        let den_coeffs = polynomial::collect_coefficients(denominator, &var)?;

        // Only handle simple cases (degree 0-2)
        let num_degree = num_coeffs.keys().max().copied().unwrap_or(0);
        let den_degree = den_coeffs.keys().max().copied().unwrap_or(0);

        if num_degree > 2 || den_degree > 2 {
            return Ok(self.clone());
        }

        // Find roots of numerator and denominator
        let num_roots = numerator.find_roots()?;
        let den_roots = denominator.find_roots()?;

        let common_roots =
            polynomial::find_common_roots(num_roots.as_slice(), den_roots.as_slice());

        if common_roots.is_empty() {
            // No common roots, can't simplify
            return Ok(self.clone());
        }

        // Rebuild fraction without common factors
        let simplified_num = Self::rebuild_polynomial(&num_roots, &common_roots, &var);
        let simplified_den = Self::rebuild_polynomial(&den_roots, &common_roots, &var);

        if simplified_den == Expression::Real(1.0) {
            Ok(simplified_num)
        } else {
            Ok(Expression::Div(
                Box::new(simplified_num),
                Box::new(simplified_den),
            ))
        }
    }

    /// Rebuild polynomial without specific roots
    /// For roots [r1, r2], rebuild as (x - r1)(x - r2)...
    fn rebuild_polynomial(
        all_roots: &[Expression],
        roots_to_remove: &[Expression],
        var: &str,
    ) -> Expression {
        let mut remaining = all_roots.to_vec();

        // Remove the common roots
        for remove_root in roots_to_remove {
            // Find and remove matching root
            if let Some(pos) = remaining.iter().position(|r| match (r, remove_root) {
                (Expression::Real(a), Expression::Real(b)) => (a - b).abs() < f64::EPSILON,
                (Expression::Complex(r1, i1), Expression::Complex(r2, i2)) => {
                    (r1 - r2).abs() < f64::EPSILON && (i1 - i2).abs() < f64::EPSILON
                }
                _ => false,
            }) {
                remaining.remove(pos);
            }
        }

        if remaining.is_empty() {
            return Expression::Real(1.0);
        }

        // Rebuild: (var - r1)(var - r2)...
        let mut result = Expression::Sub(
            Box::new(Expression::Variable(var.to_string())),
            Box::new(remaining[0].clone()),
        );

        for root in remaining.iter().skip(1) {
            let factor = Expression::Sub(
                Box::new(Expression::Variable(var.to_string())),
                Box::new(root.clone()),
            );
            result = Expression::Mul(Box::new(result), Box::new(factor));
        }

        result
    }
}

#[derive(Debug, Clone)]
struct Term {
    coefficient: f64,
    expression: Expression,
}

impl Term {
    fn new(coefficient: f64, expression: Expression) -> Self {
        Term {
            coefficient,
            expression,
        }
    }

    fn constant(value: f64) -> Self {
        Term {
            coefficient: value,
            expression: Expression::Real(1.0),
        }
    }

    fn to_expression(&self) -> Expression {
        let coeff_abs = self.coefficient.abs();

        if coeff_abs < f64::EPSILON {
            return Expression::Real(0.0);
        }

        // Check if expression is just the constant 1.0
        if matches!(self.expression, Expression::Real(n) if (n - 1.0).abs() < f64::EPSILON) {
            return Expression::Real(self.coefficient);
        }

        let abs_coeff = self.coefficient.abs();
        let is_negative = self.coefficient < 0.0;

        let base_expr = if (abs_coeff - 1.0).abs() < f64::EPSILON {
            // Coefficient is ±1, just return the expression (or its negation)
            self.expression.clone()
        } else {
            // Coefficient is not ±1, create multiplication with absolute value
            Expression::Mul(
                Box::new(Expression::Real(abs_coeff)),
                Box::new(self.expression.clone()),
            )
        };

        if is_negative {
            Expression::Neg(Box::new(base_expr))
        } else {
            base_expr
        }
    }

    // Create a unique key for grouping like terms
    fn key(&self) -> String {
        format!("{}", self.expression)
    }
}
