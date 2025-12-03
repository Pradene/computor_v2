use {
    crate::{
        constant::EPSILON,
        context::{Context, Symbol, Variable},
        error::EvaluationError,
        expression::Expression,
    },
    std::{
        collections::HashMap,
        ops::{Add, Div, Mul, Neg, Rem, Sub},
    },
};

impl Expression {
    pub fn evaluate(&self, context: &Context) -> Result<Expression, EvaluationError> {
        self.evaluate_internal(context, &HashMap::new())
    }

    pub fn evaluate_internal(
        &self,
        context: &Context,
        scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(_) => Ok(self.clone()),
            Expression::Complex(_, _) => Ok(self.clone()),

            Expression::Vector(vector) => {
                let mut evaluated_vector = Vec::new();
                for element in vector.iter() {
                    let evaluated_element = element.evaluate_internal(context, scope)?.reduce()?;
                    evaluated_vector.push(evaluated_element);
                }
                Ok(Expression::Vector(evaluated_vector))
            }

            Expression::Matrix(matrix, rows, cols) => {
                let mut evaluated_matrix = Vec::new();
                for element in matrix.iter() {
                    let evaluated_element = element.evaluate_internal(context, scope)?.reduce()?;
                    evaluated_matrix.push(evaluated_element);
                }
                Ok(Expression::Matrix(evaluated_matrix, *rows, *cols))
            }

            Expression::Variable(name) => {
                // First check function parameter scope
                if let Some(expression) = scope.get(name) {
                    return Ok(expression.clone());
                }

                // Then check context variables
                match context.get_symbol(name) {
                    Some(Symbol::Variable(Variable { expression, .. })) => {
                        // Check for self-reference before recursing
                        if expression.contains_variable(name) {
                            return Err(EvaluationError::InvalidOperation(format!(
                                "Variable '{}' is defined in terms of itself",
                                name
                            )));
                        }
                        // Recursively evaluate the variable's expression
                        expression.evaluate_internal(context, scope)
                    }
                    Some(Symbol::Function(_) | Symbol::BuiltinFunction(_)) => {
                        Err(EvaluationError::InvalidOperation(format!(
                            "Cannot use function '{}' as variable",
                            name
                        )))
                    }
                    None => Ok(Expression::Variable(name.to_string())), // Keep symbolic
                }
            }

            Expression::FunctionCall(name, args) => {
                match context.get_symbol(name.as_str()) {
                    Some(Symbol::Function(fun)) => {
                        if args.len() != fun.params.len() {
                            return Err(EvaluationError::WrongArgumentCount {
                                name: name.clone(),
                                expected: fun.params.len(),
                                got: args.len(),
                            });
                        }

                        // Evaluate arguments first
                        let evaluated_args: Result<Vec<_>, _> = args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        let evaluated_args = evaluated_args?;

                        // Create function scope by combining current scope with function parameters
                        let mut function_scope = scope.clone();
                        for (param, arg) in fun.params.iter().zip(evaluated_args.iter()) {
                            function_scope.insert(param.clone(), arg.clone());
                        }

                        // Evaluate function body with new scope
                        fun.body.evaluate_internal(context, &function_scope)
                    }
                    Some(Symbol::Variable(_)) => Err(EvaluationError::InvalidOperation(format!(
                        "'{}' is not a function",
                        name
                    ))),
                    Some(Symbol::BuiltinFunction(function)) => {
                        if args.len() != function.arity() {
                            return Err(EvaluationError::WrongArgumentCount {
                                name: name.clone(),
                                expected: function.arity(),
                                got: args.len(),
                            });
                        }

                        let evaluated_args: Vec<Expression> = args
                            .iter()
                            .map(|e| e.evaluate_internal(context, scope))
                            .collect::<Result<_, _>>()?;

                        function.call(&evaluated_args)
                    }
                    None => {
                        // Evaluate arguments and keep as symbolic function call
                        let evaluated_args: Result<Vec<_>, _> = args
                            .iter()
                            .map(|arg| arg.evaluate_internal(context, scope))
                            .collect();

                        Ok(Expression::FunctionCall(name.clone(), evaluated_args?))
                    }
                }
            }

            Expression::Paren(inner) => {
                let inner = inner.evaluate_internal(context, scope)?.reduce()?;

                match inner {
                    expression if expression.is_concrete() => Ok(expression),
                    expression => Ok(Expression::Paren(Box::new(expression))),
                }
            }

            Expression::Add(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.add(right_eval)
            }
            Expression::Sub(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.sub(right_eval)
            }
            Expression::Mul(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.mul(right_eval)
            }
            Expression::Hadamard(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.hadamard(right_eval)
            }
            Expression::Div(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.div(right_eval)
            }
            Expression::Mod(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.rem(right_eval)
            }
            Expression::Pow(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.reduce()?;
                let right_eval = right.evaluate_internal(context, scope)?.reduce()?;
                left_eval.pow(right_eval)
            }
            Expression::Neg(inner) => inner.evaluate_internal(context, scope)?.neg(),
        }
    }

    pub fn reduce(&self) -> Result<Expression, EvaluationError> {
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
                .distribute()?
                .collect_terms()?
                .simplify_fraction()?;
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
    fn distribute(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Mul(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;

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
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.add(right_dist)
            }
            Expression::Sub(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.sub(right_dist)
            }
            Expression::Neg(inner) => {
                let inner_dist = inner.distribute()?;
                inner_dist.neg()
            }
            Expression::Pow(left, right) => {
                let left_dist = left.distribute()?;
                let right_dist = right.distribute()?;
                left_dist.pow(right_dist)
            }
            _ => Ok(self.clone()),
        }
    }

    fn collect_terms(&self) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Complex(real, imag) => {
                if imag.abs() < EPSILON {
                    Ok(Expression::Real(*real))
                } else {
                    Ok(self.clone())
                }
            }
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
            Expression::Complex(r, i) if i.abs() < EPSILON => Ok(vec![Term::constant(sign * r)]),
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

        self.collect_mul_parts(&mut coefficient, &mut non_constant_parts);

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

    fn collect_mul_parts(&self, coefficient: &mut f64, parts: &mut Vec<Expression>) {
        match self {
            Expression::Real(n) => {
                *coefficient *= n;
            }
            Expression::Complex(r, i) if i.abs() < EPSILON => {
                *coefficient *= r;
            }
            Expression::Mul(left, right) => {
                left.collect_mul_parts(coefficient, parts);
                right.collect_mul_parts(coefficient, parts);
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
            if term.coefficient.abs() < EPSILON {
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

    pub fn simplify_fraction(&self) -> Result<Expression, EvaluationError> {
        let Expression::Div(numerator, denominator) = self else {
            return Ok(self.clone());
        };

        let var_num = numerator.collect_variables();
        let var_den = denominator.collect_variables();

        let mut variables = vec![];
        variables.extend(var_num);
        variables.extend(var_den);
        variables.sort();
        variables.dedup();

        if variables.len() != 1 {
            return Ok(self.clone());
        }

        let var = variables.first().unwrap().clone();

        let num_coeffs = Self::extract_poly_coeffs(numerator, &var)?;
        let den_coeffs = Self::extract_poly_coeffs(denominator, &var)?;

        // Only handle simple cases (degree 0-2)
        let num_degree = num_coeffs.keys().max().copied().unwrap_or(0);
        let den_degree = den_coeffs.keys().max().copied().unwrap_or(0);

        if num_degree > 2 || den_degree > 2 {
            return Ok(self.clone());
        }

        // Find roots of numerator and denominator
        let num_roots = Self::find_polynomial_roots(&num_coeffs)?;
        let den_roots = Self::find_polynomial_roots(&den_coeffs)?;

        // Find common roots (with tolerance for floating point)
        let common_roots = Self::find_common_roots(&num_roots, &den_roots);

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

    /// Extract polynomial coefficients for a given variable
    fn extract_poly_coeffs(
        expr: &Expression,
        var: &str,
    ) -> Result<HashMap<i32, f64>, EvaluationError> {
        let mut coeffs = HashMap::new();
        Self::collect_poly_terms(expr, var, &mut coeffs, 1.0)?;
        Ok(coeffs)
    }

    fn collect_poly_terms(
        expr: &Expression,
        var: &str,
        coeffs: &mut HashMap<i32, f64>,
        sign: f64,
    ) -> Result<(), EvaluationError> {
        match expr {
            Expression::Real(n) => {
                coeffs
                    .entry(0)
                    .and_modify(|c| *c += sign * n)
                    .or_insert(sign * n);
            }
            Expression::Variable(name) if name == var => {
                coeffs.entry(1).and_modify(|c| *c += sign).or_insert(sign);
            }
            Expression::Paren(inner) => {
                // Unwrap parentheses and recurse
                Self::collect_poly_terms(inner, var, coeffs, sign)?;
            }
            Expression::Neg(inner) => {
                // Handle negation
                Self::collect_poly_terms(inner, var, coeffs, -sign)?;
            }
            Expression::Add(left, right) => {
                Self::collect_poly_terms(left, var, coeffs, sign)?;
                Self::collect_poly_terms(right, var, coeffs, sign)?;
            }
            Expression::Sub(left, right) => {
                Self::collect_poly_terms(left, var, coeffs, sign)?;
                Self::collect_poly_terms(right, var, coeffs, -sign)?;
            }
            Expression::Mul(left, right) => {
                // Unwrap parentheses first
                let left_unwrapped = match left.as_ref() {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };
                let right_unwrapped = match right.as_ref() {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };

                // Simple cases: const * var, var * const
                if let (Expression::Real(c), Expression::Variable(name)) =
                    (left_unwrapped, right_unwrapped)
                {
                    if name == var {
                        coeffs
                            .entry(1)
                            .and_modify(|coeff| *coeff += sign * c)
                            .or_insert(sign * c);
                        return Ok(());
                    }
                }
                if let (Expression::Variable(name), Expression::Real(c)) =
                    (left_unwrapped, right_unwrapped)
                {
                    if name == var {
                        coeffs
                            .entry(1)
                            .and_modify(|coeff| *coeff += sign * c)
                            .or_insert(sign * c);
                        return Ok(());
                    }
                }
            }
            Expression::Pow(base, exp) => {
                let base_unwrapped = match base.as_ref() {
                    Expression::Paren(inner) => inner.as_ref(),
                    other => other,
                };

                if let (Expression::Variable(name), Expression::Real(e)) =
                    (base_unwrapped, exp.as_ref())
                {
                    if name == var && *e == 2.0 {
                        coeffs.entry(2).and_modify(|c| *c += sign).or_insert(sign);
                        return Ok(());
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Find roots of a polynomial given its coefficients
    fn find_polynomial_roots(coeffs: &HashMap<i32, f64>) -> Result<Vec<f64>, EvaluationError> {
        let degree = coeffs.keys().max().copied().unwrap_or(0);

        match degree {
            0 => Ok(vec![]), // Constant, no roots
            1 => {
                let a = coeffs.get(&1).copied().unwrap_or(0.0);
                let b = coeffs.get(&0).copied().unwrap_or(0.0);
                if a.abs() < EPSILON {
                    Ok(vec![])
                } else {
                    Ok(vec![-b / a])
                }
            }
            2 => {
                let a = coeffs.get(&2).copied().unwrap_or(0.0);
                let b = coeffs.get(&1).copied().unwrap_or(0.0);
                let c = coeffs.get(&0).copied().unwrap_or(0.0);

                let discriminant = b * b - 4.0 * a * c;
                if discriminant < 0.0 {
                    Ok(vec![]) // Complex roots, skip
                } else if discriminant.abs() < EPSILON {
                    let root = -b / (2.0 * a);
                    Ok(vec![root, root])
                } else {
                    let sqrt_d = discriminant.sqrt();
                    Ok(vec![(-b + sqrt_d) / (2.0 * a), (-b - sqrt_d) / (2.0 * a)])
                }
            }
            _ => Ok(vec![]), // Can't handle higher degrees
        }
    }

    /// Find roots that appear in both lists (with tolerance)
    fn find_common_roots(roots1: &[f64], roots2: &[f64]) -> Vec<f64> {
        let tolerance = EPSILON;
        let mut common = Vec::new();
        let mut used_from_roots2 = vec![false; roots2.len()];

        for &r1 in roots1 {
            for (idx, &r2) in roots2.iter().enumerate() {
                if !used_from_roots2[idx] && (r1 - r2).abs() < tolerance {
                    common.push(r1);
                    used_from_roots2[idx] = true;
                    break; // Only match this root once
                }
            }
        }

        common
    }

    /// Rebuild polynomial without specific roots
    /// For roots [r1, r2], rebuild as (x - r1)(x - r2)...
    fn rebuild_polynomial(all_roots: &[f64], roots_to_remove: &[f64], var: &str) -> Expression {
        let mut remaining = all_roots.to_vec();

        // Remove the common roots
        for &remove in roots_to_remove {
            if let Some(pos) = remaining.iter().position(|&r| (r - remove).abs() < EPSILON) {
                remaining.remove(pos);
            }
        }

        if remaining.is_empty() {
            return Expression::Real(1.0);
        }

        // Rebuild: (var - r1)(var - r2)...
        let mut result = Expression::Sub(
            Box::new(Expression::Variable(var.to_string())),
            Box::new(Expression::Real(remaining[0])),
        );

        for &root in remaining.iter().skip(1) {
            let factor = Expression::Sub(
                Box::new(Expression::Variable(var.to_string())),
                Box::new(Expression::Real(root)),
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

        if coeff_abs < EPSILON {
            return Expression::Real(0.0);
        }

        // Check if expression is just the constant 1.0
        if matches!(self.expression, Expression::Real(n) if (n - 1.0).abs() < EPSILON) {
            return Expression::Real(self.coefficient);
        }

        let abs_coeff = self.coefficient.abs();
        let is_negative = self.coefficient < 0.0;

        let base_expr = if (abs_coeff - 1.0).abs() < EPSILON {
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
