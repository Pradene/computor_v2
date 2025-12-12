use {
    crate::{
        computor::{Computor, Symbol, Variable},
        error::EvaluationError,
        expression::Expression,
    },
    std::{
        collections::HashMap,
        ops::{Add, Div, Mul, Neg, Rem, Sub},
    },
};

impl Expression {
    pub fn evaluate(&self, context: &Computor) -> Result<Expression, EvaluationError> {
        self.evaluate_internal(context, &HashMap::new())
    }

    pub fn evaluate_internal(
        &self,
        context: &Computor,
        scope: &HashMap<String, Expression>,
    ) -> Result<Expression, EvaluationError> {
        match self {
            Expression::Real(_) => Ok(self.clone()),
            Expression::Complex(_, _) => Ok(self.clone()),

            Expression::Vector(vector) => {
                let mut evaluated_vector = Vec::new();
                for element in vector.iter() {
                    let evaluated_element =
                        element.evaluate_internal(context, scope)?.simplify()?;
                    evaluated_vector.push(evaluated_element);
                }
                Ok(Expression::Vector(evaluated_vector))
            }

            Expression::Matrix(matrix, rows, cols) => {
                let mut evaluated_matrix = Vec::new();
                for element in matrix.iter() {
                    let evaluated_element =
                        element.evaluate_internal(context, scope)?.simplify()?;
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
                let inner = inner.evaluate_internal(context, scope)?.simplify()?;

                match inner {
                    expression if expression.is_concrete() => Ok(expression),
                    expression => Ok(Expression::Paren(Box::new(expression))),
                }
            }

            Expression::Add(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.add(right_eval)
            }
            Expression::Sub(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.sub(right_eval)
            }
            Expression::Mul(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.mul(right_eval)
            }
            Expression::Hadamard(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.hadamard(right_eval)
            }
            Expression::Div(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.div(right_eval)
            }
            Expression::Mod(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.rem(right_eval)
            }
            Expression::Pow(left, right) => {
                let left_eval = left.evaluate_internal(context, scope)?.simplify()?;
                let right_eval = right.evaluate_internal(context, scope)?.simplify()?;
                left_eval.pow(right_eval)
            }
            Expression::Neg(inner) => inner.evaluate_internal(context, scope)?.neg(),
        }
    }
}
