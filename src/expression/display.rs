use {crate::expression::Expression, std::fmt};

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Real(n) => {
                write!(f, "{}", n)?;
            }
            Expression::Complex(real, imag) => {
                if real.abs() < f64::EPSILON && imag.abs() < f64::EPSILON {
                    write!(f, "0")?;
                } else if real.abs() < f64::EPSILON {
                    write!(f, "{}i", imag)?;
                } else if imag.abs() < f64::EPSILON {
                    write!(f, "{}", real)?;
                } else if *imag >= 0.0 {
                    write!(f, "{} + {}i", real, imag)?;
                } else {
                    write!(f, "{} - {}i", real, -imag)?;
                }
            }
            Expression::Vector(data) => {
                write!(f, "[")?;
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")?;
            }
            Expression::Matrix(data, rows, cols) => {
                write!(f, "[")?;
                for r in 0..*rows {
                    if r > 0 {
                        write!(f, "; ")?;
                    }
                    write!(f, "[")?;
                    for c in 0..*cols {
                        if c > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", &data[r * cols + c])?;
                    }
                    write!(f, "]")?;
                }
                write!(f, "]")?;
            }
            Expression::Variable(name) => write!(f, "{}", name)?,
            Expression::FunctionCall(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")?;
            }
            Expression::Paren(inner) => write!(f, "( {} )", inner)?,
            Expression::Neg(operand) => write!(f, "-{}", operand)?,
            Expression::Add(left, right) => write!(f, "{} + {}", left, right)?,
            Expression::Sub(left, right) => {
                write!(f, "{} - ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Mul(left, right) => {
                write!(f, "{} * ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Hadamard(left, right) => {
                write!(f, "{} ** ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Div(left, right) => {
                write!(f, "{} / ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Mod(left, right) => {
                write!(f, "{} % ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
            Expression::Pow(left, right) => {
                write!(f, "{} ^ ", left)?;
                if matches!(right.as_ref(), Expression::Add(..) | Expression::Sub(..)) {
                    write!(f, "( {} )", right)?;
                } else {
                    write!(f, "{}", right)?;
                }
            }
        };

        Ok(())
    }
}
