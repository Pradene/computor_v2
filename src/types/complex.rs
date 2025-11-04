use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.real + rhs.real, self.imag + rhs.imag)
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.real - rhs.real, self.imag - rhs.imag)
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.real * rhs.real - self.imag * rhs.imag,
            self.real * rhs.imag + self.imag * rhs.real,
        )
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            return Self::new(f64::INFINITY, f64::INFINITY);
        }

        let denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
        let real = (self.real * rhs.real + self.imag * rhs.imag) / denominator;
        let imag = (self.imag * rhs.real - self.real * rhs.imag) / denominator;

        Self::new(real, imag)
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.real, -self.imag)
    }
}

impl Complex {
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub fn one() -> Self {
        Self::new(1.0, 0.0)
    }

    pub fn is_real(&self) -> bool {
        self.imag.abs() < f64::EPSILON
    }

    pub fn is_zero(&self) -> bool {
        self.real.abs() < f64::EPSILON && self.imag.abs() < f64::EPSILON
    }

    pub fn is_one(&self) -> bool {
        (self.real - 1.0).abs() < f64::EPSILON && self.imag.abs() < f64::EPSILON
    }

    pub fn magnitude(&self) -> f64 {
        self.magnitude_squared().sqrt()
    }

    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }

    pub fn pow(&self, exponent: Complex) -> Self {
        if self.is_zero() {
            if exponent.real > 0.0 || (!exponent.is_real() && exponent.real == 0.0) {
                return Complex::zero();
            } else if exponent.real < 0.0 {
                return Complex::new(f64::INFINITY, f64::INFINITY);
            } else {
                return Complex::one();
            }
        }

        if exponent.is_zero() {
            return Complex::one();
        }

        if exponent.is_one() {
            return self.clone();
        }

        let ln_z = self.ln();
        let w_ln_z = exponent.clone() * ln_z;
        w_ln_z.exp()
    }

    pub fn ln(&self) -> Self {
        if self.is_zero() {
            return Complex::new(f64::NEG_INFINITY, 0.0);
        }

        let magnitude = self.magnitude();
        let phase = self.phase();
        Self::new(magnitude.ln(), phase)
    }

    pub fn exp(&self) -> Self {
        let exp_real = self.real.exp();
        Self::new(exp_real * self.imag.cos(), exp_real * self.imag.sin())
    }

    pub fn sqrt(&self) -> Self {
        if self.is_zero() {
            return Complex::zero();
        }

        let r = self.magnitude();
        let theta = self.phase();

        let sqrt_r = r.sqrt();
        let half_theta = theta / 2.0;

        Self::new(sqrt_r * half_theta.cos(), sqrt_r * half_theta.sin())
    }

    pub fn is_finite(&self) -> bool {
        self.real.is_finite() && self.imag.is_finite()
    }

    pub fn is_infinite(&self) -> bool {
        self.real.is_infinite() || self.imag.is_infinite()
    }

    pub fn is_nan(&self) -> bool {
        self.real.is_nan() || self.imag.is_nan()
    }

    pub fn abs(&self) -> f64 {
        (self.real.powf(2.0) + self.imag.powf(2.0)).sqrt()
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_real() {
            if self.real.fract() == 0.0 {
                write!(f, "{}", self.real as i64)?;
            } else {
                write!(f, "{}", self.real)?;
            }
        } else if self.real == 0.0 {
            if self.imag == 1.0 {
                write!(f, "i")?;
            } else if self.imag == -1.0 {
                write!(f, "-i")?;
            } else if self.imag.fract() == 0.0 {
                write!(f, "{}i", self.imag as i64)?;
            } else {
                write!(f, "{}i", self.imag)?;
            }
        } else {
            if self.real.fract() == 0.0 {
                write!(f, "{}", self.real as i64)?;
            } else {
                write!(f, "{}", self.real)?;
            }

            if self.imag > 0.0 {
                write!(f, " + ")?;
                if self.imag == 1.0 {
                    write!(f, "i")?;
                } else if self.imag.fract() == 0.0 {
                    write!(f, "{}i", self.imag as i64)?;
                } else {
                    write!(f, "{}i", self.imag)?;
                }
            } else {
                write!(f, " - ")?;
                let abs_imag = self.imag.abs();
                if abs_imag == 1.0 {
                    write!(f, "i")?
                } else if abs_imag.fract() == 0.0 {
                    write!(f, "{}i", abs_imag as i64)?;
                } else {
                    write!(f, "{}i", abs_imag)?;
                }
            }
        }

        Ok(())
    }
}
