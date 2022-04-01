use super::matrix2d::Matrix2D;
use super::matrix2d::Shape;

pub struct Activation<'a> {
    activation: &'a dyn Fn(&Matrix2D) -> Matrix2D,
    derivative: &'a dyn Fn(&Matrix2D) -> Matrix2D,
}

impl<'a> Activation<'a> {
    pub fn activation(&self, x: &Matrix2D) -> Matrix2D {
        (self.activation)(x)
    }

    pub fn derivative(&self, x: &Matrix2D) -> Matrix2D {
        (self.derivative)(x)
    }
}

pub enum Loss<'a> {
    Plain {
        loss: &'a dyn Fn(&Matrix2D, &Matrix2D) -> f32,
        derivative: &'a dyn Fn(&Matrix2D, &Matrix2D) -> Matrix2D,
    },
    Combined {
        activation: &'a dyn Fn(&Matrix2D) -> Matrix2D,
        loss: &'a dyn Fn(&Matrix2D, &Matrix2D) -> f32,
        combined_derivative: &'a dyn Fn(&Matrix2D, &Matrix2D) -> Matrix2D,
    },
}

impl<'a> Loss<'a> {
    pub fn loss(&self, prediction: &Matrix2D, label: &Matrix2D) -> f32 {
        match self {
            Self::Plain {
                loss,
                derivative: _,
            } => loss(prediction, label),
            Self::Combined {
                activation: _,
                loss,
                combined_derivative: _,
            } => loss(prediction, label),
        }
    }

    pub fn derivative(&self, prediction: &Matrix2D, label: &Matrix2D) -> Matrix2D {
        match self {
            Self::Plain {
                loss: _,
                derivative,
            } => derivative(prediction, label),
            Self::Combined {
                activation: _,
                loss: _,
                combined_derivative,
            } => combined_derivative(prediction, label),
        }
    }

    pub fn activation(&self, x: &Matrix2D) -> Matrix2D {
        match self {
            Self::Plain {
                loss: _,
                derivative: _,
            } => panic!("Loss has no activation"),
            Self::Combined {
                activation,
                loss: _,
                combined_derivative: _,
            } => activation(x),
        }
    }
}

pub enum Initialization {
    Uniform,
    Normal,
    Xavier,
    Kaiming,
}

impl Initialization {
    pub fn initialize(&self, shape: Shape, add_bias: bool) -> Matrix2D {
        let without_bias = match self {
           Initialization::Uniform => Matrix2D::random_uniform(shape, -1.0, 1.0),
           Initialization::Normal => Matrix2D::random_std_normal(shape),
           Initialization::Xavier => Matrix2D::random_uniform(shape, -1.0, 1.0) * (6f32 / (shape.0 + shape.1) as f32).sqrt(),
           Initialization::Kaiming => Matrix2D::random_std_normal(shape) * (2f32 / shape.1 as f32).sqrt(),
        };
        if add_bias {
            Matrix2D::hstack(&without_bias, &Matrix2D::zeros((shape.0, 1)))
        } else {
            without_bias
        }
    }
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid(x: &Matrix2D) -> Matrix2D {
    x.map(sigmoid_scalar)
}

fn sigmoid_derivative(x: &Matrix2D) -> Matrix2D {
    x.map(|x_i| sigmoid_scalar(x_i) * (1.0 - sigmoid_scalar(x_i)))
}

fn ReLU(x: &Matrix2D) -> Matrix2D {
    x.filter(|x_i| x_i > 0.0)
}

fn ReLU_derivative(x: &Matrix2D) -> Matrix2D {
    x.map(|x_i| (x_i > 0.0) as u16 as f32)
}

fn leaky_ReLU(x: &Matrix2D) -> Matrix2D {
    x.map(|x_i| if x_i > 0.0 { x_i } else { 0.1 * x_i })
}

fn leaky_ReLU_derivative(x: &Matrix2D) -> Matrix2D {
    x.map(|x_i| if x_i > 0.0 { 1.0 } else { 0.1 })
}

fn softmax(x: &Matrix2D) -> Matrix2D {
    let maximum = x.max();
    let e = x.map(|x_i| (x_i - maximum).exp());

    let sum = e.fold(0.0, |s, x_i| s + x_i);
    e / sum
}

fn softmax_derivative(x: &Matrix2D) -> Matrix2D {
    let softmax = softmax(x);
    softmax.map(|x_i| x_i * (1.0 - x_i))
}

fn mse_loss(x: &Matrix2D, y: &Matrix2D) -> f32 {
    ((x - y).transpose() * (x - y))[(0, 0)]
}

fn mse_derivative(x: &Matrix2D, y: &Matrix2D) -> Matrix2D {
    2.0 * (x - y)
}

fn crossentrop_loss(x: &Matrix2D, y: &Matrix2D) -> f32 {
    -(x[y.argmax()] + f32::EPSILON).ln()
}

fn crossentropy_softmax_derivative(x: &Matrix2D, y: &Matrix2D) -> Matrix2D {
    x - y
}

fn binary_crossentropy_loss(x: &Matrix2D, y: &Matrix2D) -> f32 {
    let a = x.map(|x_i| (x_i + f32::EPSILON).ln());
    let b = x.map(|x_i| (1.0 - x_i + f32::EPSILON).ln());
    let mut s = 0.0;
    for y_i in 0..y.size() {
        s += if y[(y_i, 0)] == 1.0 {
            a[(y_i, 0)]
        } else {
            b[(y_i, 0)]
        }
    }
    -1.0 / x.size() as f32 * s
}

fn binary_crossentropy_sigmoid_derivative(x: &Matrix2D, y: &Matrix2D) -> Matrix2D {
    sigmoid(x) - y
}

pub const SIGMOID: Activation = Activation {
    activation: &sigmoid,
    derivative: &sigmoid_derivative,
};
pub const RELU: Activation = Activation {
    activation: &ReLU,
    derivative: &ReLU_derivative,
};
pub const LEAKY_RELU: Activation = Activation {
    activation: &leaky_ReLU,
    derivative: &leaky_ReLU_derivative,
};
pub const SOFTMAX: Activation = Activation {
    activation: &softmax,
    derivative: &softmax_derivative,
};

pub const SOFTMAX_CROSSENTROPY: Loss = Loss::Combined {
    activation: &softmax,
    loss: &crossentrop_loss,
    combined_derivative: &crossentropy_softmax_derivative,
};
pub const MSE: Loss = Loss::Plain {
    loss: &mse_loss,
    derivative: &mse_derivative,
};
pub const SIGMOID_BINARY_CE: Loss = Loss::Combined {
    activation: &sigmoid,
    loss: &binary_crossentropy_loss,
    combined_derivative: &binary_crossentropy_sigmoid_derivative,
};
