use activations::{Activation, Initialization, Loss};
use matrix2d::Matrix2D;
use std::{cmp::min, io::Write};

pub mod activations;
pub mod matrix2d;

pub struct NeuralNet<'a> {
    activation: Activation<'a>,
    loss: Loss<'a>,
    weights: Vec<Matrix2D>,
}

impl<'a> NeuralNet<'a> {
    pub fn new(
        layers: Vec<i32>,
        init: Initialization,
        activation: Activation<'a>,
        cost: Loss<'a>,
    ) -> Self {
        let mut weights = Vec::new();

        for i in 0..layers.len() - 1 {
            let shape = (layers[i + 1] as usize, layers[i] as usize);
            let w = init.initialize(shape, true);
            weights.push(w);
        }

        Self {
            activation,
            loss: cost,
            weights,
        }
    }

    fn forward_pass(&self, input: &Matrix2D) -> (Vec<Matrix2D>, Vec<Matrix2D>) {
        let mut inputs = vec![input.clone()];
        let mut activations = vec![Matrix2D::vstack(
            &self.activation.activation(input),
            &Matrix2D::ones((1, 1)), // add bias to activation
        )];
        let mut current = activations[0].clone();

        for (i, w) in self.weights.iter().enumerate() {
            let layer_input = w * &current;

            let layer_activation = if i == self.weights.len() - 1 {
                match self.loss {
                    Loss::Plain {
                        loss: _,
                        derivative: _,
                    } => self.activation.activation(&layer_input),
                    Loss::Combined {
                        activation,
                        loss: _,
                        combined_derivative: _,
                    } => activation(&layer_input),
                }
            } else {
                Matrix2D::vstack(
                    &self.activation.activation(&layer_input),
                    &Matrix2D::ones((1, 1)),
                ) //add bias to activation
            };

            if layer_activation.get_vec().iter().any(|x| !x.is_finite()) {
                panic!(
                    "Exploding Gradient ? {:?} \n {:?}-->\n{:?}",
                    &current, layer_input, layer_activation
                );
            }

            current = layer_activation.clone();
            inputs.push(layer_input);
            activations.push(layer_activation);
        }

        (inputs, activations)
    }

    fn backward_pass(&self, input: &Matrix2D, label: &Matrix2D) -> Vec<Matrix2D> {
        fn rowwise_mul_mut(matrix: &mut Matrix2D, column_vector: &Matrix2D) {
            assert_eq!(
                matrix.rows(),
                column_vector.rows(),
                "Cannot rowwise multiply matrice with shapes {:?} and {:?}",
                matrix.shape(),
                column_vector.shape()
            );
            for r in 0..matrix.rows() {
                let factor = column_vector[(r, 0)];
                unsafe {
                    for x in matrix.get_row_slice(r).iter_mut() {
                        *x *= factor;
                    }
                }
            }
        }

        let (inputs, activations) = self.forward_pass(input);

        let last_layer_activation = activations.last().unwrap();
        let last_layer_input = inputs.last().unwrap();

        let mut current = match self.loss {
            Loss::Plain {
                loss: _,
                derivative,
            } => {
                let mut tmp = derivative(&last_layer_activation, label);
                rowwise_mul_mut(&mut tmp, &self.activation.derivative(last_layer_input));
                tmp
            }
            Loss::Combined {
                activation: _,
                loss: _,
                combined_derivative,
            } => combined_derivative(last_layer_activation, label),
        };

        let mut gradient = vec![];

        for i in (0..self.weights.len()).rev() {
            let layer_weights = &self.weights[i];
            let layer_input = &inputs[i];
            let layer_activation = &activations[i];

            let g_w = layer_activation * &current.transpose();

            // // Gradient Clipping
            // let clipping_threshold = layer_weights.size() as f32 * 1000.0;
            // let g_w_mag = g_w.fold(0.0, |s, x| s + x * x).sqrt();
            // if g_w_mag > clipping_threshold {
            //     g_w *= clipping_threshold / g_w_mag;
            // }

            gradient.push(g_w.transpose());
            let mut tmp = layer_weights.transpose();
            tmp = tmp.vslice(tmp.rows() - 1).0; // remove bias weights
            rowwise_mul_mut(&mut tmp, &self.activation.derivative(layer_input));
            current = tmp * &current;
        }
        gradient.reverse();
        gradient
    }

    fn backward_pass_batch(&self, data: &[Matrix2D], labels: &[Matrix2D]) -> Vec<Matrix2D> {
        let mut gradient_acc = vec![];

        for w in &self.weights {
            gradient_acc.push(Matrix2D::zeros_like(w));
        }

        for (x, y) in data.iter().zip(labels) {
            let gradient = self.backward_pass(x, y);
            for j in 0..gradient_acc.len() {
                gradient_acc[j] += &gradient[j];
            }
        }

        for g in &mut gradient_acc {
            *g /= data.len() as f32;
        }

        gradient_acc
    }

    pub fn predict(&self, input: &Matrix2D) -> Matrix2D {
        self.forward_pass(input).1.last().unwrap().clone()
    }

    pub fn evaluate(&self, test_data: &[Matrix2D], test_labels: &[Matrix2D]) -> (f32, f32) {
        let mut total_cost = 0.0;
        let mut total_accuracy = 0;

        for (x, y) in test_data.iter().zip(test_labels) {
            let prediction = self.predict(x);
            total_cost += self.loss.loss(&prediction, y);
            total_accuracy += (prediction.argmax() == y.argmax()) as usize;
        }

        (
            total_cost / test_data.len() as f32,
            total_accuracy as f32 / test_data.len() as f32,
        )
    }

    pub fn sgd(
        &mut self,
        batch_size: usize,
        learning_rate: f32,
        data: &[Matrix2D],
        labels: &[Matrix2D],
        silent: bool,
        test_data: &[Matrix2D],
        test_labels: &[Matrix2D],
    ) {
        let n_batches = data.len() / batch_size;
        if !silent {
            let (cost, accuracy) = self.evaluate(&test_data, &test_labels);
            println!("\n(0/{}) Cost: {:.4},    Accuracy: {:.2}%", n_batches, cost, 100.0 * accuracy);
        }

        for left in (0..data.len()).step_by(batch_size) {
            if !silent {
                print!("-");
                std::io::stdout().flush().unwrap();
            }

            let right = min(left + batch_size, data.len());

            let batch_data = &data[left..right];
            let batch_labels = &labels[left..right];

            let gradient = self.backward_pass_batch(batch_data, batch_labels);

            for (w, g) in self.weights.iter_mut().zip(gradient) {
                *w -= learning_rate * g;
            }

            if !silent {
                let batch = left / batch_size;
                let n_printed_updates = 10;
                if batch % (n_batches / n_printed_updates) == (n_batches / n_printed_updates) - 1 {
                    let (cost, accuracy) = self.evaluate(&test_data, &test_labels);
                    println!(
                        "\n({}/{}) Cost: {:.4},    Accuracy: {:.2}%",
                        batch + 1,
                        n_batches,
                        cost,
                        100.0 * accuracy
                    );
                }
            }
        }
    }

    pub fn momentum(
        &mut self,
        batch_size: usize,
        learning_rate: f32,
        data: &[Matrix2D],
        labels: &[Matrix2D],
    ) {
        todo!()
    }

    pub fn rms_prop(
        &mut self,
        batch_size: usize,
        learning_rate: f32,
        data: &[Matrix2D],
        labels: &[Matrix2D],
    ) {
        todo!()
    }

    pub fn adam(
        &mut self,
        batch_size: usize,
        learning_rate: f32,
        data: &[Matrix2D],
        labels: &[Matrix2D],
    ) {
        todo!()
    }
}
