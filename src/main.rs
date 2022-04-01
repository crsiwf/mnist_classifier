use neural_net::{activations, matrix2d::Matrix2D, NeuralNet};
mod load_idx;

fn load_labels(file_name: &str) -> Vec<Matrix2D> {
    let idx = load_idx::load_idx::<u8>(file_name).unwrap();

    assert_eq!(1, idx.dimensions.len());

    let mut labels = vec![];
    for label in idx.array {
        let mut label_vector = Matrix2D::zeros((10, 1));
        label_vector[(label as usize, 0)] = 1.0;
        labels.push(label_vector);
    }
    labels
}

fn load_images(file_name: &str) -> Vec<Matrix2D> {
    fn normalise_pixels(v: Vec<u8>) -> Vec<f32> {
        v.into_iter().map(|x| x as f32 / 128.0 - 1.0).collect()
    }

    let idx = load_idx::load_idx::<u8>(file_name).unwrap();

    assert_eq!(3, idx.dimensions.len());
    let n_images = idx.dimensions[0];
    let image_height = idx.dimensions[1];
    let image_width = idx.dimensions[2];
    let image_size = image_height * image_width;

    let shape = (image_size, 1);

    let values = normalise_pixels(idx.array);

    let mut images = vec![];

    for i in (0..values.len()).step_by(image_size) {
        images.push(Matrix2D::from_vec(
            shape,
            values[i..i + image_size].to_vec(),
        ))
    }

    images
}

fn main() {
    let training_images = load_images("D:/Daten/Rust/nn/mnist/train-images.idx3-ubyte");
    let training_labels = load_labels("D:/Daten/Rust/nn/mnist/train-labels.idx1-ubyte");
    let test_images = load_images("D:/Daten/Rust/nn/mnist/t10k-images.idx3-ubyte");
    let test_labels = load_labels("D:/Daten/Rust/nn/mnist/t10k-labels.idx1-ubyte");

    // let (training_images, training_labels): (Vec<Matrix2D>, Vec<Matrix2D>) = training_images
    //     .into_iter()
    //     .zip(training_labels.into_iter())
    //     .filter(|(_, l)| l.argmax().0 == 3)
    //     .unzip();
    // let (test_images, test_labels): (Vec<Matrix2D>, Vec<Matrix2D>) = test_images
    //     .into_iter()
    //     .zip(test_labels.into_iter())
    //     .filter(|(_, l)| l.argmax().0 == 3)
    //     .unzip();

    let input_size = 28 * 28;
    let output_size = 10;

    let mut model = NeuralNet::new(
        vec![input_size, 300, 100, output_size],
        activations::Initialization::Kaiming,
        activations::LEAKY_RELU,
        activations::SOFTMAX_CROSSENTROPY,
    );

    let batch_size = 40;
    let learning_rate = 0.01;

    let (cost, correct) = model.evaluate(&test_images, &test_labels);
    println!("\nCost: {}\nAccuracy: {}%", cost, 100.0 * correct);

    for (ti, tl) in test_images.iter().zip(&test_labels).take(20) {
        let prediction = model.predict(&ti);
        let p_argmax = prediction.argmax();
        let t_argmax = tl.argmax();
        println!("{} {}", t_argmax.0, p_argmax.0);
    }

    model.sgd(
        batch_size,
        learning_rate,
        &training_images,
        &training_labels,
        &test_images,
        &test_labels,
    );

    let (cost, correct) = model.evaluate(&test_images, &test_labels);
    println!("\nCost: {}\nAccuracy: {}%", cost, 100.0 * correct);

    for (ti, tl) in test_images.iter().zip(test_labels).take(20) {
        let prediction = model.predict(&ti);
        let p_argmax = prediction.argmax();
        let t_argmax = tl.argmax();
        println!("{} {}", t_argmax.0, p_argmax.0);
    }
}
