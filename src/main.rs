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
    // let n_images = idx.dimensions[0];
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
    let training_images = load_images("mnist/train-images.idx3-ubyte");
    let training_labels = load_labels("mnist/train-labels.idx1-ubyte");
    let test_images = load_images("mnist/t10k-images.idx3-ubyte");
    let test_labels = load_labels("mnist/t10k-labels.idx1-ubyte");

    let input_size = 28 * 28;
    let output_size = 10;

    let mut model = NeuralNet::new(
        vec![input_size, 50, output_size],
        activations::Initialization::Kaiming,
        activations::LEAKY_RELU,
        activations::SOFTMAX_CROSSENTROPY,
    );

    let batch_size = 40;
    let learning_rate = 0.01;

    model.sgd(
        batch_size,
        learning_rate,
        &training_images,
        &training_labels,
        false,
        &test_images,
        &test_labels,
    );
}
