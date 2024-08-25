use std::ops::Index;

use rustai::matrix::Matrix;
use rustai::network::ActivationVariant;
use rustai::network::Network;

#[test]
fn test_xor() {
    const TD_SAMPLE: [f64; (2 + 1) * 4] = [
        0.0, 0.0, 0.0, // 0 ^ 0 = 0
        0.0, 1.0, 1.0, // 0 ^ 1 = 1
        1.0, 0.0, 1.0, // 1 ^ 0 = 1
        1.0, 1.0, 0.0, // 1 ^ 1 = 0
    ];

    const LEARN_RATE: f64 = 10.0;

    let mut rng = rand::thread_rng();
    let mut network = Network::<f64, ActivationVariant>::with_activation(&[2, 2, 1], ActivationVariant::Sigmoid);
    let tdata = Matrix::with_items::<usize, [f64; 12]>(TD_SAMPLE, 2 + 1, 4);

    network.randomize(&mut rng);

    for _ in 0..50_000 {
        network.backpropagate(&tdata);
        network.learn(LEARN_RATE);

        // let cost = network.cost(&tdata);
        // println!("{cost}");
    }

    for i in 0..2 {
        for j in 0..2 {
            network.forward(&[i as f64, j as f64]);
            let output = network.output();
            let output = output.index((0, 0));

            println!("{i} ^ {j} = {}", output);
            assert_eq!(output.round(), TD_SAMPLE[(i * 2 + j) * 3 + 2]);
        }
    }
}
