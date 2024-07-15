#![feature(stmt_expr_attributes)]

use matrix::Matrix;
use network::Activation;
use network::Network;

mod matrix;
mod network;

const TD_SAMPLE: [f64; (2 + 1) * 4] = [
    0.0, 0.0, 0.0, // 0 ^ 0 = 0
    0.0, 1.0, 1.0, // 0 ^ 1 = 1
    1.0, 0.0, 1.0, // 1 ^ 0 = 1
    1.0, 1.0, 0.0, // 1 ^ 1 = 0
];

const LEARN_RATE: f64 = 10.0;

fn main() {
    let mut rng = rand::thread_rng();
    let mut network: Network<f64> = Network::with_activation(&[2, 2, 1], Activation::Sigmoid);
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
            println!("{i} ^ {j} = {}", network.output()[(0, 0)]);
        }
    }
}
