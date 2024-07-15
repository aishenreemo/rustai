#![feature(stmt_expr_attributes)]

use network::Activation;
use network::Network;

mod matrix;
mod network;

fn main() {
    let mut rng = rand::thread_rng();
    let mut network: Network<f64> = Network::with_activation(&[2, 2, 1], Activation::Sigmoid);

    network.randomize(&mut rng);

    println!("{:#?}", network);
}
