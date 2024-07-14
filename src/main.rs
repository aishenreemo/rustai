use anyhow::Ok;
use anyhow::Result;
use network::Network;

use crate::activation::Activation;

mod activation;
mod matrix;
mod network;

fn main() -> Result<()> {
    let network: Network<f64> = Network::with_activation(&[2, 2, 1], Activation::Sigmoid)?;

    println!("{:?}", network);
    Ok(())
}
