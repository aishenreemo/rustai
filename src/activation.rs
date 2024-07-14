#[derive(Default, Debug)]
pub enum Activation<T> {
    #[default]
    Identity,
    Sigmoid,
    Custom {
        f: fn(T) -> T,
        d: fn(T) -> T,
    },
}
