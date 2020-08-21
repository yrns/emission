pub use compute::*;
pub use render::*;

pub mod compute;
pub mod render;

pub trait QueryEmitters {
    fn query_emitters(&self) -> &Vec<Emitter>;
}

pub trait QueryProjView {
    fn query_proj_view(&self) -> glam::Mat4;
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
