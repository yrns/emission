pub use compute::*;
pub use render::*;

pub mod compute;
pub mod render;

use nalgebra::{Matrix4, Vector3, Vector4};

pub type Mat4 = Matrix4<f32>;
pub type Vec4 = Vector4<f32>;
pub type Vec3 = Vector3<f32>;

pub trait QueryEmitters {
    fn query_emitters(&self) -> &Vec<Emitter>;
}

pub trait QueryProjView {
    fn query_proj_view(&self) -> (Mat4, Mat4);
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
