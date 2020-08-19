pub mod compute;
pub mod render;

//use rendy;

// pub use self::compute::*;
// pub use self::render::*;

// is this even doable?
pub trait QueryEmitters {
    // impl Iterator<Item = (&Emitter, &Transform, &Mesh, &Texture)>?
    fn query_emitters(&self) -> ();
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
