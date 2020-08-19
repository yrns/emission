use emission::{compute::*, render::*};
use rendy::{
    command::Families,
    factory::{Config, Factory},
    graph::{
        present::PresentNode,
        render::{RenderGroupBuilder, SimpleGraphicsPipeline},
        Graph, GraphBuilder, NodeDesc,
    },
    hal,
    wsi::winit::{EventsLoop, Window, WindowBuilder},
};

// #[cfg(feature = "dx12")]
// type Backend = rendy::dx12::Backend;

// #[cfg(feature = "metal")]
// type Backend = rendy::metal::Backend;

// #[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    window: &Window,
) -> Result<(), failure::Error> {
    let mut graph = build_graph(factory, families, window.clone());

    let started = std::time::Instant::now();

    let mut last_window_size = window.get_inner_size();
    let mut need_rebuild = false;

    let mut frames = 0u64..;
    let mut elapsed = started.elapsed();

    for _ in &mut frames {
        factory.maintain(families);
        event_loop.poll_events(|_| ());
        let new_window_size = window.get_inner_size();

        if last_window_size != new_window_size {
            need_rebuild = true;
        }

        if need_rebuild && last_window_size == new_window_size {
            need_rebuild = false;
            let started = std::time::Instant::now();
            graph.dispose(factory, &());
            println!("Graph disposed in: {:?}", started.elapsed());
            graph = build_graph(factory, families, window.clone());
        }

        last_window_size = new_window_size;

        graph.run(factory, families, &());

        elapsed = started.elapsed();
        if elapsed >= std::time::Duration::new(5, 0) {
            break;
        }
    }

    let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

    log::info!(
        "Elapsed: {:?}. Frames: {}. FPS: {}",
        elapsed,
        frames.start,
        frames.start * 1_000_000_000 / elapsed_ns
    );

    graph.dispose(factory, &mut ());
    Ok(())
}

//#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("quads", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    run(&mut event_loop, &mut factory, &mut families, &window).unwrap();
    log::debug!("Done");

    log::debug!("Drop families");
    drop(families);

    log::debug!("Drop factory");
    drop(factory);
}

//#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn build_graph(
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    window: &Window,
) -> Graph<Backend, ()> {
    let surface = factory.create_surface(window);

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    // let emitters =
    //     graph_builder.create_buffer(MAX_EMITTERS as u64 * std::mem::size_of::<Emitter>() as u64);
    // let emitter_state = graph_builder
    //     .create_buffer(MAX_EMITTERS as u64 * std::mem::size_of::<EmitterState>() as u64);
    let particles =
        graph_builder.create_buffer(MAX_PARTICLES as u64 * std::mem::size_of::<Particle>() as u64);
    let indirect = graph_builder.create_buffer(
        MAX_PARTICLES as u64 * std::mem::size_of::<rendy::command::DrawCommand>() as u64,
    );

    // TODO: particle count buffer

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());
    let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);

    let color = graph_builder.create_image(
        window_kind,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([0.9, 0.9, 0.9, 1.0].into())),
    );

    // let depth = graph_builder.create_image(
    //     window_kind,
    //     1,
    //     hal::format::Format::D16Unorm,
    //     Some(hal::command::ClearValue::DepthStencil(
    //         hal::command::ClearDepthStencil(1.0, 0),
    //     )),
    // );

    let compute = graph_builder.add_node(
        ComputeNodeDesc
            .builder()
            // .with_buffer(emitters)
            // .with_buffer(emitter_state)
            .with_buffer(particles)
            .with_buffer(indirect),
    );

    let pass = graph_builder.add_node(
        RenderNode::builder()
            .with_buffer(particles)
            .with_buffer(indirect)
            .with_dependency(compute)
            .into_subpass()
            .with_color(color)
            //.with_depth_stencil(depth)
            .into_pass(),
    );

    graph_builder.add_node(PresentNode::builder(&factory, surface, color).with_dependency(pass));

    let started = std::time::Instant::now();
    let graph = graph_builder.build(factory, families, &()).unwrap();
    println!("Graph built in: {:?}", started.elapsed());
    graph
}
