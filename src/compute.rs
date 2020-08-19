use once_cell::sync::Lazy;
use rendy::{
    command::{
        CommandBuffer, CommandPool, Compute, DrawCommand, ExecutableState, Family, MultiShot,
        PendingState, QueueId, SimultaneousUse, Submit,
    },
    factory::{BufferState, Factory},
    frame::Frames,
    graph::{
        gfx_acquire_barriers, gfx_release_barriers, BufferAccess, GraphContext, Node, NodeBuffer,
        NodeDesc, NodeImage, NodeSubmittable,
    },
    hal::{self, Device as _},
    memory::Dynamic,
    resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
    shader::{Shader, ShaderKind, SourceLanguage, SourceShaderInfo, SpirvReflection, SpirvShader},
};

static COMPUTE: Lazy<SpirvShader> = Lazy::new(|| {
    SourceShaderInfo::new(
        include_str!("../src/shaders/simulate.comp"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/simulate.comp").into(),
        ShaderKind::Compute,
        SourceLanguage::GLSL,
        "main",
    )
    .precompile()
    .unwrap()
});

// this is only used for reflection?
static SHADERS: Lazy<rendy::shader::ShaderSetBuilder> = Lazy::new(|| {
    rendy::shader::ShaderSetBuilder::default()
        .with_compute(&*COMPUTE)
        .unwrap()
});

static SHADER_REFLECTION: Lazy<SpirvReflection> = Lazy::new(|| SHADERS.reflect().unwrap());

// make these dynamic, see ensure_buffer in amethyst
pub const MAX_EMITTERS: u32 = 2;
pub const MAX_PARTICLES: u32 = MAX_EMITTERS * 32;

// #[allow(non_camel_case_types)]
// type vec3 = [f32; 3];

#[allow(non_camel_case_types)]
type vec4 = [f32; 4];

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Emitter {
    pub position: vec4,
    gen: u32,
    spawn_rate: f32,
    lifetime: f32,
    max_particles: u32,
    spawn_offset_min: vec4,
    spawn_offset_max: vec4,
    accel: vec4,
    scale: vec4,
    color: vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct EmitterState {
    gen: u32,
    num: u32,
    new: u32,
    last_spawn: f32,
    t: f32,
    reset: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct Particle {
    color: vec4,
    position: vec4,
    velocity: vec4,
    lifetime: f32,
    emitter: u32,
    gen: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Stats {
    particles: u32,
    //debug1: u32,
}

#[derive(Debug)]
pub struct ComputeNode<B: hal::Backend> {
    emitters: Escape<Buffer<B>>,
    emitter_state: Escape<Buffer<B>>,
    stats: Escape<Buffer<B>>,
    set_layout: Handle<DescriptorSetLayout<B>>,
    pipeline_layout: B::PipelineLayout,
    pipeline: B::ComputePipeline,
    descriptor_set: Escape<DescriptorSet<B>>,
    command_pool: CommandPool<B, Compute>,
    command_buffer:
        CommandBuffer<B, Compute, PendingState<ExecutableState<MultiShot<SimultaneousUse>>>>,
    submit: Submit<B, SimultaneousUse>,
}

impl<'a, B> NodeSubmittable<'a, B> for ComputeNode<B>
where
    B: hal::Backend,
{
    type Submittable = &'a Submit<B, SimultaneousUse>;
    type Submittables = &'a [Submit<B, SimultaneousUse>];
}

impl<B, T> Node<B, T> for ComputeNode<B>
where
    B: hal::Backend,
    T: ?Sized,
{
    type Capability = Compute;
    type Desc = ComputeNodeDesc;

    fn run<'a>(
        &'a mut self,
        _ctx: &GraphContext<B>,
        factory: &Factory<B>,
        _aux: &T,
        _frames: &'a Frames<B>,
    ) -> &'a [Submit<B, SimultaneousUse>] {
        let range = 0..(std::mem::size_of::<Stats>() as u64);
        let mut mapped = self.stats.map(factory.device(), range.clone()).unwrap();
        unsafe {
            let read: &[Stats] = mapped.read(factory.device(), range).unwrap();
            dbg!(read[0]);
        }

        std::slice::from_ref(&self.submit)
    }

    unsafe fn dispose(mut self, factory: &mut Factory<B>, _aux: &T) {
        drop(self.submit);
        self.command_pool
            .free_buffers(Some(self.command_buffer.mark_complete()));
        factory.destroy_command_pool(self.command_pool);
        factory.destroy_compute_pipeline(self.pipeline);
        factory.destroy_pipeline_layout(self.pipeline_layout);
    }
}

pub fn initialize_buffers<B: hal::Backend>(
    factory: &mut Factory<B>,
    queue: QueueId,
    emitters: &Escape<Buffer<B>>,
    emitter_state: &Escape<Buffer<B>>,
    particles: &Handle<Buffer<B>>,
    indirect: &Handle<Buffer<B>>,
) -> Result<(), failure::Error> {
    let e1 = Emitter {
        position: [0.0, 0.0, 0.0, 0.0],
        gen: 1,
        spawn_rate: 50.0, // (/ 1.0 1000.0)
        lifetime: 0.5,
        max_particles: 100,
        spawn_offset_min: [-0.5, 0.0, 0.0, 0.0],
        spawn_offset_max: [0.5, 0.0, 0.0, 0.0],
        accel: [0.0, -4.0, 0.0, 0.0],
        scale: [1.0, 1.0, 1.0, 0.0],
        color: [0.0, 0.0, 1.0, 1.0],
        ..Default::default()
    };

    let emitter_data = vec![e1];

    unsafe {
        factory.upload_buffer(
            emitters,
            0,
            &emitter_data,
            None,
            BufferState {
                queue,
                stage: hal::pso::PipelineStage::COMPUTE_SHADER,
                access: hal::buffer::Access::SHADER_READ,
            },
        )
    }?;

    unsafe {
        factory.upload_buffer(
            emitter_state,
            0,
            &(0..MAX_EMITTERS)
                .map(|_| EmitterState::default())
                .collect::<Vec<_>>(),
            None,
            BufferState {
                queue,
                stage: hal::pso::PipelineStage::COMPUTE_SHADER,
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
            },
        )
    }?;

    unsafe {
        factory.upload_buffer(
            particles,
            0,
            &(0..MAX_PARTICLES)
                .map(|_| Particle {
                    //color: [0.5, 1.0, 0.5, 1.0],
                    ..Default::default()
                })
                .collect::<Vec<_>>(),
            None,
            BufferState {
                queue,
                //stage: hal::pso::PipelineStage::COMPUTE_SHADER,
                stage: hal::pso::PipelineStage::VERTEX_SHADER,
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
            },
        )
    }?;

    unsafe {
        factory.upload_buffer(
            indirect,
            0,
            &(0..MAX_PARTICLES)
                .map(|i| DrawCommand {
                    vertex_count: 6,
                    instance_count: 0, // this gets set to one in the shader
                    first_vertex: 0,
                    first_instance: i,
                })
                .collect::<Vec<_>>(),
            None,
            BufferState {
                queue,
                //stage: hal::pso::PipelineStage::COMPUTE_SHADER,
                stage: hal::pso::PipelineStage::VERTEX_SHADER,
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
            },
        )
    }?;

    Ok(())
}

#[derive(Debug, Default)]
pub struct ComputeNodeDesc;

impl<B, T> NodeDesc<B, T> for ComputeNodeDesc
where
    B: hal::Backend,
    T: ?Sized,
{
    type Node = ComputeNode<B>;

    fn buffers(&self) -> Vec<BufferAccess> {
        vec![
            // BufferAccess {
            //     // This needs write access to appease rendy chain somehow.
            //     access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
            //     stages: hal::pso::PipelineStage::COMPUTE_SHADER,
            //     usage: hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
            // },
            // BufferAccess {
            //     access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
            //     stages: hal::pso::PipelineStage::COMPUTE_SHADER,
            //     usage: hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
            // },

            // instance/particles
            BufferAccess {
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
                stages: hal::pso::PipelineStage::COMPUTE_SHADER,
                usage: hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
            },
            // indirect
            BufferAccess {
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
                stages: hal::pso::PipelineStage::COMPUTE_SHADER,
                usage: hal::buffer::Usage::STORAGE
                    | hal::buffer::Usage::TRANSFER_DST
                    | hal::buffer::Usage::INDIRECT,
            },
        ]
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        family: &mut Family<B>,
        queue: usize,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
    ) -> Result<Self::Node, failure::Error> {
        assert!(images.is_empty());
        assert_eq!(buffers.len(), 2);

        // let emitters = ctx.get_buffer(buffers[0].id).unwrap();
        // let emitter_state = ctx.get_buffer(buffers[1].id).unwrap();

        let particles = ctx.get_buffer(buffers[0].id).unwrap();
        let indirect = ctx.get_buffer(buffers[1].id).unwrap();

        let queue = QueueId {
            index: queue,
            family: family.id(),
        };

        let emitters = factory.create_buffer(
            BufferInfo {
                size: std::mem::size_of::<Emitter>() as u64 * MAX_EMITTERS as u64,
                usage: hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
            },
            Dynamic,
        )?;

        let emitter_state = factory.create_buffer(
            BufferInfo {
                size: std::mem::size_of::<EmitterState>() as u64 * MAX_EMITTERS as u64,
                usage: hal::buffer::Usage::STORAGE | hal::buffer::Usage::TRANSFER_DST,
            },
            Dynamic,
        )?;

        initialize_buffers(
            factory,
            queue,
            &emitters,
            &emitter_state,
            &particles,
            &indirect,
        )?;

        let stats = factory.create_buffer(
            BufferInfo {
                size: std::mem::size_of::<Stats>() as u64,
                usage: hal::buffer::Usage::STORAGE,
            },
            Dynamic,
        )?;

        let layout = SHADER_REFLECTION.layout()?;

        log::trace!("Load shader module BOUNCE_COMPUTE");
        //let shader_set = SHADERS.build(factory, Default::default())?;
        let module = unsafe { COMPUTE.module(factory) }?;

        let set_layout =
            Handle::from(factory.create_descriptor_set_layout(layout.sets[0].bindings.clone())?);

        let pipeline_layout = unsafe {
            factory.device().create_pipeline_layout(
                std::iter::once(set_layout.raw()),
                std::iter::empty::<(hal::pso::ShaderStageFlags, std::ops::Range<u32>)>(),
            )
        }?;

        let pipeline = unsafe {
            factory.device().create_compute_pipeline(
                &hal::pso::ComputePipelineDesc {
                    shader: hal::pso::EntryPoint {
                        entry: "main",
                        module: &module,
                        specialization: hal::pso::Specialization::default(),
                    },
                    layout: &pipeline_layout,
                    flags: hal::pso::PipelineCreationFlags::empty(),
                    parent: hal::pso::BasePipeline::None,
                },
                None,
            )
        }?;

        unsafe { factory.destroy_shader_module(module) };

        let descriptor_set = factory.create_descriptor_set(set_layout.clone())?;

        unsafe {
            factory
                .device()
                .write_descriptor_sets(std::iter::once(hal::pso::DescriptorSetWrite {
                    set: descriptor_set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![
                        hal::pso::Descriptor::Buffer(
                            emitters.raw(),
                            Some(0)..Some(emitters.size()),
                        ),
                        hal::pso::Descriptor::Buffer(
                            emitter_state.raw(),
                            Some(0)..Some(emitter_state.size()),
                        ),
                        hal::pso::Descriptor::Buffer(
                            particles.raw(),
                            Some(0)..Some(particles.size()),
                        ),
                        hal::pso::Descriptor::Buffer(
                            indirect.raw(),
                            Some(0)..Some(indirect.size()),
                        ),
                        hal::pso::Descriptor::Buffer(stats.raw(), Some(0)..Some(stats.size())),
                    ],
                }));
        }

        let mut command_pool = factory
            .create_command_pool(family)?
            .with_capability::<Compute>()
            .expect("Graph builder must provide family with Compute capability");
        let initial = command_pool.allocate_buffers(1).remove(0);
        let mut recording = initial.begin(MultiShot(SimultaneousUse), ());
        let mut encoder = recording.encoder();
        encoder.bind_compute_pipeline(&pipeline);
        unsafe {
            encoder.bind_compute_descriptor_sets(
                &pipeline_layout,
                0,
                std::iter::once(descriptor_set.raw()),
                std::iter::empty::<u32>(),
            );

            {
                let (stages, barriers) = gfx_acquire_barriers(ctx, &*buffers, None);
                log::info!("Acquire {:?} : {:#?}", stages, barriers);
                encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
            }
            encoder.dispatch(MAX_PARTICLES / 1, 1, 1);
            {
                let (stages, barriers) = gfx_release_barriers(ctx, &*buffers, None);
                log::info!("Release {:?} : {:#?}", stages, barriers);
                encoder.pipeline_barrier(stages, hal::memory::Dependencies::empty(), barriers);
            }
        }

        let (submit, command_buffer) = recording.finish().submit();

        Ok(Self::Node {
            emitters,
            emitter_state,
            stats,
            set_layout,
            pipeline_layout,
            pipeline,
            descriptor_set,
            command_pool,
            command_buffer,
            submit,
        })
    }
}
