use once_cell::sync::Lazy;
use rendy::{
    command::{DrawCommand, QueueId, RenderPassEncoder},
    factory::Factory,
    graph::{
        render::{Layout, PrepareResult, SimpleGraphicsPipeline, SimpleGraphicsPipelineDesc},
        BufferAccess, GraphContext, NodeBuffer, NodeImage,
    },
    hal::{self},
    memory::Dynamic,
    mesh::PosTex,
    resource::{Buffer, BufferInfo, DescriptorSetLayout, Escape, Handle},
    shader::{
        ShaderKind, ShaderSet, SourceLanguage, SourceShaderInfo, SpirvReflection, SpirvShader,
    },
    util::types::vertex::VertexFormat,
};

static RENDER_VERTEX: Lazy<SpirvShader> = Lazy::new(|| {
    SourceShaderInfo::new(
        include_str!("../src/shaders/render.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/render.vert").into(),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    )
    .precompile()
    .unwrap()
});

static RENDER_FRAGMENT: Lazy<SpirvShader> = Lazy::new(|| {
    SourceShaderInfo::new(
        include_str!("../src/shaders/render.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/shaders/render.frag").into(),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    )
    .precompile()
    .unwrap()
});

static SHADERS: Lazy<rendy::shader::ShaderSetBuilder> = Lazy::new(|| {
    rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*RENDER_VERTEX)
        .unwrap()
        .with_fragment(&*RENDER_FRAGMENT)
        .unwrap()
});

static SHADER_REFLECTION: Lazy<SpirvReflection> = Lazy::new(|| SHADERS.reflect().unwrap());

use crate::compute::*;

#[derive(Debug, Default)]
pub struct RenderNodeDesc;

#[derive(Debug)]
pub struct RenderNode<B: hal::Backend> {
    vertices: Escape<Buffer<B>>,
    instance: Handle<Buffer<B>>,
    indirect: Handle<Buffer<B>>,
    //descriptor_set: Escape<DescriptorSet<B>>,
}

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for RenderNodeDesc
where
    B: hal::Backend,
    T: ?Sized,
{
    type Pipeline = RenderNode<B>;

    fn load_shader_set(&self, factory: &mut Factory<B>, _aux: &T) -> ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn depth_stencil(&self) -> Option<hal::pso::DepthStencilDesc> {
        None
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<hal::pso::Element<hal::format::Format>>,
        hal::pso::ElemStride,
        hal::pso::VertexInputRate,
    )> {
        return vec![
            SHADER_REFLECTION
                //.attributes(&["position", "uv"])
                .attributes_range(0..2)
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            // We are not using all the fields in the particles
            // structure, so we have to specify the stride?
            VertexFormat::with_stride(
                SHADER_REFLECTION.attributes_range(2..4).unwrap(),
                std::mem::size_of::<Particle>() as u32,
            )
            .gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ];
    }

    fn layout(&self) -> Layout {
        return SHADER_REFLECTION.layout().unwrap();
    }

    fn buffers(&self) -> Vec<BufferAccess> {
        vec![
            BufferAccess {
                access: hal::buffer::Access::SHADER_READ | hal::buffer::Access::SHADER_WRITE,
                stages: hal::pso::PipelineStage::VERTEX_SHADER,
                usage: hal::buffer::Usage::STORAGE
                    | hal::buffer::Usage::TRANSFER_DST
                    | hal::buffer::Usage::VERTEX,
            },
            BufferAccess {
                access: hal::buffer::Access::SHADER_READ
                    | hal::buffer::Access::SHADER_WRITE
                    | hal::buffer::Access::INDIRECT_COMMAND_READ,
                stages: hal::pso::PipelineStage::VERTEX_SHADER,
                usage: hal::buffer::Usage::STORAGE
                    | hal::buffer::Usage::INDIRECT
                    | hal::buffer::Usage::TRANSFER_DST,
            },
        ]
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _aux: &T,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<RenderNode<B>, failure::Error> {
        assert_eq!(buffers.len(), 2);
        assert!(images.is_empty());

        let instance = ctx.get_buffer(buffers[0].id).unwrap();
        let indirect = ctx.get_buffer(buffers[1].id).unwrap();

        // let mut instance = factory
        //     .create_buffer(
        //         BufferInfo {
        //             size: std::mem::size_of::<Particle>() as u64,
        //             usage: hal::buffer::Usage::INDIRECT,
        //         },
        //         Dynamic,
        //     )
        //     .unwrap();

        // let mut indirect = factory
        //     .create_buffer(
        //         BufferInfo {
        //             size: std::mem::size_of::<DrawCommand>() as u64,
        //             usage: hal::buffer::Usage::INDIRECT,
        //         },
        //         Dynamic,
        //     )
        //     .unwrap();

        // unsafe {
        //     factory.upload_visible_buffer(
        //         &mut instance,
        //         0,
        //         &(0..1)
        //             .map(|_| Particle {
        //                 color: [0.0, 1.0, 0.0, 1.0],
        //                 ..Default::default()
        //             })
        //             .collect::<Vec<_>>(),
        //     )
        // }?;

        // unsafe {
        //     factory.upload_visible_buffer(
        //         &mut indirect,
        //         0,
        //         &(0..1)
        //             .map(|_i| DrawCommand {
        //                 vertex_count: 6,
        //                 instance_count: 1, // this gets set to one in the shader
        //                 first_vertex: 0,
        //                 first_instance: 0,
        //             })
        //             .collect::<Vec<_>>(),
        //     )
        // }?;

        // let mut indirect = factory
        //     .create_buffer(
        //         BufferInfo {
        //             size: std::mem::size_of::<DrawCommand>() as u64 * DIVIDE as u64,
        //             usage: hal::buffer::Usage::INDIRECT,
        //         },
        //         Dynamic,
        //     )
        //     .unwrap();

        // unsafe {
        //     factory
        //         .upload_visible_buffer(
        //             &mut indirect,
        //             0,
        //             &(0..DIVIDE)
        //                 .map(|index| DrawCommand {
        //                     vertex_count: 6,
        //                     instance_count: PER_CALL,
        //                     first_vertex: 0,
        //                     first_instance: index * PER_CALL,
        //                 })
        //                 .collect::<Vec<_>>(),
        //         )
        //         .unwrap();
        // }

        let mut vertices = factory.create_buffer(
            BufferInfo {
                size: std::mem::size_of::<PosTex>() as u64 * 6,
                usage: hal::buffer::Usage::VERTEX,
            },
            Dynamic,
        )?;

        unsafe {
            factory.upload_visible_buffer(
                &mut vertices,
                0,
                &[
                    PosTex {
                        position: [-0.5, -0.5, 0.0].into(),
                        tex_coord: [0.0, 0.0].into(),
                    },
                    PosTex {
                        position: [-0.5, 0.5, 0.0].into(),
                        tex_coord: [0.0, 1.0].into(),
                    },
                    PosTex {
                        position: [0.5, 0.5, 0.0].into(),
                        tex_coord: [1.0, 1.0].into(),
                    },
                    PosTex {
                        position: [-0.5, -0.5, 0.0].into(),
                        tex_coord: [0.0, 0.0].into(),
                    },
                    PosTex {
                        position: [0.5, 0.5, 0.0].into(),
                        tex_coord: [1.0, 1.0].into(),
                    },
                    PosTex {
                        position: [0.5, -0.5, 0.0].into(),
                        tex_coord: [1.0, 0.0].into(),
                    },
                ],
            )?;
        }

        // The panic on indexing out of bounds is actually more descriptive...
        //assert_eq!(set_layouts.len(), 1);

        // let descriptor_set = factory.create_descriptor_set(set_layouts[0].clone())?;

        // unsafe {
        //     factory
        //         .device()
        //         .write_descriptor_sets(std::iter::once(hal::pso::DescriptorSetWrite {
        //             set: descriptor_set.raw(),
        //             binding: 0,
        //             array_offset: 0,
        //             descriptors: vec![hal::pso::Descriptor::Buffer(
        //                 instance.raw(),
        //                 Some(0)..Some(instance.size() as u64),
        //             )],
        //         }))
        // }

        Ok(Self::Pipeline {
            vertices,
            indirect: indirect.clone(),
            instance: instance.clone(),
            //descriptor_set,
        })
    }
}

const STRIDE: u32 = std::mem::size_of::<DrawCommand>() as u32;

impl<B, T> SimpleGraphicsPipeline<B, T> for RenderNode<B>
where
    B: hal::Backend,
    T: ?Sized,
{
    type Desc = RenderNodeDesc;

    fn prepare(
        &mut self,
        _factory: &Factory<B>,
        _queue: QueueId,
        _sets: &[Handle<DescriptorSetLayout<B>>],
        _index: usize,
        _aux: &T,
    ) -> PrepareResult {
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        _layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        unsafe {
            // encoder.bind_graphics_descriptor_sets(
            //     layout,
            //     0,
            //     std::iter::once(self.descriptor_set.raw()),
            //     std::iter::empty::<u32>(),
            // );

            // encoder
            //     .bind_vertex_buffers(0, vec![(self.vertices.raw(), 0), (self.instance.raw(), 0)]);

            encoder.bind_vertex_buffers(0, vec![(self.vertices.raw(), 0)]);
            encoder.bind_vertex_buffers(1, vec![(self.instance.raw(), 0)]);

            //for i in 0..MAX_PARTICLES {
            encoder.draw_indirect(
                self.indirect.raw(),
                //STRIDE as u64,
                //1,
                //i.into(),
                0,
                MAX_PARTICLES,
                // skip first one
                //std::mem::size_of::<DrawCommand>() as u64,
                //MAX_PARTICLES - 1,
                STRIDE,
            );
            //}
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &T) {}
}
