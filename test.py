import gradio as gr
import os
import tqdm, time


##################
from PIL import Image
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.point_cloud import PointCloud
#######################

from point_e.util.ply_util import convert

#out_3d = os.path.join(os.path.dirname(__file__), "63bbaeb2cf5517c87ab32f2e.glb")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_mesh(img, progress=gr.Progress()):
    progress(0, desc="Starting...")
    # time.sleep(1)
    # for i in progress.tqdm(range(100)):
    #     time.sleep(0.1)

    base_name = 'base40M' # use base300M or base1B for better results
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

    #print('creating upsample model...')
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

    #print('downloading base checkpoint...')
    base_model.load_state_dict(load_checkpoint(base_name, device))

    #print('downloading upsampler checkpoint...')
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))

    #print('Point Cloud Sampling...')
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, upsampler_model],
        diffusions=[base_diffusion, upsampler_diffusion],
        num_points=[1024, 4096 - 1024],
        aux_channels=['R', 'G', 'B'],
        guidance_scale=[3.0, 3.0],
    )

    # Produce a sample from the model.
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]
    #fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))


    with open('./tmp_out/test_pc.ply', 'wb') as f:
        pc.write_ply(f)

    #############################################

    name = 'sdf'
    mesh_model = model_from_config(MODEL_CONFIGS[name], device)
    mesh_model.eval()

    #print('loading SDF model...')
    mesh_model.load_state_dict(load_checkpoint(name, device))

    #Plot the point cloud as a sanity check.
    #fig = plot_point_cloud(pc, grid_size=2)

    # Produce a mesh (with vertex colors)
    #print('Producing a mesh...')
    out_3d = marching_cubes_mesh(
        pc=pc,
        model=mesh_model,
        batch_size=4096,
        grid_size=128, # increase to 128 for resolution used in evals (32)
        progress=True,
    )

    # with open('./tmp_out/test_mesh.ply', 'wb') as f:
    #      out_3d.write_ply(f)    
    # convert('./tmp_out/test_mesh.ply', './tmp_out/test_mesh.obj')

    with open('./tmp_out/test_mesh.obj', 'w') as f:
        out_3d.write_obj(f)

    out_3d_file = "./tmp_out/test_mesh.obj"

    #out_3d_file = "./tmp_out/humanoid_tri.obj"

    return out_3d_file



with gr.Blocks() as demo:
    gr.Markdown("""
    ***Realtime 3D Mesh!!!*** Take a picture or upload a file to make ***3D mesh***
     """)
    with gr.Row():
        inp = gr.Image(type="pil")
        out = gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0],  label="3D Model")
    btn = gr.Button("Run")
    btn.click(fn=make_mesh, inputs=[inp], outputs=[out])


demo.queue(concurrency_count=3)

if __name__ == "__main__":

#    demo.launch(share=True)
    demo.launch()
