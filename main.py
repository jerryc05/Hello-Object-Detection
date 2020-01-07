import utils
import asyncio


# Just a demo ↓
async def main():
    pipeline_config_path = './config/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config'
    model_dir = './config/infer_graph/'
    ckpt_path_prefix_only = './config/model.ckpt-25000'
    await utils.export_inference_graph_async(pipeline_config_path, model_dir, ckpt_path_prefix_only)


asyncio.run(main())