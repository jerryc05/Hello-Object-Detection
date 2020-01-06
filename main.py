import utils
import asyncio


async def main():
    pipeline_config_path = './config/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync.config'
    model_dir = './config'
    await utils.train_legacy_async(pipeline_config_path, model_dir)


asyncio.run(main())