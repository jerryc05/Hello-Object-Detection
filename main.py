import utils
import asyncio


# Just a demo ↓
async def main():
    pipeline_config_path = 'config_faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco.config'
    model_dir = 'config_faster_rcnn'
    # ckpt_path_prefix_only = './config/model.ckpt-25000'
    await utils.train_legacy_async(pipeline_config_path, model_dir)#, ckpt_path_prefix_only)


    # detector = utils.TfObjectDetector('config/infer_graph/frozen_inference_graph.pb',
    #                                   'data/train/label_map.pbtxt')
    # detector.detect('data/train/data/0ac7f7a3-2748-4f58-ae8d-426cdc5113f4___FAM_B.Msls 4432.JPG')


asyncio.run(main())