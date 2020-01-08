import utils
import asyncio


# Just a demo ↓
async def main():
    import os

    # pipeline_config_path = 'config_faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco.config'
    # model_dir = 'config_faster_rcnn'
    # ckpt_path_prefix_only = './config/model.ckpt-25000'
    # await utils.train_legacy_async(pipeline_config_path, model_dir)#, ckpt_path_prefix_only)
    detector = utils.TfObjectDetector('config/infer_graph/frozen_inference_graph.pb',
                                      'data/train/label_map.pbtxt',
                                      input_size=640,
                                      cpu_only=False)
    with detector as detector_sess:
        for root, _, files in os.walk(
                r'F:\PycharmProjects\Hello-Object-Detection\data\PlantVillage-Dataset-master\raw'
                r'\color\Potato___Early_blight'):
            for file in files:
                detector_sess.detect(os.path.join(root, file))


asyncio.run(main())