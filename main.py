import utils
import asyncio


# Just a demo ↓
async def main():
    import os

    # pipeline_config_path = 'config_faster_rcnn/faster_rcnn_inception_resnet_v2_atrous_coco.config'
    # model_dir = 'config_faster_rcnn'
    # ckpt_path_prefix_only = './config/model.ckpt-25000'
    # await utils.train_legacy_async(pipeline_config_path, model_dir)#, ckpt_path_prefix_only)
    import cv2

    camera = cv2.VideoCapture(0)

    detector = utils.TfObjectDetector('config/config2/infer/frozen_inference_graph.pb',
                                      'data/train2/label_map.pbtxt',
                                      input_size=(int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                  int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))),
                                      cpu_only=False)
    with detector as detector_sess:
        while camera.isOpened():
            _, frame = camera.read()
            # cv2.imshow('',cv2.flip(frame, 1))
            # cv2.waitKey(1)
            detector_sess.detect_from_np(cv2.flip(frame, 1), no_wait=True)

        # for root, _, files in os.walk(
        #         r'F:\PycharmProjects\Hello-Object-Detection\data\PlantVillage-Dataset-master\raw'
        #         r'\color\Potato___Early_blight'):
        #     for file in files:
        #         detector_sess.detect(os.path.join(root, file))


asyncio.run(main())