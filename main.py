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

    detector = utils.TfObjectDetector('config/config2/infer/frozen_inference_graph.pb',
                                      'data/train2/label_map.pbtxt',
                                      # graph_input_size=640,
                                      cpu_only=False)
    camera = cv2.VideoCapture(0)
    with detector as detector_sess:
        while camera.isOpened():
            _, frame = camera.read()
            cv2.flip(frame, 1, dst=frame)
            detector_sess.detect_from_ndarray(frame[..., ::-1], no_wait=True, threshold=.7)

        # for root, _, files in os.walk(r'data\train2\data'):
        #     for file in files:
        #         detector_sess.detect_from_file(os.path.join(root, file), window_name='')

        # for root, _, files in os.walk(
        #         r'data\PlantVillage-Dataset-master\raw'
        #         r'\color\Apple___Cedar_apple_rust'):
        #     for file in files:
        #         detector_sess.detect_from_file(os.path.join(root, file), window_name='')
        # detector_sess.detect_from_file(r'data\PlantVillage-Dataset-master\raw\color\Apple___Cedar_apple_rust\0cd24b0c-0a9d-483f-8734-5c08988e029f___FREC_C.Rust 3762.JPG',
        #                                window_name='')
        # detector_sess.detect(r'data\PlantVillage-Dataset-master\raw\color\Apple___Cedar_apple_rust\0cd24b0c-0a9d-483f-8734-5c08988e029f___FREC_C.Rust 3762.JPG',
        #                               )


asyncio.run(main())