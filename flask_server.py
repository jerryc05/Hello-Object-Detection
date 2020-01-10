while ...:
    try:
        from flask import *
        import os
        import socket
        import time
        import numpy as _np
        import cv2 as _cv2
        import json

        break
    except ImportError:
        # noinspection PyProtectedMember
        from pip._internal.main import main as _pip

        _pip(['install', '-U', 'flask'])
        del _pip

try:
    from utils.eval import TfObjectDetector
except ImportError as e:
    print(e)
    exit(1)

with TfObjectDetector('config/infer/frozen_inference_graph.pb',
                      'data/train/label_map.pbtxt') as detect_sess:
    app = Flask(__name__)
    cwd = os.getcwd()


    @app.route('/plant', methods=('POST',))
    def plant():
        data: str = request.json["data"]
        width = int(request.json["width"])
        height = int(request.json["height"])
        if not data:
            return abort(400)
        data2: _np.ndarray = _np.array(data)
        print(data2.shape)
        data2 = data2.reshape((width, height, 3))
        boxes, scores, classes = detect_sess._detect(data2)  # _cv2.imread(filename)[..., ::-1])
        return json.dumps({
            'scores' : scores.tolist(),
            'classes': classes.tolist(),
            'boxes'  : boxes.tolist()
        })

        # Visualization of the results of a detection.
        image_labeled = data
        if image_labeled.dtype != _np.uint8:
            image_labeled = image_labeled.astype(_np.uint8, copy=False)

        image_labeled = image_labeled[..., ::-1]  # Required by cv2

        category_index = detect_sess.category_index()

        # Resize cv2 output
        width, height, _ = data.shape
        if ...:
            ratio = min(1800 / width, 1000 / height)
            width = int(width * ratio)
            height = int(height * ratio)
            image_labeled = _cv2.resize(image_labeled, (width, height),
                                        interpolation=_cv2.INTER_LANCZOS4)
            print(f'Image size adjusted to [{image_labeled.shape}]!')

        # Add labels and boxes
        if 'vis_util' not in locals():
            from object_detection.utils import visualization_utils as vis_util

        # noinspection PyUnboundLocalVariable
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_labeled,
            boxes,
            classes,
            scores,
            category_index,
            use_normalized_coordinates=True,
            min_score_thresh=.5,
            line_thickness=1
        )

        _cv2.imwrite('testtesttesttest.jpg', image_labeled)
        return f'Done!'


    @app.route('/dl')
    def dl():
        filename = request.args.get("file")
        print(f"file  = {filename}")
        token = request.args.get("token")
        print(f"token = {token}")
        formatted_time = time.strftime("%Y%m%d%H%M")
        print(f"key   = {formatted_time}")
        return send_file(filename) \
            if token == formatted_time \
            else abort(403)


    @app.route('/echo')
    def echo():
        return f"args = [{request.args}]\n" \
               f"path = [{request.path}]\n" \
               f"file = [{request.files}]\n" \
               f"hder = [{request.headers}]\n" \
               f"burl = [{request.base_url}]"


    if __name__ == '__main__':
        port = 6007
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('1.1.1.1', 1))
            print(f"Listening on [{s.getsockname()[0]}:{port}]")

        app.run(
            host='0.0.0.0',
            port=port,
            debug=True
        )

#  redirect(url_for('hello_admin'))