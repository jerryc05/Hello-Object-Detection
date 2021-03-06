import cv2 as _cv2
import numpy as _np
import os as _os
from typing import Iterable as _Iterable
from time import time as _time


class TfObjectDetector(object):
    def __init__(self,
                 frozen_graph_pb_path: str,
                 label_map_pbtxt_path: str,
                 num_classes=None,
                 graph_input_size=None,
                 cpu_only=False):
        # frozen_graph_pb_path
        if not _os.path.isfile(frozen_graph_pb_path):
            raise FileNotFoundError(f'File [{frozen_graph_pb_path}] not found!')
        self.__frozen_graph_pb_path = frozen_graph_pb_path

        # label_map_pbtxt_path
        if not _os.path.isfile(label_map_pbtxt_path):
            raise FileNotFoundError(f'File [{label_map_pbtxt_path}] not found!')
        self.__label_map_pbtxt_path = label_map_pbtxt_path

        # num_classes
        if num_classes is not None:
            self.__num_classes = num_classes
        else:
            with open(label_map_pbtxt_path) as f:
                lines = reversed(f.readlines())
            for line in lines:
                line = line.strip()
                if line.startswith('id'):
                    line = line[2:].strip()
                    if line.startswith(':'):
                        self.__num_classes = int(line[1:].strip())
                        break
        assert isinstance(self.__num_classes, int)

        assert isinstance(cpu_only, bool)
        if cpu_only:
            _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # placeholder
        self.__detection_graph = None
        self.__input_size = None
        self.__input_size__ = graph_input_size
        self.__category_index = None
        self.__tf = None

    # noinspection PyPep8Naming,PyUnresolvedReferences,PyBroadException
    def __enter__(self):
        import tensorflow as tf
        from object_detection.utils import label_map_util

        label_map = label_map_util.load_labelmap(self.__label_map_pbtxt_path)
        categories = label_map_util.convert_label_map_to_categories(label_map, self.__num_classes)
        self.__category_index = label_map_util.create_category_index(categories)

        self.__tf = tf
        detection_graph = tf.Graph()
        self.__detection_graph = detection_graph
        with detection_graph.as_default():
            try:
                TfGraphDef = tf.compat.v1.GraphDef
                TfGFile = tf.io.gfile.GFile
            except AttributeError:
                TfGraphDef = tf.GraphDef
                TfGFile = tf.gfile.GFile

            graph_def = TfGraphDef()
            with TfGFile(self.__frozen_graph_pb_path, 'rb') as gf:
                graph_def.ParseFromString(gf.read())
                tf.import_graph_def(graph_def, name='')

            # input_size parsing
            __input_size__ = self.__input_size__
            input_size: tuple
            if isinstance(__input_size__, int):
                input_size = (__input_size__, __input_size__)
            elif (isinstance(__input_size__, tuple) and
                  isinstance(__input_size__[0], int) and
                  isinstance(__input_size__[1], int)):
                input_size = __input_size__
            else:
                import re

                print(f'${{input_size}} not found! Trying auto detection...')
                pattern = re.compile('^Preprocessor/map/[A-Za-z]+(:?/.*)*$')

                for node in graph_def.node:
                    try:
                        if pattern.search(node.name):
                            dim = node.attr['element_shape'].shape.dim
                            if dim:
                                input_size = (dim[0].size, dim[1].size)
                                if input(
                                        f'${{input_size}} detected from graph node: '
                                        f'[{input_size}]!\n'
                                        f'Confirm [Y/n]?'
                                ).strip().upper() == 'N':
                                    continue
                                else:
                                    break
                    except:
                        pass
                if input_size is None:
                    input_size = input('Width = '), input('Height = ')
            self.__input_size = input_size
            print(f'${{input_size}} = [{input_size}]!')
            del __input_size__
            del self.__input_size__

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(graph=self.detection_graph, config=config) as sess:
        try:
            Session = tf.compat.v1.Session
        except AttributeError:
            Session = tf.Session
        tf_session = Session(graph=self.__detection_graph)
        self.__tf_session = tf_session
        tf_session.__enter__()
        print('Tensorflow session initialized successfully!'.center(90, '·'))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _cv2.destroyAllWindows()
        self.__tf_session.__exit__(exc_type, exc_val, exc_tb)

    def detect_from_ndarray(self, np_arr: _np.ndarray,
                            no_wait: bool = None,
                            converter: list = None,
                            threshold: float = None,
                            resize_cv2_output: bool = None,
                            window_name: str = None):
        start = _time()
        if not isinstance(threshold, float):
            threshold = .5
        if not isinstance(resize_cv2_output, bool):
            resize_cv2_output = True
        if not isinstance(no_wait, bool):
            no_wait = False
        if not isinstance(window_name, str):
            window_name = 'TfObjectDetector'

        detection_graph = self.__detection_graph
        if detection_graph is None:
            import utils.log_helper

            print(f'{utils.log_helper.str_error}\n'
                  'Please run detection under "with" block!')
            exit(1)
        with detection_graph.as_default():
            print()
            print(window_name.center(90, '·'))

            input_width, input_height = self.__input_size
            image_as_f32 = np_arr
            width, height, _ = image_as_f32.shape
            if image_as_f32.dtype != _np.float32:
                image_as_f32 = image_as_f32.astype(_np.float32, copy=False)

            image_resized = image_as_f32
            if width != input_width and height != input_height:
                image_resized: _np.ndarray = \
                    _cv2.resize(image_as_f32, (input_width, input_height))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_reshaped = image_resized.reshape((1, input_width, input_height, 3))

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            boxes, scores, classes, num_detections = self.__tf_session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_reshaped}
            )
            boxes = boxes[0]
            scores = scores[0]
            classes = classes[0].astype(_np.int32, copy=False)

            # Visualization of the results of a detection.
            image_labeled: _np.ndarray = np_arr
            if image_labeled.dtype != _np.uint8:
                image_labeled = image_labeled.astype(_np.uint8, copy=False)

            image_labeled = image_labeled[..., ::-1]  # Required by cv2
            if isinstance(converter, _Iterable):
                for x in converter:
                    _cv2.cvtColor(image_labeled, x, dst=image_labeled)

            category_index = self.__category_index

            # Resize cv2 output
            if resize_cv2_output:
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
                min_score_thresh=threshold,
                line_thickness=1
            )

            # for i in range(len(boxes)):
            #     if scores[i] <= threshold:
            #         continue
            #     x1 = int(boxes[i][1] * width)
            #     y1 = int(boxes[i][0] * height)
            #     x4 = int(boxes[i][3] * width)
            #     y4 = int(boxes[i][2] * height)
            #     # print(f'box = ({x1},{y1}), ({x4},{y4})!')
            #     image_labeled = _cv2.rectangle(
            #         image_labeled,
            #         (x1, y1),
            #         (x4, y4),
            #         color=(266, 43, 138),
            #         thickness=1
            #     )
            #
            #     image_labeled = _cv2.putText(
            #         image_labeled,
            #         f'{category_index[classes[i]]["name"]}:{scores[i]:.2%}',
            #         (x1, y1),
            #         fontFace=_cv2.FONT_HERSHEY_DUPLEX,
            #         fontScale=1,
            #         color=(266, 43, 138),
            #         thickness=1,
            #         lineType=_cv2.LINE_AA
            #     )

            # Show image to screen
            # _cv2.namedWindow(window_name, _cv2.WINDOW_KEEPRATIO)
            _cv2.imshow(window_name, image_labeled)

            # Optional stdout output
            result = []
            for i, x in enumerate(scores):
                result.append({'score': x, 'index': i})
            from operator import itemgetter
            result.sort(key=itemgetter('score'), reverse=True)
            result = [x for x in result if x['score'] > threshold]
            for x in result:
                x['name'] = self.__category_index[int(classes[x['index']])]["name"]
                try:
                    x['box'] = boxes[x['index']]
                except:
                    x['box'] = []
                print(f'{x["name"]}\t: {x["score"]:.15%}')

            print(f'\nTime elapsed\t: {(_time() - start) * 1000:20.15f}\tms')
            _cv2.waitKey(1 if no_wait else 0)

    def detect_from_file(self, filename: str,
                         no_wait: bool = None,
                         converter: list = None,
                         threshold: float = None,
                         resize_cv2_output: bool = None,
                         window_name: str = None):
        if not _os.path.isfile(filename):
            from .log_helper import str_error

            print(f'{str_error}\n'
                  f'File [{filename} not found]!')
        else:
            if not isinstance(window_name, str):
                _window_name = filename
            else:
                _window_name = window_name

            return self.detect_from_ndarray(_cv2.imread(filename)[..., ::-1],
                                            no_wait=no_wait,
                                            converter=converter,
                                            threshold=threshold,
                                            resize_cv2_output=resize_cv2_output,
                                            window_name=_window_name)

    def category_index(self):
        return self.__category_index

    def _detect(self, np_arr: _np.ndarray,
                window_name: str = None):
        if not isinstance(window_name, str):
            window_name = 'TfObjectDetector'

        detection_graph = self.__detection_graph
        if detection_graph is None:
            import utils.log_helper

            print(f'{utils.log_helper.str_error}\n'
                  'Please run detection under "with" block!')
            exit(1)
        with detection_graph.as_default():
            print()
            print(window_name.center(90, '·'))

            input_width, input_height = self.__input_size
            image_as_f32 = np_arr
            width, height, _ = image_as_f32.shape
            if image_as_f32.dtype != _np.float32:
                image_as_f32 = image_as_f32.astype(_np.float32, copy=False)

            image_resized = image_as_f32
            if width != input_width and height != input_height:
                image_resized: _np.ndarray = \
                    _cv2.resize(image_as_f32, (input_width, input_height))

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_reshaped = image_resized.reshape((1, input_width, input_height, 3))

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            boxes, scores, classes, num_detections = self.__tf_session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_reshaped}
            )
            boxes = boxes[0]
            scores = scores[0]
            classes = classes[0].astype(_np.int32, copy=False)
            return boxes, scores, classes