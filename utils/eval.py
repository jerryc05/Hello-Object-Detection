import cv2 as _cv2
import numpy as _np
import os as _os


class TfObjectDetector(object):
    def __init__(self,
                 frozen_graph_pb_path: str,
                 label_map_pbtxt_path: str,
                 num_classes=None,
                 input_size=None,
                 cpu_only=False):
        import os
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
            import os
            _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # placeholder
        self.__detection_graph = None
        self.__input_size = None
        self.__input_size__ = input_size
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
        self.__detection_graph = tf.Graph()
        with self.__detection_graph.as_default():
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
            if isinstance(__input_size__, int):
                self.__input_size = (__input_size__, __input_size__)
            elif (isinstance(__input_size__, tuple) and
                  isinstance(__input_size__[0], int) and
                  isinstance(__input_size__[1], int)):
                self.__input_size = __input_size__
            else:
                import re

                print(f'${{input_size}} not found! Trying auto detection...')
                pattern = re.compile('^Preprocessor/map/[A-Za-z]+(:?/.*)*$')

                # placeholder
                self.__input_size = None
                for node in graph_def.node:
                    try:
                        if pattern.search(node.name):
                            dim = node.attr['element_shape'].shape.dim
                            if dim:
                                self.__input_size = (dim[0].size, dim[1].size)
                                if input(
                                        f'${{input_size}} detected from graph node: '
                                        f'[{self.__input_size}]!\n'
                                        f'Confirm [Y/n]?'
                                ).strip().upper() == 'N':
                                    continue
                                else:
                                    break
                    except:
                        pass
                if self.__input_size is None:
                    self.__input_size = input('Width = '), input('Height = ')
                print(f'Now ${{input_size}} = [{self.__input_size}]!')
            del __input_size__
            del self.__input_size__

        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # with tf.Session(graph=self.detection_graph, config=config) as sess:
        try:
            Session = tf.compat.v1.Session
        except AttributeError:
            Session = tf.Session
        self.__tf_session = Session(graph=self.__detection_graph)
        self.__tf_session.__enter__()
        print('Tensorflow session initialized successfully!'.center(90, '='))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _cv2.destroyAllWindows()

        self.__tf_session.__exit__(exc_type, exc_val, exc_tb)

    def detect_from_file(self, filename: str, no_wait=False):
        assert isinstance(no_wait, bool)

        detection_graph = self.__detection_graph
        if detection_graph is None:
            import utils.log_helper

            print(f'{utils.log_helper.str_error}\n'
                  'Please run detection under "with" block!')
            exit(1)
        with detection_graph.as_default():
            tf = self.__tf

            try:
                decode_jpeg = tf.io.decode_jpeg
                read_file = tf.io.read_file
                resize = tf.image.resize
            except AttributeError:
                decode_jpeg = tf.image.decode_jpeg
                read_file = tf.read_file
                resize = tf.image.resize_images
            image_decoded: tf.Tensor = decode_jpeg(read_file(filename))
            image_resized = resize(image_decoded, [self.__input_size[0], self.__input_size[1]])

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_reshaped = tf.reshape(
                image_resized, [1, self.__input_size[0], self.__input_size[1], 3])

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            boxes, scores, classes, num_detections = self.__tf_session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: (image_reshaped.eval())}
            )

            # Visualization of the results of a detection.
            from object_detection.utils import visualization_utils as vis_util

            # Add labels and boxes
            image_resized_np = image_resized.eval()
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_resized_np,
                _np.squeeze(boxes),
                _np.squeeze(classes).astype(_np.int32),
                _np.squeeze(scores),
                self.__category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )

            # Visualization postprocessing
            width, height, _ = tf.shape(image_decoded).eval()

            image_restored = resize(image_resized_np, [width, height]).eval().astype(_np.uint8)

            # Show image to screen
            # image_decoded_cv2 = cv2.imread(filename)
            base_name = _os.path.basename(filename)
            _cv2.namedWindow(base_name, _cv2.WINDOW_KEEPRATIO)

            # width, height, _ = image_decoded_np.shape
            # for x in result:
            #     x1 = int(x['box'][0] * (width))  # /self.input_size[0]))
            #     y1 = int(x['box'][1] * (height))  # /self.input_size[1]))
            #     x2 = int(x['box'][2] * (width))  # /self.input_size[0]))
            #     y2 = int(x['box'][3] * (height))  # /self.input_size[1] ))
            #     print(x1, y1, x2, y2)
            #     cv2.rectangle(image_decoded_cv2, (x1, y1), (x2, y2),
            #                   color=(17, 17, 255),
            #                   thickness=1)

            # cv2.imshow(TfObjectDetector.__name__, image_decoded_cv2)
            _cv2.imshow(base_name,
                        _cv2.cvtColor(image_restored, _cv2.COLOR_RGB2BGR))

            # Optional stdout output
            print()
            print(filename.center(90, '='))
            result = []
            for i, x in enumerate(scores[0]):
                result.append({'score': x, 'index': i})
            from operator import itemgetter
            result.sort(key=itemgetter('score'), reverse=True)
            result = [x for x in result if x['score'] * 4 > 1]
            for x in result:
                x['name'] = self.__category_index[int(classes[0][x['index']])]["name"]
                try:
                    x['box'] = boxes[0][x['index']]
                except:
                    x['box'] = []
                print(f'{x["name"]}\t: {x["score"] * 100:5.15f}%')

            _cv2.waitKey(1 if no_wait else 0)

    def detect_from_np(self, np_arr: _np.ndarray, no_wait=False, converter=None):
        if converter is None:
            converter = []
        assert isinstance(no_wait, bool)

        detection_graph = self.__detection_graph
        if detection_graph is None:
            import utils.log_helper

            print(f'{utils.log_helper.str_error}\n'
                  'Please run detection under "with" block!')
            exit(1)
        with detection_graph.as_default():
            tf = self.__tf

            try:
                # decode_jpeg = tf.io.decode_jpeg
                # read_file = tf.io.read_file
                resize = tf.image.resize
            except AttributeError:
                # decode_jpeg = tf.image.decode_jpeg
                # read_file = tf.read_file
                resize = tf.image.resize_images
            image_decoded: tf.Tensor = tf.convert_to_tensor(np_arr)
            image_resized = resize(image_decoded, [self.__input_size[0], self.__input_size[1]])

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_reshaped = tf.reshape(
                image_resized, [1, self.__input_size[0], self.__input_size[1], 3])

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Actual detection.
            from time import time
            s1=time()
            boxes, scores, classes, num_detections = self.__tf_session.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: (image_reshaped.eval())}
            )

            # Visualization of the results of a detection.
            s2=time()
            from object_detection.utils import visualization_utils as vis_util

            # Add labels and boxes
            image_resized_np:_np.ndarray = image_resized.eval()
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_resized_np,
                _np.squeeze(boxes),
                _np.squeeze(classes).astype(_np.int32),
                _np.squeeze(scores),
                self.__category_index,
                use_normalized_coordinates=True,
                line_thickness=4
            )
            s3=time()

            # Visualization postprocessing
            width, height, _ = np_arr.shape

            image_restored = resize(image_resized_np, [width, height])
            image_restored = tf.cast(image_restored, tf.uint8).eval()
            s4=time()

            # Show image to screen
            # image_decoded_cv2 = cv2.imread(filename)
            base_name = f'RAW ND-ARRAY SOURCE'
            _cv2.namedWindow(base_name, _cv2.WINDOW_KEEPRATIO)

            # width, height, _ = image_decoded_np.shape
            # for x in result:
            #     x1 = int(x['box'][0] * (width))  # /self.input_size[0]))
            #     y1 = int(x['box'][1] * (height))  # /self.input_size[1]))
            #     x2 = int(x['box'][2] * (width))  # /self.input_size[0]))
            #     y2 = int(x['box'][3] * (height))  # /self.input_size[1] ))
            #     print(x1, y1, x2, y2)
            #     cv2.rectangle(image_decoded_cv2, (x1, y1), (x2, y2),
            #                   color=(17, 17, 255),
            #                   thickness=1)

            # cv2.imshow(TfObjectDetector.__name__, image_decoded_cv2)
            if isinstance(converter,list) or isinstance(converter,tuple):
                for x in converter:
                    image_restored = _cv2.cvtColor(image_restored, x)
            _cv2.imshow(base_name, image_restored)

            # Optional stdout output
            print()
            print(base_name.center(90, '='))
            result = []
            for i, x in enumerate(scores[0]):
                result.append({'score': x, 'index': i})
            from operator import itemgetter
            result.sort(key=itemgetter('score'), reverse=True)
            result = [x for x in result if x['score'] * 4 > 1]
            for x in result:
                x['name'] = self.__category_index[int(classes[0][x['index']])]["name"]
                try:
                    x['box'] = boxes[0][x['index']]
                except:
                    x['box'] = []
                print(f'{x["name"]}\t: {x["score"] * 100:5.15f}%')

            _cv2.waitKey(1 if no_wait else 0)
            s5=time()
            print(f's1->s2: {s2-s1}')
            print(f's3->s2: {s3-s2}')
            print(f's4->s3: {s4-s3}')
            print(f's5->s4: {s5-s4}')
            # print(f's4->s3: {s4-s3}')