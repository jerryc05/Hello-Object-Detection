class TfObjectDetector(object):
    def __init__(self,
                 frozen_graph_pb_path: str,
                 label_map_pbtxt_path: str,
                 num_classes=None,
                 input_size=None,
                 cpu_only=False):
        import os
        # frozen_graph_pb_path
        if not os.path.isfile(frozen_graph_pb_path):
            raise FileNotFoundError(f'File [{frozen_graph_pb_path}] not found!')
        self.frozen_graph_pb_path = frozen_graph_pb_path

        # label_map_pbtxt_path
        if not os.path.isfile(label_map_pbtxt_path):
            raise FileNotFoundError(f'File [{label_map_pbtxt_path}] not found!')
        self.label_map_pbtxt_path = label_map_pbtxt_path

        # num_classes
        if num_classes:
            self.num_classes = num_classes
        else:
            with open(label_map_pbtxt_path) as f:
                for line in reversed(f.readlines()):
                    line = line.strip()
                    if line.startswith('id'):
                        line = line[2:].strip()
                        if line.startswith(':'):
                            self.num_classes = int(line[1:].strip())
                            break
        assert isinstance(self.num_classes, int)

        from object_detection.utils import label_map_util
        self.category_index: dict = label_map_util.create_category_index(
            label_map_util.convert_label_map_to_categories(
                label_map_util.load_labelmap(
                    self.label_map_pbtxt_path
                ),
                max_num_classes=self.num_classes,
                use_display_name=True
            )
        )

        assert isinstance(cpu_only, bool)
        if cpu_only:
            import os
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        import tensorflow as tf
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            try:
                TfGraphDef = tf.compat.v1.GraphDef
                TfGFile = tf.io.gfile.GFile
            except AttributeError:
                TfGraphDef = tf.GraphDef
                TfGFile = tf.gfile.GFile

            graph_def = TfGraphDef()
            with TfGFile(self.frozen_graph_pb_path, 'rb') as gf:
                graph_def.ParseFromString(gf.read())
                tf.import_graph_def(graph_def, name='')

            # input_size
            if input_size:
                self.input_size = input_size
            else:
                import re
                pattern = re.compile('^Preprocessor\/map\/[A-Za-z]+(:?\/.*)?$')
                for node in graph_def.node:
                    try:
                        if pattern.search(node.name):
                            dim = node.attr['element_shape'].shape.dim
                            if dim:
                                self.input_size = (dim[0].size, dim[1].size)
                                if input(
                                        f'Variable ${{input_size}} detected: [{self.input_size}]!\n'
                                        f'Please confirm [Y/n]?'
                                ).strip().upper() == 'N':
                                    exit(1)
                                break
                    except Exception as e:
                        print(e)
                        pass

    def detect(self, filename):
        import tensorflow as tf
        try:
            decode_jpeg = tf.io.decode_jpeg
            read_file = tf.io.read_file
            resize = tf.image.resize
        except AttributeError:
            decode_jpeg = tf.image.decode_jpeg
            read_file = tf.read_file
            resize = tf.image.resize_images

        with self.detection_graph.as_default():
            image_decoded: tf.Tensor = decode_jpeg(read_file(filename))
            image_resized = resize(image_decoded, [self.input_size[0], self.input_size[1]])

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_reshaped = tf.reshape(
                image_resized, [1, self.input_size[0], self.input_size[1], 3])

            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            # with tf.Session(graph=self.detection_graph, config=config) as sess:
            try:
                Session = tf.compat.v1.Session
            except AttributeError:
                Session = tf.Session
            with Session(graph=self.detection_graph) as sess:
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                boxes, scores, classes, num_detections = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: (image_reshaped.eval())}
                )

                # Visualization of the results of a detection.
                from object_detection.utils import visualization_utils as vis_util
                import numpy as np

                # Code is broken!!
                # image_resized_np = image_resized.eval()
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     image_resized_np,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     self.category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=8
                # )

                # Visualization postprocessing
                # image_restored = resize(image_resized_np, [width, height]).eval()
                image_decoded_np = image_decoded.eval()

        # Optional stdout output
        result = []
        for i, x in enumerate(scores[0]):
            result.append({'score': x, 'index': i})
        from operator import itemgetter
        result.sort(key=itemgetter('score'), reverse=True)
        result = [x for x in result if x['score'] * 3 > 1]
        for x in result:
            x['name'] = self.category_index[int(classes[0][x['index']])]["name"]
            x['box'] = boxes[0][x['index']]
            print(f'{x["name"]:30}: {x["score"] * 100:5.03f}% | {x["box"]}')

        # Show image to screen
        import cv2
        image_decoded_cv2 = cv2.imread(filename)
        cv2.namedWindow(TfObjectDetector.__name__, cv2.WINDOW_KEEPRATIO)

        # width, height, _ = image_decoded_np.shape
        # for x in result:
        #     x1 = int(x['box'][0] * ( height))# /self.input_size[0]))
        #     y1 = int(x['box'][1] * ( width))# /self.input_size[1]))
        #     x2 = int(x['box'][2] * ( height))#/self.input_size[0]))
        #     y2 = int(x['box'][3] * ( width))#/self.input_size[1] ))
        #     print(x1, y1, x2, y2)
        #     cv2.rectangle(image_decoded_cv2, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=1)

        cv2.imshow(TfObjectDetector.__name__, image_decoded_cv2)
        cv2.waitKey(0)