import cv2
import numpy as np
from object_detection.utils import label_map_util
import os
import tensorflow as tf
from object_detection.utils import visualization_utils as vis_util


class TfObjectDetector(object):
    def __init__(self,
                 frozen_graph_pb_path: str,
                 label_map_pbtxt_path: str,
                 num_classes=None,
                 input_size=None):
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
                                self.input_size = dim[0].size
                                if input(
                                        f'Variable ${{input_size}} detected: [{self.input_size}]!\n'
                                        f'Please confirm [y/N]?'
                                ).strip().upper() != 'Y':
                                    exit(1)
                                break
                    except Exception as e:
                        print(e)
                        pass

            self.max_uint8_constant = tf.constant([tf.uint8.max], dtype=tf.float32)

        self.category_index: dict = label_map_util.create_category_index(
            label_map_util.convert_label_map_to_categories(
                label_map_util.load_labelmap(
                    self.label_map_pbtxt_path
                ),
                max_num_classes=self.num_classes,
                use_display_name=True
            )
        )

    def detect(self, filename):
        try:
            decode_jpeg = tf.io.image.decode_jpeg
            read_file = tf.io.read_file
        except AttributeError:
            decode_jpeg = tf.image.decode_jpeg
            read_file = tf.read_file


        with self.detection_graph.as_default():
            image_decoded: tf.Tensor = decode_jpeg(read_file(filename))
            image = tf.image.resize_images(image_decoded, [self.input_size, self.input_size])
            # image = tf.divide(tf.subtract(image, self.max_uint8_constant),
            #                   self.max_uint8_constant, name=None)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = tf.reshape(image, [1, self.input_size, self.input_size, 3])
            image_expanded = tf.cast(image_expanded, tf.float32)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(graph=self.detection_graph, config=config) as sess:
                # image_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

                # Actual detection.
                boxes, scores, classes, num_detections = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: (image_expanded.eval(session=sess))}
                )

                scores_list=[]
                for i, x in enumerate(scores[0]):
                    scores_list.append((i, x))
                import operator
                scores_list.sort(key=operator.itemgetter(1))
                print(scores_list)

                print()
                print()
                # print(boxes)
                print()
                print()
                # print(scores.shape)
                # for s in scores:
                #     print(max(s))
                # print(max(scores[0]))
                print()
                print()
                # print(classes)
                print()
                print()

                # # Visualization of the results of a detection.
                # vis_util.visualize_boxes_and_labels_on_image_array(
                #     image,
                #     np.squeeze(boxes),
                #     np.squeeze(classes).astype(np.int32),
                #     np.squeeze(scores),
                #     self.category_index,
                #     use_normalized_coordinates=True,
                #     line_thickness=8
                # )

        # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        # cv2.imshow("detection", image)
        # cv2.waitKey(0)

# if __name__ == '__main__':
# image = cv2.imread('/home/lyf/test/000084.jpg')
# detecotr = TfObjectDetector()
# detecotr.detect(image)