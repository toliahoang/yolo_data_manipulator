import os
import numpy as np
import cv2
from PIL import Image

#from ocr import get_message
import matplotlib.pyplot as plt
from IPython.display import display



def convert_box(b, width, height):
    """Convert box from xywh format to x1y1x2y2 format, max width and heigth should not exceed limit"""
    x, y, w, h = b
    return max(0, x), max(0, y), min(x + w, width), min(y + h, height)

def try_ocr_for_single_image(objects, image):
    height, width, _ = image.shape
    class_eliminated = [8]
    for b, prob, class_id in objects:
        if class_id in class_eliminated:
            # boxes[class_id] = b
            x1, y1, x2, y2 = convert_box(b, width, height)
            x1 = max(x1 - 3, 0)
            x2 = min(x2 + 3, width)
            y1 = max(y1 - 3, 0)
            y2 = min(y2 + 3, height)

            if class_id == 8:
                is_playing = True
                message = get_message(image[y1:y2, x1:x2])
                if 'eliminated' in message:
                    is_eliminated = True
                    return True
                else:
                    return False





# use opencv instead of tf
class ImageClassifier:
    def __init__(self, model_folder='model', config_file='yolov3_apex_watermark.cfg',
                 label_file='obj.names',
                 # weight_file='yolov4-apex_last.weights'):
                 weight_file='yolov3_apex_watermark_final.weights'):
        self.model_folder = os.path.join(os.path.dirname(__file__), model_folder)
        self.config_file = config_file
        self.label_file = label_file
        self.weight_file = weight_file

        self.config = None
        self.net = None
        self.class_names = None
        self.output_layers = None

    def load_config(self):
        if self.config is None:
            cfgfile = os.path.join(self.model_folder, self.config_file)
            blocks = parse_cfg(cfgfile)

            net_block = find_block(blocks, 'net')
            model_shape = int(net_block['height']), int(net_block['width']), int(net_block['channels'])

            yolo_block = find_block(blocks, 'yolo')
            num_classes = int(yolo_block['classes'])

            self.config = (blocks, model_shape, num_classes)

        return self.config

    def load_class_names(self):
        if self.class_names is None:
            file = os.path.join(self.model_folder, self.label_file)
            with open(file, 'r') as f:
                self.class_names = f.read().splitlines()

        return self.class_names

    def get_class_id(self, name):
        return self.load_class_names().index(name)

    def get_num_classes(self):
        return len(self.load_class_names())

    def load_model(self):
        if self.net is None:
            weightfile = os.path.join(self.model_folder, self.weight_file)
            cfgfile = os.path.join(self.model_folder, self.config_file)
            self.net = cv2.dnn.readNet(weightfile, cfgfile)
            device = os.environ.get('device')
            if device and device.lower() == 'gpu':
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        return self.net

    def get_output_layers(self):
        if self.output_layers is None:
            self.output_layers = self.load_model().getUnconnectedOutLayersNames()

        return self.output_layers

    def run_inference_for_single_image_pure(self, image, image_format='BGR', threshold=0.5, nms_threshold=0.4):
        """Run inference for image"""
    
        height, width, channels = image.shape
    
        _, model_shape, _ = self.load_config()
    
        net = self.load_model()
    
        # if image is of RGB format then swapRB=False
        blob = cv2.dnn.blobFromImage(image,
                                     scalefactor=1.0 / 255,
                                     size=(model_shape[0], model_shape[1]),
                                     mean=(0, 0, 0),
                                     swapRB=(image_format == 'BGR'),
                                     crop=False)
        net.setInput(blob)
    
        # predict
        class_ids = []
        confidences = []
        boxes = []
        outs = net.forward(self.get_output_layers())
    
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > threshold:
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    left = int(width * detection[0] - w / 2)
                    top = int(height * detection[1] - h / 2)
                    boxes.append((left, top, w, h))
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
    
        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, threshold, nms_threshold)
        # TODO check indicies when no object detected
        if indices is not None and len(indices) > 0:
            indices = np.squeeze(indices, axis=-1)
            return [(boxes[i], confidences[i], class_ids[i]) for i in indices]
        else:
            return []









    def run_inference_for_single_image(self, image, image_format='BGR', threshold=0.5, nms_threshold=0.4):
        """Run inference for image"""
        height, width, channels = image.shape
        _, model_shape, _ = self.load_config()
        net = self.load_model()
        # if image is of RGB format then swapRB=False
        blob = cv2.dnn.blobFromImage(image,
                                    scalefactor=1.0 / 255,
                                    size=(model_shape[0], model_shape[1]),
                                    mean=(0, 0, 0),
                                    swapRB=(image_format == 'BGR'),
                                    crop=False)
        net.setInput(blob)
        outs = net.forward(self.get_output_layers())
        detections = np.concatenate(outs)
        scores = detections[:, 5:]
        cls_id = np.argmax(scores, axis=1)
        n_values = scores.shape[1]
        one_hot = np.eye(n_values)[cls_id.squeeze()]
        conf = np.max(scores * one_hot, axis=1)
        good_detections = detections[conf > threshold]
        bbox = good_detections[:, :4]
        w = (bbox[:, 2] * width).astype(np.int)
        h = (bbox[:, 3] * height).astype(np.int)
        left = (width * bbox[:, 0] - w / 2).astype(np.int)
        top = (height * bbox[:, 1] - h / 2).astype(np.int)
        new_bbox = np.vstack([left, top, w, h]).T.tolist()
        cls_id = cls_id[conf > threshold].tolist()
        conf = conf[conf > threshold].tolist()
        indices = cv2.dnn.NMSBoxes(new_bbox, conf, threshold, nms_threshold)
        if len(indices) > 0:
            #indices = np.squeeze(indices, axis=-1)
            return [(new_bbox[i], conf[i], cls_id[i]) for i in indices]
        else:
            return []

    def show_inference(self, image, image_format='BGR', threshold=0.5, nms_threshold=0.5, show=True):
        """Show detected object on image"""

        detected_objects = self.run_inference_for_single_image(image, image_format=image_format, threshold=threshold,
                                                               nms_threshold=nms_threshold)

        class_names = self.load_class_names()

        height, width, channels = image.shape
        shift = 40  # space for annotation
        img = np.zeros((height + shift, width, channels), dtype=np.uint8)  # this is where drawing happen
        img[shift:] = image.copy()

        for box, score, class_id in detected_objects:
            left, top = box[0], box[1]
            w, h = box[2], box[3]
            right, bottom = left + w, top + h

            print("top: ",top)
            print("w: ",h)
            print("bottom: ",bottom)
            # mid = top + h//2
            # if mid < (height//3):
            #     print("its failed champion")
            text = '{} {:.4f}'.format(class_names[class_id], score)
            img = cv2.rectangle(img, (left, top + shift), (right, bottom + shift), (0, 255, 0), 2)
            img = cv2.putText(img, text, (left, top + shift), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        """Test """
        # state_ocr = try_ocr_for_single_image(detected_objects,img)
        # if state_ocr:
        #     print("eliminated")
        # else:
        #     print("no eliminated")
        get_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        get_img.show()

        # if show:
        #     display(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))  # Image use RGB format
        #
        # else:
        #     return img

    def get_champion(self,name, image, image_format='BGR', threshold=0.5, nms_threshold=0.5, show=True):
        """Show detected object on image"""

        detected_objects = self.run_inference_for_single_image(image, image_format=image_format, threshold=threshold,
                                                               nms_threshold=nms_threshold)

        class_names = self.load_class_names()

        height, width, channels = image.shape
        shift = 40  # space for annotation
        img = np.zeros((height + shift, width, channels), dtype=np.uint8)  # this is where drawing happen
        img[shift:] = image.copy()

        for box, score, class_id in detected_objects:
            if class_id == 9:

                left, top = box[0], box[1]
                w, h = box[2], box[3]
                right, bottom = left + w, top + h
                image = image[ top:bottom,left:right]
                # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                cv2.imwrite("img_tet/name_{}".format(name), image)




    def meta_pseudo_label(self, name, image, image_format='BGR', threshold=0.5, nms_threshold=0.5, show=True):
        """Show detected object on image"""

        detected_objects = self.run_inference_for_single_image(image, image_format=image_format, threshold=threshold,
                                                               nms_threshold=nms_threshold)

        class_names = self.load_class_names()
        height, width, channels = image.shape
        size = [width, height]
        shift = 40  # space for annotation
        img = np.zeros((height + shift, width, channels), dtype=np.uint8)  # this is where drawing happen
        img[shift:] = image.copy()
        label_values = []
        for box, score, class_id in detected_objects:
            left, top = box[0], box[1]
            w, h = box[2], box[3]
            right, bottom = left + w, top + h
            box = [left, right, top, bottom]
            x,y,w,h = convert(size, box)
            label_values.append('{class_id} {x} {y} {w} {h}'.format(class_id = class_id, x = x, y = y, w = w, h = h))
            # mid = top + h//2
            # if mid < (height//3):
            #     print("its failed champion")
            text = '{} {:.4f}'.format(class_names[class_id], score)
            img = cv2.rectangle(img, (left, top + shift), (right, bottom + shift), (0, 255, 0), 2)
            img = cv2.putText(img, text, (left, top + shift), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        print(label_values)
        sorted_lines = list(sorted(label_values, key=lambda x: int(x.split()[0])))
        with open('watermark/{}.txt'.format(name), 'w') as f:
            for line in sorted_lines:
                f.write(line)
                f.write('\n')
        # get_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # get_img.show()




def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h



def parse_cfg(cfgfile):
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks


def find_block(blocks, block_type):
    for x in blocks:
        if x['type'] == block_type:
            return x
    return None




test_single_img = ImageClassifier()

list_name = [i for i in os.listdir("watermark") if i.endswith(".jpg")]
# list_name = ['watermark_00.JPG', 'watermark_01.JPG', 'watermark_02.JPG']
# list_name = ['Watermark (143).jpg']
# list_name = ['gameover.JPG','gameover1.JPG', 'gameover2.JPG']

for name in list_name:
    if name is not None:
        print(name)
        # path = name
        path = 'watermark/' + name
        img = cv2.imread(path)
        new_img = test_single_img.meta_pseudo_label(name.split(".")[0], img, image_format='BGR', threshold=0.6, nms_threshold=0.5, show=True)
        # new_img = test_single_img.get_champion(name, img, image_format='BGR', threshold=0.6, nms_threshold=0.5, show=True)
