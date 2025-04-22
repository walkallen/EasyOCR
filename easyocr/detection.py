import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
from pathlib import Path
import openvino as ov
import onnx

import cv2
import numpy as np
from .craft_utils import getDetBoxes, adjustResultCoordinates
from .imgproc import resize_aspect_ratio, normalizeMeanVariance
from .craft import CRAFT

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def test_net(canvas_size, mag_ratio, net, image, text_threshold, link_threshold, low_text, poly, device, estimate_num_chars=False, use_openvino=False ):
    if isinstance(image, np.ndarray) and len(image.shape) == 4:  # image is batch of np arrays
        image_arrs = image
    else:                                                        # image is single numpy array
        image_arrs = [image]

    img_resized_list = []
    # resize
    for img in image_arrs:
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(img, canvas_size,
                                                                      interpolation=cv2.INTER_LINEAR,
                                                                      mag_ratio=mag_ratio)
        img_resized_list.append(img_resized)
    ratio_h = ratio_w = 1 / target_ratio
    # preprocessing
    x = [np.transpose(normalizeMeanVariance(n_img), (2, 0, 1))
         for n_img in img_resized_list]
    x = torch.from_numpy(np.array(x))

    print(f'suhao 准备推理, test_net 输入形状 x.shape {x.shape}')

    if not use_openvino:
        x = x.to(device)
        # forward pass
        with torch.no_grad():
            y, feature = net(x)
    else:

        print('suhao use openvino')
        core = ov.Core()
        model_ir = core.read_model(model=Path("examples/craft_mlt_25k.xml"))
        compiled_model_ir = core.compile_model(model=model_ir, device_name='CPU')
        input_layer_ir = compiled_model_ir.input(0)
        output_layer_ir = compiled_model_ir.output(0)
        output_layer_ir_feature = compiled_model_ir.output(1)


        result_output = compiled_model_ir([x])



        y_ir = result_output[output_layer_ir]
        feature_ir = result_output[output_layer_ir_feature]


        y = torch.from_numpy(y_ir)

    print(f'suhao y.shape is {y.shape}')

    boxes_list, polys_list = [], []
    for out in y:
        # make score and link map
        score_text = out[:, :, 0].cpu().data.numpy()
        score_link = out[:, :, 1].cpu().data.numpy()

        # Post-processing
        boxes, polys, mapper = getDetBoxes(
            score_text, score_link, text_threshold, link_threshold, low_text, poly, estimate_num_chars)

        # coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = adjustResultCoordinates(polys, ratio_w, ratio_h)
        if estimate_num_chars:
            boxes = list(boxes)
            polys = list(polys)
        for k in range(len(polys)):
            if estimate_num_chars:
                boxes[k] = (boxes[k], mapper[k])
            if polys[k] is None:
                polys[k] = boxes[k]
        boxes_list.append(boxes)
        polys_list.append(polys)

    return boxes_list, polys_list

def get_detector(trained_model, device='cpu', quantize=True, cudnn_benchmark=False, use_openvino=False):
    print(f'suhao get_detector {trained_model}')
    onnx_name = 'examples/craft_mlt_25k.onnx'
    onnx_path = Path(onnx_name)
    ovir_path = onnx_path.with_suffix(".xml")
    if use_openvino:
        if ovir_path.exists():
            
            return None
        else:
            print(f'ovir_path {ovir_path} not found 转换模型')
            if not ovir_path.exists():
                net = CRAFT()
                net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device, weights_only=False)))

                # 示例 3：使用字典指定动态形状 (如果输入有明确名称)
                # input_shape_dict = {"input.1": [-1, 3, -1, -1]}

                dummy_input = torch.randn(1, 3, 128, 640)
                input_shape = [("0", [-1, 3, -1, -1])]

                try:
                    ov_model_dynamic = ov.convert_model(net, example_input=dummy_input, input=input_shape)
                    print("PyTorch model with dynamic shape converted to OpenVINO.")
                    ov.save_model(ov_model_dynamic, ovir_path )
                    print("Dynamic OpenVINO model saved.")
                except Exception as e:
                    print(f"Error during dynamic conversion: {e}")

                # ov_model = ov.convert_model(net, input=input_shape_dict)
                # ov.save_model(ov_model, ovir_path)
            return None


    net = CRAFT()

    if device == 'cpu':
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device, weights_only=False)))
        if quantize:
            try:
                torch.quantization.quantize_dynamic(net, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        net.load_state_dict(copyStateDict(torch.load(trained_model, map_location=device, weights_only=False)))
        if torch.xpu.is_available():
            net = net.to(device)
        else:
            net = torch.nn.DataParallel(net).to(device)
        cudnn.benchmark = cudnn_benchmark

    net.eval()
    return net

def get_textbox(detector, image, canvas_size, mag_ratio, text_threshold, link_threshold, low_text, poly, device, use_openvino=False, optimal_num_chars=None, **kwargs):
    result = []
    estimate_num_chars = optimal_num_chars is not None
    bboxes_list, polys_list = test_net(canvas_size, mag_ratio, detector,
                                       image, text_threshold,
                                       link_threshold, low_text, poly,
                                       device, estimate_num_chars, use_openvino=use_openvino)
    if estimate_num_chars:
        polys_list = [[p for p, _ in sorted(polys, key=lambda x: abs(optimal_num_chars - x[1]))]
                      for polys in polys_list]

    for polys in polys_list:
        single_img_result = []
        for i, box in enumerate(polys):
            poly = np.array(box).astype(np.int32).reshape((-1))
            single_img_result.append(poly)
        result.append(single_img_result)

    return result
