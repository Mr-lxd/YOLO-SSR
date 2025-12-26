# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolov8n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ yolo mode=predict model=yolov8n.pt                 # PyTorch
                              yolov8n.torchscript        # TorchScript
                              yolov8n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolov8n_openvino_model     # OpenVINO
                              yolov8n.engine             # TensorRT
                              yolov8n.mlmodel            # CoreML (macOS-only)
                              yolov8n_saved_model        # TensorFlow SavedModel
                              yolov8n.pb                 # TensorFlow GraphDef
                              yolov8n.tflite             # TensorFlow Lite
                              yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov8n_paddle_model       # PaddlePaddle
"""
import platform
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn  # ##### LXD ##### Added for nn.Sigmoid

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data import load_inference_source
from ultralytics.yolo.data.augment import LetterBox, classify_transforms
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, SETTINGS, callbacks, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz, check_imshow
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.torch_utils import select_device, smart_inference_mode

# ##### LXD START: Correct import for Results class #####
from ultralytics.yolo.engine.results import Results, Boxes, Masks # Assuming Boxes and Masks are also needed and in the same module
# If Boxes or Masks are elsewhere, import them accordingly
# ##### LXD END: Correct import for Results class #####


STREAM_WARNING = """
    WARNING ⚠️ stream/video/webcam/dir predict source will accumulate results in RAM unless `stream=True` is passed,
    causing potential out-of-memory errors for large sources or long-running streams/videos.

    Usage:
        results = model(source=..., stream=True)  # generator of Results objects
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs
            masks = r.masks  # Masks object for segment masks outputs
            probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """
    BasePredictor

    A base class for creating predictors.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_setup (bool): Whether the predictor has finished setup.
        model (nn.Module): Model used for prediction.
        data (dict): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_path (str): Path to video file.
        vid_writer (cv2.VideoWriter): Video writer for saving video output.
        annotator (Annotator): Annotator used for prediction.
        data_path (str): Path to data.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initializes the BasePredictor class.

        Args:
            cfg (str, optional): Path to a configuration file. Defaults to DEFAULT_CFG.
            overrides (dict, optional): Configuration overrides. Defaults to None.
        """

        self.args = get_cfg(cfg, overrides)
        project = self.args.project or Path(SETTINGS['runs_dir']) / self.args.task
        name = self.args.name or f'{self.args.mode}'
        self.save_dir = increment_path(Path(project) / name, exist_ok=self.args.exist_ok)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_path, self.vid_writer = None, None
        self.plotted_img = None  # ##### LXD ##### This will store a list: [detection_plot, seg_mask1, seg_mask2,...]
        self.data_path = None
        self.source_type = None
        self.batch = None
        self.sigmoid = nn.Sigmoid()
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.
        """
        if not isinstance(im, torch.Tensor):
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)
        # NOTE: assuming im with (b, 3, h, w) if it's a tensor
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def pre_transform(self, im):
        """Pre-tranform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        same_shapes = all(x.shape == im[0].shape for x in im)
        auto = same_shapes and self.model.pt  # ##### LXD ##### For ONNX, auto might be False
        # ##### LXD ##### Ensure LetterBox uses correct imgsz and stride for ONNX
        letterbox_transform = LetterBox(self.imgsz, auto=auto, stride=self.model.stride)
        return [letterbox_transform(image=x) for x in im]

    def write_results(self, idx, results_list, batch):
        """Write inference results to a file or directory."""
        p, im, _ = batch  # im here is preprocessed image tensor
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')

        ##### LXD ##### Use original image shape for logging if available, else preprocessed
        orig_im0 = self.batch[1][idx]  # Original image for this specific index
        log_string += '%gx%g ' % orig_im0.shape[:2]  # print string (original HxW)

        # ##### LXD ##### `results_list` now contains processed results for each task
        # For detection (first element of results_list if it's the detection result object)
        # For segmentation (subsequent elements if they are processed masks)
        plotted_detection_img = None  # This will hold the image with detection boxes

        for i, result_item in enumerate(results_list):
            if hasattr(result_item, 'verbose'):  # Check if it's a Results object (likely detection)
                try:
                    log_string += result_item.verbose()
                    if self.args.save or self.args.show:
                        plot_args = dict(line_width=self.args.line_width,
                                         boxes=self.args.boxes,
                                         conf=self.args.show_conf,
                                         labels=self.args.show_labels)
                        # ##### LXD ##### Use the original image for plotting detections
                        # im_to_plot_on = self.batch[1][idx].copy() # Get the original image for this specific item in batch
                        # plotted_detection_img = result_item.plot(img=im_to_plot_on, **plot_args)

                        # ##### LXD ##### Plot on a copy of the original image for this specific index
                        # The `result.plot` method itself uses the `orig_img` from the Results object.
                        # Ensure `orig_img` in the detection `Results` object is correct.
                        plotted_detection_img = result_item.plot(**plot_args)

                except Exception as e:
                    LOGGER.warning(f"Error in verbose/plot for detection result: {e}")
                    pass  # Continue if error
            # Segmentation results (masks) are handled separately in save_preds

        # ##### LXD ##### Store the plotted detection image and raw segmentation masks
        # self.plotted_img will be a list: [plotted_detection_image, seg_mask_tensor_1, seg_mask_tensor_2, ...]
        self.plotted_img = []
        if plotted_detection_img is not None:
            self.plotted_img.append(plotted_detection_img)
        else:
            # If no detections or error, use a copy of the original image as a base for segmentation overlay
            self.plotted_img.append(self.batch[1][idx].copy())

        return log_string

    def postprocess(self, preds, img, orig_img):
        """Post-processes predictions for an image and returns them."""
        # For multi-task, specific postprocess_det and postprocess_seg are used.
        return self.postprocess_det(preds, img, orig_img)  # Default to detection

    # ##### LXD ##### Add a detection-specific postprocessor if not inherited
    def postprocess_det(self, preds, img, orig_imgs):
        """Postprocesses detection predictions."""
        # preds from model output (either PT or ONNX)
        # For PT, it might be (predictions, [optional_other_stuff])
        # For ONNX, it's a raw tensor from detection head.
        # This function should convert raw preds to a list of Results objects.

        #####LXD##### for debug
        # Handle if PT model's detection_pred_source is a tuple
        # if self.model.pt and isinstance(preds, tuple) and len(preds) > 0:
        #     actual_preds_for_nms = preds[0]  # Assume first element of tuple is the prediction tensor
            # LOGGER.info(
            #     f"PT postprocess_det: Extracted NMS tensor from tuple. Shape: {actual_preds_for_nms.shape if hasattr(actual_preds_for_nms, 'shape') else 'N/A'}")
        #####LXD#####

        # If preds is already a list of Results objects (from PT model), return it
        if isinstance(preds, list) and all(hasattr(p, 'boxes') for p in preds):  # crude check for Results obj
            return preds
        if hasattr(preds, 'boxes'):  # single Results object
            return [preds]

        p = ops.non_max_suppression(
            preds[0] if isinstance(preds, list) and not isinstance(preds[0], Results) else preds,
            # Handle list from ONNX, but not if already Results
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            nc=(len(self.model.names) if self.model.names else 0)
            )

        results_list_obj = []  # ##### LXD ##### Renamed to avoid conflict with imported Results
        for i, pred_item in enumerate(p):  # ##### LXD ##### Renamed pred to pred_item
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            current_path = self.batch[0][i] if self.batch and isinstance(self.batch[0], list) and i < len(
                self.batch[0]) else "image.jpg"

            # Ensure orig_img is a NumPy array for Results constructor if it expects that, or tensor if it expects tensor.
            if isinstance(orig_img, torch.Tensor):
                # Convert to HWC NumPy BGR
                orig_img_np = orig_img.permute(1, 2, 0).cpu().numpy() * 255
                orig_img_np = orig_img_np.astype(np.uint8)
                if orig_img_np.shape[2] == 3:  # RGB to BGR if needed by OpenCV based Results plotting
                    orig_img_np = cv2.cvtColor(orig_img_np, cv2.COLOR_RGB2BGR)
            else:
                orig_img_np = orig_img.copy()

            # Boxes tensor for Results object
            # pred_item contains [x1, y1, x2, y2, conf, cls]
            # Scale boxes
            scaled_boxes = ops.scale_boxes(img.shape[2:], pred_item[:, :4], orig_img_np.shape)

            # ##### LXD START: Use corrected Results reference #####
            if pred_item.shape[0] > 0:  # If there are detections

                # Create a Boxes object
                # boxes_data = torch.cat((scaled_boxes, pred_item[:, 4:6]), dim=1) # xyxy, conf, cls
                # boxes_obj = Boxes(boxes_data, orig_img_np.shape[:2]) # Pass original image shape

                # Alternative: Pass components to Results if it can construct Boxes internally
                current_result = Results(orig_img=orig_img_np,
                                         path=current_path,
                                         names=self.model.names,
                                         boxes=pred_item)  # Pass raw NMS output tensor [N, 6]
            else:  # No detections
                current_result = Results(orig_img=orig_img_np,
                                         path=current_path,
                                         names=self.model.names,
                                         boxes=None)  # Pass None for boxes
            results_list_obj.append(current_result)
            # ##### LXD END: Use corrected Results reference #####
        return results_list_obj

    def __call__(self, source=None, model=None, stream=False):
        """Performs inference on an image or stream."""
        self.stream = stream
        if stream:
            return self.stream_inference(source, model)
        else:
            # ##### LXD ##### For non-stream, collect all results
            all_results_collected = []
            for r_batch in self.stream_inference(source, model):
                all_results_collected.extend(r_batch)  # r_batch is a list for each item in batch
            return all_results_collected

    def predict_cli(self, source=None, model=None):
        """Method used for CLI prediction. It uses always generator as outputs as not required by CLI mode."""
        gen = self.stream_inference(source, model)
        for _ in gen:  # running CLI inference without accumulating any outputs (do not modify)
            pass

    def setup_source(self, source):
        """Sets up source and inference mode."""
        self.imgsz = check_imgsz(self.args.imgsz, stride=self.model.stride, min_dim=2)  # check image size
        self.transforms = getattr(self.model.model, 'transforms', classify_transforms(
            self.imgsz[0])) if self.args.task == 'classify' else None
        self.dataset = load_inference_source(source=source, imgsz=self.imgsz, vid_stride=self.args.vid_stride)
        self.source_type = self.dataset.source_type
        if not getattr(self, 'stream', True) and (self.dataset.mode == 'stream' or  # streams
                                                  len(self.dataset) > 1000 or  # images
                                                  any(getattr(self.dataset, 'video_flag', [False]))):  # videos
            LOGGER.warning(STREAM_WARNING)
        self.vid_path, self.vid_writer = [None] * self.dataset.bs, [None] * self.dataset.bs

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None):
        """Streams real-time inference on camera feed and saves results to file."""
        if self.args.verbose:
            LOGGER.info('')

        # Setup model
        if not self.model:
            self.setup_model(model)
        # Setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # Check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.batch, profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile())
        self.run_callbacks('on_predict_start')
        for batch_data in self.dataset:  # Renamed batch to batch_data to avoid conflict with self.batch
            self.run_callbacks('on_predict_batch_start')  # ##### LXD ##### Moved callback
            self.batch = batch_data  # Store current batch_data in self.batch
            path, im0s, vid_cap, s = self.batch  # Unpack self.batch

            visualize = increment_path(self.save_dir / Path(path[0]).stem, mkdir=True) if self.args.visualize and (not self.source_type.tensor) else False

            # Preprocess
            with profilers[0]:
                im = self.preprocess(im0s)

            # Inference
            with profilers[1]:
                preds = self.model(im, augment=self.args.augment, visualize=visualize)
                # For ONNX, preds will be a list of numpy arrays.
                # For PT, preds is usually a tuple (det_out, seg_out1, seg_out2, ...) or similar

            # Postprocess
            with profilers[2]:
                # ##### LXD START: Modified postprocessing for multi-task ONNX/PT #####
                # `preds` from ONNX AutoBackend is a list of outputs
                # `preds` from PT model might be a tuple or a list of Results for detection part

                # Initialize results containers for this batch
                batch_det_results = [None] * len(im0s)
                batch_seg_masks = [[] for _ in range(len(im0s))]  # Each item can have multiple seg masks

                if self.args.task == 'multi':
                    # For PT models, the first element of preds tuple is often detection.
                    # The rest are segmentation masks.
                    # For ONNX models, preds is a list of tensors. We need to identify them.

                    detection_pred_source = None
                    segmentation_pred_sources = []

                    if self.model.pt:  # PyTorch model
                        # ##### LXD PT: Handle preds = [detection_tuple, segmentation_tensor] #####
                        if isinstance(preds, list) and len(preds) == 2:
                            detection_pred_source = preds[0]  # This is the detection_tuple
                            segmentation_pred_sources = [preds[1]]  # This is the segmentation_tensor, wrapped in a list
                            # LOGGER.info(
                            #     f"PT: Detected list output. Det source type: {type(detection_pred_source)}, Seg source type: {type(segmentation_pred_sources[0])}")
                        elif isinstance(preds, list) and all(isinstance(item, Results) for item in
                                                             preds):  # Handles if PT model returns list of Results
                            detection_pred_source = preds
                            # Seg masks expected inside Results objects here
                        elif isinstance(preds, Results):  # Handles if PT model returns single Results
                            detection_pred_source = [preds]
                            # Seg masks expected inside Results object here
                        else:
                            LOGGER.warning(f"PT: Unexpected multi-task output format. Type: {type(preds)}")
                    else:  # ONNX or other backends (preds is a list of tensors)
                        # Identify by shape/dimension
                        # Detection output: e.g., (1, 6, 6300) -> ndim = 3
                        # Segmentation output: e.g., (1, 2, 640, 480) -> ndim = 4
                        temp_seg_sources = []
                        for p_idx, p_tensor in enumerate(preds):
                            p_tensor_torch = torch.from_numpy(p_tensor).to(self.device) if isinstance(p_tensor, np.ndarray) else p_tensor
                            if p_tensor_torch.ndim == 4 and p_tensor_torch.shape[2] > 1 and p_tensor_torch.shape[
                                3] > 1:  # Likely segmentation HxW > 1
                                temp_seg_sources.append(p_tensor_torch)
                            elif p_tensor_torch.ndim == 3:  # Likely detection
                                if detection_pred_source is None:
                                    detection_pred_source = p_tensor_torch  # Use the first 3D tensor as detection
                                else:
                                    LOGGER.warning(
                                        f"Multiple 3D tensors in ONNX output, using first as detection: {p_tensor_torch.shape}")
                            else:
                                LOGGER.warning(f"Unknown ONNX output tensor shape: {p_tensor_torch.shape}")

                        # Sort segmentation sources by channel count (heuristic, may need adjustment)
                        # e.g., if one segmentation output has more channels (more classes for one head)
                        segmentation_pred_sources = sorted(temp_seg_sources, key=lambda x: x.shape[1], reverse=True)

                    # Process detection
                    if detection_pred_source is not None:
                        # self.postprocess_det expects raw preds, preprocessed_img, list_of_orig_imgs
                        if isinstance(detection_pred_source, list) and all(
                                isinstance(item, Results) for item in detection_pred_source):
                            processed_det_batch = detection_pred_source
                        elif isinstance(detection_pred_source, Results):  # Single item batch
                            processed_det_batch = [detection_pred_source]
                        else:  # Raw tensor from ONNX or PT that needs NMS etc.
                            processed_det_batch = self.postprocess_det(detection_pred_source, im, im0s)

                        for i in range(len(im0s)):
                            if i < len(processed_det_batch):
                                batch_det_results[i] = processed_det_batch[i]
                    else:  # No detection output identified or processed
                        for i in range(len(im0s)):  # Create empty Results if no detection
                            batch_det_results[i] = Results(orig_img=im0s[i], path=path[i], names=self.model.names,
                                                               boxes=None)

                    # Process segmentation
                    for seg_pred_source in segmentation_pred_sources:
                        # self.postprocess_seg expects raw seg_preds [B, C, H, W]
                        # It should return a list of [H, W] masks for the batch
                        processed_seg_batch_masks = self.postprocess_seg(seg_pred_source)  # Returns list of masks for batch
                        for i in range(len(im0s)):
                            if i < len(processed_seg_batch_masks):
                                batch_seg_masks[i].append(processed_seg_batch_masks[i])

                else:  # Single task (e.g., detection only)
                    processed_batch_results = self.postprocess(preds, im, im0s)
                    for i in range(len(im0s)):
                        if i < len(processed_batch_results):
                            batch_det_results[i] = processed_batch_results[i]
                        else:  # Create empty Results if no detection
                            batch_det_results[i] = Results(orig_img=im0s[i], path=path[i], names=self.model.names,
                                                               boxes=None)

                # ##### LXD END: Modified postprocessing #####
                # self.results will be a list of lists, one for each image in the batch.
                # Each inner list: [DetectionResultsObject, seg_mask_tensor1, seg_mask_tensor2, ...]
                self.results_per_item_in_batch = []
                for i in range(len(im0s)):
                    current_item_results = [batch_det_results[i]] + batch_seg_masks[i]
                    self.results_per_item_in_batch.append(current_item_results)

            self.run_callbacks('on_predict_postprocess_end')  # ##### LXD ##### Moved callback

            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):  # Iterate through items in the batch
                current_item_results_list = self.results_per_item_in_batch[i]
                # current_item_results_list[0] is detection Results object for item i
                # current_item_results_list[1:] are segmentation masks for item i

                # Speed calculation needs to be per batch, not per item's result object
                # speed_data = {
                #     'preprocess': profilers[0].dt * 1E3 / n,
                #     'inference': profilers[1].dt * 1E3 / n,
                #     'postprocess': profilers[2].dt * 1E3 / n}
                # if hasattr(current_item_results_list[0], 'speed'):
                #    current_item_results_list[0].speed = speed_data

                if self.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue

                p_item, im0_item = Path(path[i]), im0s[i].copy()  # Use _item suffix for clarity

                # ##### LXD ##### Pass current_item_results_list to write_results
                # write_results is responsible for logging and preparing self.plotted_img

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    # Pass detection results (current_item_results_list[0]) to write_results for logging
                    s += self.write_results(i, [current_item_results_list[0]],
                                            (p_item, im, im0_item))  # Pass only det result for logging/plotting

                # ##### LXD ##### Populate self.plotted_img for save_preds
                # self.plotted_img should be [image_with_detections, seg_mask1_tensor, seg_mask2_tensor, ...]
                # self.plotted_img[0] was set by write_results (it contains the image with detection boxes)
                if self.plotted_img:  # If write_results set the base image
                    for seg_mask_tensor in current_item_results_list[1:]:
                        self.plotted_img.append(seg_mask_tensor)
                else:  # Fallback if self.plotted_img was not set (e.g. not verbose, not save, etc.)
                    self.plotted_img = [im0_item.copy()]  # Start with original image
                    for seg_mask_tensor in current_item_results_list[1:]:
                        self.plotted_img.append(seg_mask_tensor)

                if self.args.show and self.plotted_img is not None and len(self.plotted_img) > 0:
                    # ##### LXD ##### show method needs to handle overlaying masks if not done by save_preds
                    # For now, show just displays the first image in self.plotted_img (detections)
                    self.show(p_item)  # p_item is Path object

                if self.args.save and self.plotted_img is not None and len(self.plotted_img) > 0:
                    # self.plotted_img is [img_with_dets_or_orig, raw_mask1, raw_mask2, ...]
                    self.save_preds(vid_cap, i, str(self.save_dir / p_item.name))

            self.run_callbacks('on_predict_batch_end')  # ##### LXD ##### Moved callback

            yield self.results_per_item_in_batch  # ##### LXD ##### Yield results for current batch

            # 显式断开引用，帮助 GC
            self.results_per_item_in_batch = None
            self.plotted_img = None


            # Print time (inference-only)
            # if self.args.verbose:
            #     LOGGER.info(f'{s}{profilers[1].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):  # Check last video writer
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s_info = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s_info}")

        self.run_callbacks('on_predict_end')  # ##### LXD ##### Moved callback

    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        model_path = model or self.args.model  # ##### LXD ##### Use model_path for clarity
        self.args.half &= device.type != 'cpu'  # half precision only supported on CUDA

        # ##### LXD ##### Pass task to AutoBackend if it's an ONNX model for multi-task hint
        # AutoBackend itself doesn't take a 'task' argument directly in older versions.
        # We rely on the postprocessing logic to handle multi-task based on self.args.task
        self.model = AutoBackend(model_path,
                                 device=device,
                                 dnn=self.args.dnn,
                                 data=self.args.data,  # For PT, helps load names
                                 fp16=self.args.half,
                                 fuse=True,  # Fuse Conv+BN (PT only)
                                 verbose=verbose)
        self.device = device
        self.model.eval()

        # ##### LXD ##### For ONNX, names might not be loaded by AutoBackend; try to get from args if PT fails
        if not self.model.names and self.data:
            try:
                # Attempt to load names from data YAML, useful if model is ONNX but data YAML exists
                from ultralytics.yolo.data.utils import check_det_dataset
                data_dict = check_det_dataset(self.data)
                if 'names' in data_dict:
                    self.model.names = data_dict['names']
                    LOGGER.info(f"Loaded class names for ONNX from data YAML: {self.model.names}")
            except Exception as e:
                LOGGER.warning(f"Could not load names from data YAML for ONNX model: {e}. Using default names.")

        if not self.model.names:  # Fallback if still no names
            self.model.names = {i: f'class_{i}' for i in range(80)}  # Default 80 classes
            if model_path and Path(model_path).suffix == '.pt':  # If it's a PT model, it should have names
                LOGGER.warning("PT model loaded without class names. Check model file.")
            else:  # For ONNX/others without explicit names
                LOGGER.info(f"Using default class names for {Path(model_path).name} as specific names not found.")

    def show(self, p):
        """Display an image in a window using OpenCV imshow()."""
        # ##### LXD ##### self.plotted_img[0] should be the image with detections
        im_to_show = self.plotted_img[0] if self.plotted_img and len(self.plotted_img) > 0 else None
        if im_to_show is None:
            LOGGER.warning(f"No image to show for {p}. self.plotted_img is empty or not set.")
            return

        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im_to_show.shape[1], im_to_show.shape[0])
        cv2.imshow(str(p), im_to_show)
        cv2.waitKey(500 if self.batch[3].startswith('image') else 1)  # 1 millisecond

    def save_preds(self, vid_cap, idx, save_path):
        """Save video predictions as mp4 at specified path."""
        # ##### LXD ##### self.plotted_img is a list: [img_with_dets, mask_tensor1, mask_tensor2, ...]
        if not self.plotted_img or len(self.plotted_img) == 0:
            LOGGER.warning(f"No plotted image or masks to save for {save_path}.")
            return

        base_img_with_dets = self.plotted_img[0].copy()  # Image with detections (or original if no dets)
        segmentation_masks_tensors = self.plotted_img[1:]  # List of [H,W] mask tensors

        # Save imgs
        if self.dataset.mode == 'image':
            im_to_save = base_img_with_dets

            # ##### LXD ##### Overlay segmentation masks
            alpha = 0.5  # transparency factor
            for i, mask_tensor in enumerate(segmentation_masks_tensors):
                if mask_tensor is None or not isinstance(mask_tensor, torch.Tensor): continue

                # Convert single channel mask tensor [H,W] to [H,W,3] color mask
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)  # Ensure uint8 [H,W]

                # Define colors for segmentation classes (example for two seg classes beyond background)
                # Color for mask_tensor where value is 1 (first segmentation class)
                # Color for mask_tensor where value is 2 (second segmentation class)
                if mask_np.max() <= 1:  # Binary mask (0 for bg, 1 for the class)
                    color_for_this_mask_class = np.array([0, 255, 0] if i == 0 else [255, 0, 0],
                                                         dtype=np.uint8)  # Green for first, Red for second
                    # Create a 3-channel color mask
                    color_mask_viz = np.zeros_like(im_to_save, dtype=np.uint8)
                    pixels_to_color = mask_np == 1
                    color_mask_viz[pixels_to_color] = color_for_this_mask_class
                else:  # Multi-class index mask (0, 1, 2...)
                    color_mask_viz = np.zeros_like(im_to_save, dtype=np.uint8)
                    if mask_np.max() >= 1:
                        color_mask_viz[mask_np == 1] = [0, 255, 0]  # Green for class 1
                    if mask_np.max() >= 2:  # Example for a second class
                        color_mask_viz[mask_np == 2] = [255, 0, 0]  # Red for class 2
                    # Add more colors if you have more segmentation classes in one mask output

                # Overlay this color_mask_viz onto im_to_save
                # Find pixels where mask is active (not black background of color_mask_viz)
                active_mask_pixels = np.any(color_mask_viz != [0, 0, 0], axis=-1)
                im_to_save[active_mask_pixels] = (1 - alpha) * im_to_save[active_mask_pixels] + \
                                                 alpha * color_mask_viz[active_mask_pixels]

            cv2.imwrite(save_path, im_to_save)

        else:  # 'video' or 'stream'
            # ##### LXD ##### Similar overlay logic for video frames
            im_to_write = base_img_with_dets  # Start with detection frame
            alpha = 0.5
            for i, mask_tensor in enumerate(segmentation_masks_tensors):
                if mask_tensor is None or not isinstance(mask_tensor, torch.Tensor): continue
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                # Simplified color mapping for video (adjust as above for multi-class if needed)
                color_for_this_mask_class = np.array([0, 255, 0] if i == 0 else [255, 0, 0], dtype=np.uint8)
                color_mask_viz = np.zeros_like(im_to_write, dtype=np.uint8)
                pixels_to_color = mask_np == 1  # Assuming binary mask 0 or 1
                color_mask_viz[pixels_to_color] = color_for_this_mask_class

                active_mask_pixels = np.any(color_mask_viz != [0, 0, 0], axis=-1)
                im_to_write[active_mask_pixels] = (1 - alpha) * im_to_write[active_mask_pixels] + \
                                                  alpha * color_mask_viz[active_mask_pixels]

            if self.vid_path[idx] != save_path:  # new video
                self.vid_path[idx] = save_path
                if isinstance(self.vid_writer[idx], cv2.VideoWriter):
                    self.vid_writer[idx].release()  # release previous video writer
                if vid_cap:  # video
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im_to_write.shape[1], im_to_write.shape[0]
                save_path_mp4 = str(Path(save_path).with_suffix('.mp4'))
                self.vid_writer[idx] = cv2.VideoWriter(save_path_mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            self.vid_writer[idx].write(im_to_write)

    def run_callbacks(self, event: str):
        """Runs all registered callbacks for a specific event."""
        for callback_fn in self.callbacks.get(event, []):  # ##### LXD ##### Renamed `callback` to `callback_fn`
            callback_fn(self)

    def add_callback(self, event: str, func):
        """
        Add callback
        """
        self.callbacks[event].append(func)