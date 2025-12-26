# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import numpy as np
from ultralytics.yolo.engine.predictor_multi import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops


class MultiPredictor(BasePredictor):

    def postprocess_det(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    # def postprocess_seg(self, preds):
    #     """Postprocesses YOLO predictions and returns output detections with proto."""
    #
    #     #####LXD
    #     img_array = self.batch[1][0]
    #     orig_h, orig_w = img_array.shape[:2]# get original resolution
    #     #####
    #
    #     preds = torch.nn.functional.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    #     preds = self.sigmoid(preds)
    #     _, preds = torch.max(preds, 1)
    #     return preds

    def postprocess_seg(self, preds_seg_raw):  # ##### LXD ##### preds_seg_raw is a single segmentation head output [B, C, H, W]
        """Postprocesses YOLO segmentation predictions and returns a list of [H,W] mask tensors for the batch."""

        batch_masks = []
        # ##### LXD ##### preds_seg_raw should be a tensor [B, C, H_pred, W_pred]
        # Ensure it's a torch tensor
        if isinstance(preds_seg_raw, np.ndarray):
            preds_seg_raw = torch.from_numpy(preds_seg_raw).to(self.device)

        # 我们需要的是模型推理的输入尺寸 (如 640)，这存储在 self.imgsz 中。
        if isinstance(self.imgsz, (list, tuple)):
            input_h, input_w = self.imgsz[0], self.imgsz[1]
        else:
            input_h, input_w = self.imgsz, self.imgsz


        # ##### LXD ##### Iterate through batch if preds_seg_raw is batched
        with torch.no_grad():
            for i in range(preds_seg_raw.shape[0]):  # Iterate over batch dimension
                current_pred_seg_item = preds_seg_raw[i].unsqueeze(0)  # Keep batch dim for interpolate: [1, C, H_pred, W_pred]


                # Interpolate to original image size
                interpolated_mask = torch.nn.functional.interpolate(
                    current_pred_seg_item,
                    size=(input_h, input_w),
                    mode='bilinear',
                    align_corners=False
                )

                # Apply sigmoid (if logits) and argmax to get class indices [1, H, W]
                activated_mask = self.sigmoid(interpolated_mask)  # -> [1, C, H, W] probabilities

                # Get class indices by taking argmax over the channel dimension
                # Resulting mask_indices will be [1, H, W]
                _, mask_indices = torch.max(activated_mask, 1)

                # 3. 关键优化：transfer to uint8 and squeeze
                mask_indices = mask_indices.squeeze(0).to(dtype=torch.uint8)

                # to CPU
                mask_cpu = mask_indices.cpu()
                batch_masks.append(mask_cpu) # Squeeze to [H,W] and add to list

                del interpolated_mask, activated_mask, mask_indices, current_pred_seg_item

                # batch_masks.append(mask_indices.squeeze(0))  # Squeeze to [H,W] and add to list

        return batch_masks  # List of [H,W] tensors, one for each image in the batch


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = MultiPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
