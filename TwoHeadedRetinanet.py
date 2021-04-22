"""
Sprout a new classification head on an already trained retinanet from 
https://pytorch.org/vision/stable/_modules/torchvision/models/detection/retinanet.html
"""
import torchvision
import torch
from torchvision.ops import  boxes as box_ops
from torchvision.models.detection.retinanet import *
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNetRegressionHead


class TwoHeadedRetinaNetHead(torch.nn.Module):
    """
    A regression and classification head for use in RetinaNet.

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes_task1 (int): number of classes to be predicted for the first task
        num_classes_task2 (int): number of classes to be predicted for the second task
    """

    def __init__(self, in_channels, num_anchors, num_classes_task1, num_classes_task2):
        super().__init__()
        self.classification_head_task1 = RetinaNetClassificationHead(in_channels, num_anchors, num_classes_task1)
        self.classification_head_task2 = RetinaNetClassificationHead(in_channels, num_anchors, num_classes_task2)
        self.regression_head = RetinaNetRegressionHead(in_channels, num_anchors)

    def compute_loss(self, targets, head_outputs, anchors, matched_idxs):
        # type: (List[Dict[str, Tensor]], Dict[str, Tensor], List[Tensor], List[Tensor]) -> Dict[str, Tensor]
        
        #Set logits and targets for auxillary task
        head_outputs["cls_logits"] = head_outputs["cls_logits_task2"]     
        classification_task2 = self.classification_head_task2.compute_loss(targets, head_outputs, matched_idxs)
        
        #The classification_head loss depends on a dictionary named cls_logit, temporarily rename each
        head_outputs["cls_logits"] = head_outputs["cls_logits_task1"]
        
        #The first task of tree detection always has a label of 0, there is one class and no negatives.
        for index, x in enumerate(targets):
            targets[index]["labels"] = torch.zeros(targets[index]["labels"].shape, dtype=torch.int64)
            
        classification_task1 = self.classification_head_task1.compute_loss(targets, head_outputs, matched_idxs)
        
        return {
            'classification_task1': classification_task1,
            'classification_task2': classification_task2,            
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }


    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits_task1': self.classification_head_task1(x),
            'cls_logits_task2': self.classification_head_task2(x),            
            'bbox_regression': self.regression_head(x)
        }


class TwoHeadedRetinanet(RetinaNet):
    def __init__(self,trained_model=None, num_classes_task2=2, freeze_original=True):
        """Sprout a new head from an existing torchvision retinanet model to form a two headed retinanet for multiple classification tasks
        Args:
            trained_model: existing torchvision retinanet model
            num_classes_task2 (int): The number of classes to predict in the new task head
            freeze_original (bool): Whether to freeze the original retinanet layers. Defaults to True
        Returns:
            model: a retinanet model
        """    
        #Create a pytorch module
        
        ##Construct a new model if no trained backbone
        if not trained_model:
            trained_model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)   
            
        #Init underlying retinanet module, we will overwrite methods below
        super().__init__(trained_model.backbone, trained_model.head.classification_head.num_classes)
        
        self.backbone = trained_model.backbone
        
        num_anchors = trained_model.anchor_generator.num_anchors_per_location()[0]
        
        self.head = TwoHeadedRetinaNetHead(
            in_channels=self.backbone.out_channels,
            num_anchors=num_anchors,
            num_classes_task1=trained_model.head.classification_head.num_classes,
            num_classes_task2=num_classes_task2
        )
                        
        #Update the weights from the original classification head
        self.head.classification_head_task1.load_state_dict(trained_model.head.classification_head.state_dict()) 
    
        #Update the weights from the original regression head
        self.head.regression_head.load_state_dict(trained_model.head.regression_head.state_dict()) 
        
        if freeze_original:
            for param in self.head.classification_head_task1.parameters():
                param.requires_grad = False
            
            for param in self.head.regression_head.parameters():
                param.requires_grad = False        
    
    def get_detections(self, head_outputs, features, images, anchors, original_image_sizes):
        num_anchors_per_level = [x.size(2) * x.size(3) for x in features]
        HW = 0
        for v in num_anchors_per_level:
            HW += v
        HWA = head_outputs['cls_logits_task1'].size(1)
        A = HWA // HW
        num_anchors_per_level = [hw * A for hw in num_anchors_per_level]

        # split outputs per level
        split_head_outputs: Dict[str, List[Tensor]] = {}
        for k in head_outputs:
            split_head_outputs[k] = list(head_outputs[k].split(num_anchors_per_level, dim=1))
        split_anchors = [list(a.split(num_anchors_per_level)) for a in anchors]

        # compute the detections
        detections = self.postprocess_detections(split_head_outputs, split_anchors, images.image_sizes)
        
        #postprocess each task seperately, for convience wrap into seperate tasks list and then reform dict
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            
        return detections
    
    def postprocess_image(self, box_regression_per_image, logits_per_image,logits_per_image_task2, anchors_per_image, image_shape):
        """The primary task class is score thresholded, the secondary task (task2) is indexed based on the filters from the first task"""
        image_boxes = []
        image_labels = []
        image_scores = []                
        image_scores_task2 = []        
        image_labels_task2 = []

        for box_regression_per_level, logits_per_level,logits_per_level_task2, anchors_per_level in \
                zip(box_regression_per_image, logits_per_image,logits_per_image_task2, anchors_per_image):
            num_classes = logits_per_level.shape[-1]

            # remove low scoring boxes
            scores_per_level = torch.sigmoid(logits_per_level).flatten()
            keep_idxs = scores_per_level > self.score_thresh
            scores_per_level = scores_per_level[keep_idxs]
            topk_idxs = torch.where(keep_idxs)[0]

            # keep only topk scoring predictions
            num_topk = min(self.topk_candidates, topk_idxs.size(0))
            scores_per_level, idxs = scores_per_level.topk(num_topk)
            topk_idxs = topk_idxs[idxs]

            anchor_idxs = topk_idxs // num_classes
            labels_per_level = topk_idxs % num_classes
            
            #Repeat for task 2, but use selected box ids from task1, return empty list if no task 1 boxes were selected
            scores_per_level_task2 = torch.sigmoid(logits_per_level_task2[anchor_idxs,:])
            
            if anchor_idxs.shape[0] == 0:
                labels_per_level_task2 = labels_per_level
            else:
                labels_per_level_task2 = torch.argmax(scores_per_level_task2, 1)
            
            boxes_per_level = self.box_coder.decode_single(box_regression_per_level[anchor_idxs],
                                                           anchors_per_level[anchor_idxs])
            boxes_per_level = box_ops.clip_boxes_to_image(boxes_per_level, image_shape)

            image_boxes.append(boxes_per_level)
            image_scores.append(scores_per_level)
            image_labels.append(labels_per_level)
            image_scores_task2.append(scores_per_level_task2)            
            image_labels_task2.append(labels_per_level_task2)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_labels_task2 = torch.cat(image_labels_task2, dim=0)
        image_scores_task2 = torch.cat(image_scores_task2, dim=0)

        # non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[:self.detections_per_img]

        result_dict = {
            'boxes': image_boxes[keep],
            'scores': image_scores[keep],
            'labels': image_labels[keep],
            'scores_task2': image_scores_task2[keep],                        
            'labels_task2': image_labels_task2[keep],
            
        }
        
        return result_dict
    
    def postprocess_detections(self, head_outputs, anchors, image_shapes):
        # type: (Dict[str, List[Tensor]], List[List[Tensor]], List[Tuple[int, int]]) -> List[Dict[str, Tensor]]
        class_logits_task1 = head_outputs['cls_logits_task1']
        class_logits_task2 = head_outputs['cls_logits_task2']
        box_regression = head_outputs['bbox_regression']

        num_images = len(image_shapes)

        detections: List[Dict[str, Tensor]] = []

        for index in range(num_images):
            box_regression_per_image = [br[index] for br in box_regression]
            logits_per_image_task1 = [cl[index] for cl in class_logits_task1]
            logits_per_image_task2 = [cl[index] for cl in class_logits_task2]
    
            anchors_per_image, image_shape = anchors[index], image_shapes[index]
            detections.append(self.postprocess_image(box_regression_per_image, logits_per_image_task1,logits_per_image_task2, anchors_per_image, image_shape))

        return detections
    
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        # get the features from the backbone
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        # TODO: Do we want a list or a dict?
        features = list(features.values())

        # compute the retinanet heads outputs using the features
        head_outputs = self.head(features)

        # create the set of anchors
        anchors = self.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, Tensor]] = []
        if self.training:
            assert targets is not None            
            losses = self.compute_loss(targets, head_outputs, anchors)
        else:
            # recover level sizes
            detections = self.get_detections(head_outputs, features, images, anchors, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RetinaNet always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.eager_outputs(losses, detections)            
        
