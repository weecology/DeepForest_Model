from torchvision.models.detection.retinanet import RetinaNetClassificationHead,RetinaNetRegressionHead, RetinaNet
from torch import nn

class TwoHeadedRetinaNetHead(nn.Module):
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
        return {
            'classification_task1': self.classification_head_task1.compute_loss(targets, head_outputs, matched_idxs),
            'classification_task2': self.classification_head_task2.compute_loss(targets, head_outputs, matched_idxs),            
            'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, anchors, matched_idxs),
        }

    def forward(self, x):
        # type: (List[Tensor]) -> Dict[str, Tensor]
        return {
            'cls_logits': self.classification_head_task1(x),
            'cls_logits_task2': self.classification_head_task2(x),            
            'bbox_regression': self.regression_head(x)
        }
    
def create(trained_model, num_classes_task2=2, freeze_original=True):
    """Sprout a new head from an existing torchvision retinanet model to form a two headed retinanet for multiple classification tasks
    Args:
        trained_model: existing torchvision retinanet model
        num_classes_task2 (int): The number of classes to predict in the new task head
        freeze_original (bool): Whether to freeze the original retinanet layers. Defaults to True
    Returns:
        model: a retinanet model
    """    
    ##Construct a new model
    backbone = trained_model.backbone
    num_anchors = trained_model.anchor_generator.num_anchors_per_location()[0]
    
    head = TwoHeadedRetinaNetHead(
        in_channels=backbone.out_channels,
        num_anchors=num_anchors,
        num_classes_task1=trained_model.head.classification_head.num_classes,
        num_classes_task2=num_classes_task2
    )
            
    model = RetinaNet(backbone=backbone, head=head, num_classes=trained_model.head.classification_head.num_classes)
    
    #Update the weights from the original classification head
    model.head.classification_head_task1.load_state_dict(trained_model.head.classification_head.state_dict()) 

    #Update the weights from the original regression head
    model.head.regression_head.load_state_dict(trained_model.head.regression_head.state_dict()) 
    
    if freeze_original:
        for param in model.head.classification_head_task1.parameters():
            param.requires_grad = False
        
        for param in model.head.regression_head.parameters():
            param.requires_grad = False        
    
    return model