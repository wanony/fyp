import torch
from mmengine.runner import Runner
from mmdet.registry import RUNNERS
import copy
from losses.IASS_loss import InstanceAwareSemanticSegmentationLoss
from mmdet.models import CondInst

@RUNNERS.register_module()
class MAMLRunner(Runner):

    def __init__(
            self, 
            model, 
            datasets, 
            work_dir,
            n_shots, 
            outer_lr, 
            inner_lr=0.01, 
            n_inner_steps=5,
            train_dataloader = None,
            val_dataloader = None,
            test_dataloader = None,
            train_cfg = None,
            val_cfg = None,
            test_cfg = None,
            auto_scale_lr = None,
            optim_wrapper = None,
            param_scheduler = None,
            val_evaluator = None,
            test_evaluator = None,
            default_hooks = None,
            custom_hooks = None,
            data_preprocessor = None,
            load_from= None,
            resume= False,
            launcher= 'none',
            env_cfg= dict(dist_cfg=dict(backend='nccl')),
            log_processor = None,
            log_level = 'INFO',
            visualizer = None,
            default_scope = 'mmengine',
            randomness = dict(seed=None),
            experiment_name = None,
            cfg = None, 
            ):
        super().__init__(model=model,
                         work_dir=work_dir
                         )
        self.n_shots = n_shots
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

    def fast_adaptation(self, model, support_set, inner_lr, n_inner_steps):
        # Create a copy of the model for task-specific adaptation
        task_model = copy.deepcopy(model)

        # Move the task model to the same device as the original model
        task_model.to(next(model.parameters()).device)

        # Create a task-specific optimizer for the inner loop
        task_optimizer = torch.optim.SGD(task_model.parameters(), lr=inner_lr)

        # Get the support set images and labels
        support_imgs, support_labels = support_set

        # Perform inner loop optimization
        for _ in range(n_inner_steps):
            # Forward pass
            support_preds = task_model(support_imgs)

            # Calculate the loss
            loss = self.compute_task_loss(support_preds, support_labels)

            # Backward pass
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()

        return task_model

    def compute_loss(self, model, data):
        # Get the images and labels
        imgs, labels = data

        # Forward pass
        preds = model(imgs)

        # Calculate the loss
        loss = self.compute_task_loss(preds, labels)

        return loss

    def compute_task_loss(self, preds, gt_labels, gt_bboxes, gt_masks):
        # The outputs of the CondInst forward pass should be a dictionary containing
        # the relevant outputs, such as 'cls_scores', 'bbox_preds', 'mask_preds', and 'mask_feats'

        cls_scores = preds['cls_scores']
        bbox_preds = preds['bbox_preds']
        mask_preds = preds['mask_preds']
        mask_feats = preds['mask_feats']

        # Calculate the instance segmentation loss (e.g., Mask R-CNN loss)
        # You can use the existing loss functions in the MMdetection library for this purpose
        from mmdet.models.losses import CrossEntropyLoss, SmoothL1Loss
        cls_criterion = CrossEntropyLoss()
        bbox_criterion = SmoothL1Loss()
        cls_loss = cls_criterion(cls_scores, gt_labels)
        bbox_loss = bbox_criterion(bbox_preds, gt_bboxes)

        # Calculate the instance-aware semantic segmentation loss
        # You can use a custom loss function for this purpose
        iass_criterion = InstanceAwareSemanticSegmentationLoss()
        iass_loss = iass_criterion(mask_preds, mask_feats, gt_masks, gt_labels)

        # Combine both losses
        total_loss = cls_loss + bbox_loss + iass_loss

        return total_loss

    def train(self):
        # Prepare the meta-optimizer
        meta_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.outer_lr)

        # Run the MAML training loop
        for epoch in range(self.cfg.total_epochs):
            for i, data_batch in enumerate(self.data_loaders[0]):
                support_set, query_set = data_batch

                # Initialize the meta-gradients
                meta_grads = None

                # Iterate over tasks
                for task_support_set, task_query_set in zip(support_set, query_set):
                    # Update the model on the support set (inner loop)
                    task_model = self.fast_adaptation(self.model, task_support_set, self.inner_lr, self.n_inner_steps)

                    # Calculate the loss on the query set
                    loss = self.compute_loss(task_model, task_query_set)

                    # Calculate the gradients with respect to the model's initial parameters
                    grads = torch.autograd.grad(loss, self.model.parameters())

                    # Accumulate the meta-gradients
                    if meta_grads is None:
                        meta_grads = grads
                    else:
                        meta_grads = [g + mg for g, mg in zip(grads, meta_grads)]

                # Update the model's initial parameters (outer loop)
                meta_optimizer.zero_grad()
                for param, meta_grad in zip(self.model.parameters(), meta_grads):
                    param.grad = meta_grad / len(support_set)
                meta_optimizer.step()

    @classmethod
    def from_cfg(cls, cfg):
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
            datasets=None,
            # added for this class
            n_shots=cfg.get("n_shots", 1),
            outer_lr=cfg.optim_wrapper.optimizer.lr,
            inner_lr=cfg.get("inner_lr", 0.01),
            n_inner_steps=cfg.get("n_inner_steps", 5)
        )
        return runner
