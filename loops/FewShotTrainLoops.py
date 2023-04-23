from mmengine.runner.base_loop import BaseLoop
from mmengine.runner.loops import IterBasedTrainLoop
from mmdet.registry import LOOPS
import torch
from losses.IASS_loss import InstanceAwareSemanticSegmentationLoss
from mmdet.models import CondInst
import torch
from typing import Union, Dict, Optional, List, Tuple, Sequence
import copy


@LOOPS.register_module()
class MAMLIterBasedTrainLoop(IterBasedTrainLoop):
    def __init__(self, *args, inner_lr: float = 0.01, n_inner_steps: int = 5, outer_lr: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.outer_lr = outer_lr

    def fast_adaptation(self, model, support_set, inner_lr, n_inner_steps):
        adapted_model = copy.deepcopy(model)
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
        criterion = torch.nn.CrossEntropyLoss()

        for step in range(n_inner_steps):
            logits = adapted_model(support_set['img'])
            loss = criterion(logits, support_set['gt_label'])
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)

        support_set, query_set = data_batch
        meta_grads = None

        for task_support_set, task_query_set in zip(support_set, query_set):
            task_model = self.fast_adaptation(self.runner.model, task_support_set, self.inner_lr, self.n_inner_steps)

            logits = task_model(task_query_set['img'])
            criterion = torch.nn.CrossEntropyLoss()
            loss = criterion(logits, task_query_set['gt_label'])

            grads = torch.autograd.grad(loss, task_model.parameters(), retain_graph=True)
            if meta_grads is None:
                meta_grads = grads
            else:
                meta_grads = [g + mg for g, mg in zip(grads, meta_grads)]

        meta_optimizer = torch.optim.SGD(self.runner.model.parameters(), lr=self.outer_lr)
        meta_optimizer.zero_grad()
        for param, meta_grad in zip(self.runner.model.parameters(), meta_grads):
            param.grad = meta_grad / len(support_set)
        meta_optimizer.step()

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=None)
        self._iter += 1