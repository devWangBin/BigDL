#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import logging
from bigdl.orca.learn.pytorch.training_operator import TrainingOperator


class Estimator(object):
    @staticmethod
    def from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   metrics=None,
                   backend="spark",
                   config=None,
                   workers_per_node=1,
                   scheduler_creator=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   model_dir=None,
                   sync_stats=False,
                   log_level=logging.INFO,
                   log_to_driver=True,
                   training_operator_cls=TrainingOperator,
                   ):
        """
        Create an Estimator for PyTorch.

        :param model: A model creator function that takes the parameter "config" and returns a
               PyTorch model.
        :param optimizer: An optimizer creator function that has two parameters "model" and
               "config" and returns a PyTorch optimizer.
               Default: None if training is not performed.
        :param loss: An instance of PyTorch loss.
               Default: None if loss computation is not needed.
        :param metrics: One or a list of Orca validation metrics. Function(s) that computes the
               metrics between the output and target tensors are also supported.
               Default: None if no validation is involved.
        :param backend: The distributed backend for the Estimator. One of "spark",  "ray",
               "bigdl" or "horovod".
               Default: "spark".
        :param config: A parameter config dict, CfgNode or any class instance that plays a role of
               configuration to create model, loss, optimizer, scheduler and data.
               Default: None if no config is needed.
        :param workers_per_node: The number of PyTorch workers on each node.
               Default: 1.
        :param scheduler_creator: A scheduler creator function that has two parameters "optimizer"
               and "config" and returns a PyTorch learning rate scheduler wrapping the optimizer.
               Note that if you specify this parameter, you need to take care of the argument
               scheduler_step_freq accordingly as well.
               Default: None if no scheduler is needed.
        :param scheduler_step_freq: The frequency when `scheduler.step` is called.
               "batch" or "epoch" if there is a scheduler.
               Default: "batch".
        :param use_tqdm: Whether to use tqdm to monitor the training progress.
               Default: False.
        :param model_dir: The path to save the PyTorch model during the training if
               checkpoint_trigger is defined and triggered.
               Default: None.
        :param sync_stats: Whether to sync metrics across all distributed workers after each epoch.
               If set to False, only the metrics of the worker with rank 0 are printed.
               Default: True
        :param log_level: The log_level of each distributed worker.
               Default: logging.INFO.
        :param log_to_driver: Whether to display executor log on driver in cluster mode for spark
               backend. Default: True.

        :return: A Estimator object for PyTorch.
        """
        if backend in {"horovod", "ray"}:
            from bigdl.orca.learn.pytorch.pytorch_ray_estimator import PyTorchRayEstimator
            return PyTorchRayEstimator(model_creator=model,
                                       optimizer_creator=optimizer,
                                       loss_creator=loss,
                                       metrics=metrics,
                                       scheduler_creator=scheduler_creator,
                                       training_operator_cls=training_operator_cls,
                                       config=config,
                                       scheduler_step_freq=scheduler_step_freq,
                                       use_tqdm=use_tqdm,
                                       workers_per_node=workers_per_node,
                                       backend=backend,
                                       sync_stats=sync_stats,
                                       log_level=log_level)
        elif backend == "bigdl":
            from bigdl.orca.learn.pytorch.pytorch_spark_estimator import PyTorchSparkEstimator
            return PyTorchSparkEstimator(model=model,
                                         loss=loss,
                                         optimizer=optimizer,
                                         config=config,
                                         metrics=metrics,
                                         model_dir=model_dir,
                                         bigdl_type="float")
        elif backend == "spark":
            from bigdl.orca.learn.pytorch.pytorch_pyspark_estimator import PyTorchPySparkEstimator
            return PyTorchPySparkEstimator(model_creator=model,
                                           optimizer_creator=optimizer,
                                           loss_creator=loss,
                                           metrics=metrics,
                                           scheduler_creator=scheduler_creator,
                                           training_operator_cls=training_operator_cls,
                                           config=config,
                                           scheduler_step_freq=scheduler_step_freq,
                                           use_tqdm=use_tqdm,
                                           workers_per_node=workers_per_node,
                                           sync_stats=sync_stats,
                                           log_level=log_level,
                                           model_dir=model_dir,
                                           log_to_driver=log_to_driver,
                                           )
        else:
            from bigdl.dllib.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "Only horovod, ray, bigdl and spark backends are "
                              f"supported for now, got backend: {backend}")

    @staticmethod
    def latest_checkpoint(checkpoint_dir):
        from .callbacks.model_checkpoint import ModelCheckpoint
        checkpoint_path = ModelCheckpoint.get_latest_checkpoint(checkpoint_dir)
        return checkpoint_path
