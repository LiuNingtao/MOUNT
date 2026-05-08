
from mmcv.runner.hooks import HOOKS, Hook
from globals import SharedData
import os


@HOOKS.register_module(force=True)
class SaveFlowDictHook(Hook):
    def __init__(self, num_last_epochs=15, save_path='./'):
        self.num_last_epochs = num_last_epochs
        self._restart_dataloader = False
        self.save_path = save_path
    def after_train_epoch(self, runner):
        epoch = runner.epoch
        print('SAVE FLOW TRAIN')
        SharedData.dump_shared_var(os.path.join(self.save_path, f'train_{str(epoch)}_flow.json'))
    # def after_val_epoch(self, runner):
    #     epoch = runner.epoch
    #     print('SAVE FLOW VAL')
    #     SharedData.dump_shared_var(os.path.join(self.save_path, f'val_{str(epoch)}_flow.json'))