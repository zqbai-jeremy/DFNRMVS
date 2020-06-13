import os, datetime, warnings
from torch.utils.data import dataloader
from tqdm import tqdm
import core_dl.module_util as dl_util
from core_dl.train_params import TrainParameters
from core_dl.logger import Logger
from colorama import Back, Style

class BaseTrainBox:

    # verbose mode
    verbose_mode = False

    # cuda device id
    dev_id = 0

    # training optimizer
    optimizer = None

    # network model instance
    model = None

    # training logger
    logger = None

    # loss function
    criterion = None

    # training parameters
    train_params = TrainParameters()

    def __init__(self, train_params: TrainParameters, workspace_dir=None, checkpoint_path=None, comment_msg=None, load_optimizer=True):

        self.verbose_mode = train_params.VERBOSE_MODE
        self.train_params = train_params
        self.load_optimizer = load_optimizer
        self.checkpoint_path = checkpoint_path

        # set workspace for temp dataset, checkpoints etc.
        self.workspace_dir = workspace_dir
        if workspace_dir is not None and not os.path.exists(workspace_dir):
            os.mkdir(workspace_dir)

        # set the device to run training process
        self._set_dev_id(train_params.DEV_ID)

        # load Checkpoints if needed
        self.pre_checkpoint = None if (checkpoint_path is None or not os.path.exists(checkpoint_path)) \
            else dl_util.load_checkpoints(checkpoint_path)

        # set network
        self._set_network()
        if self.model is not None and self.pre_checkpoint is not None and 'net_instance' in self.pre_checkpoint.keys():
            self.model.load_state(self.pre_checkpoint['net_instance'])
            if self.verbose_mode:
                print('[Init. Network] Load net States from checkpoint: ' + checkpoint_path)

        # set the loss function
        self._set_loss_func()

        # set the optimizer
        self._set_optimizer()
        if self.load_optimizer is True:
            if self.optimizer is not None and self.pre_checkpoint is not None and 'optimizer' in self.pre_checkpoint.keys():
                self.optimizer.load_state_dict(self.pre_checkpoint['optimizer'])
                if self.verbose_mode:
                    print('[Init. Optimizer] Load Optimizer from checkpoint: ' + checkpoint_path)

        # print comment or tag message
        if self.verbose_mode and comment_msg is not None:
            print('[Tag] -------------------------------------------------------------------------------------------')
            print(comment_msg)

        # set the logger
        self._set_logger(workspace_dir, comment_msg)
        if self.verbose_mode and self.logger is not None:
            self.logger.meta_dict['dev_id'] = train_params.DEV_ID
            self.logger.meta_dict['start_learning_rate'] = train_params.START_LR
            # print('[Logger] Meta Info:')
            # self.logger.print_meta_info()
            print('[Logger] Dir: %s' % self.logger.log_base_dir)

        # save net definition
        self._save_net_def()

        # report the training init
        self.report()

        self.train_start_time = -1

    def _save_net_def(self):
        # save the net definitions
        self.model_def_dir = None
        if self.model is not None and self.logger is not None:
            self.model_def_dir = os.path.join(self.logger.log_base_dir, 'net_def')
            if not os.path.exists(self.model_def_dir):
                os.mkdir(self.model_def_dir)
            self.model.save_net_def(self.model_def_dir)
            self.train_params.save(os.path.join(self.model_def_dir, 'train_param.json'))

    def _set_dev_id(self, id: int):
        self.dev_id = id
        # torch.cuda.set_device(id)

    def _set_logger(self, workspace_dir, comment_msg):
        if workspace_dir is not None:
            log_base_dir = os.path.join(workspace_dir, 'logs')

            # setup the logger if dir has provided and add default keys
            if self.checkpoint_path is not None and self.train_params.LOG_CONTINUE_STEP > 0:
                self.logger = Logger(base_dir=self.train_params.LOG_CONTINUE_DIR,
                                     log_types='tensorboard',
                                     tag=comment_msg,
                                     description=self.train_params.DESCRIPTION,
                                     ckpt_path=self.checkpoint_path,
                                     continue_from_step=self.train_params.LOG_CONTINUE_STEP)
            else:
                self.logger = Logger(base_dir=log_base_dir, log_types='tensorboard', tag=comment_msg)
            self.logger.add_keys(['Epoch', 'Iteration', 'net', 'Event'])

            # prepare save model dir
            self.model_checkpoint_dir = os.path.join(self.logger.log_base_dir, 'checkpoints')
            if not os.path.exists(self.model_checkpoint_dir):
                os.mkdir(self.model_checkpoint_dir)
        else:
            self.logger = None

    def _set_network(self):
        pass

    def _set_optimizer(self):
        pass

    def _set_loss_func(self):
        pass

    def _add_log_keys(self, keys):
        if self.logger is not None:
            self.logger.add_keys(keys)

    """ Training Routines ----------------------------------------------------------------------------------------------
    """
    def _prepare_train(self):
        pass

    def _optimizer_update(self):
        pass

    def _train_feed(self, train_sample, cur_train_epoch, cur_train_itr) -> dict or None:
        """
        Train the model by feeding one sample
        :param train_sample: single sample that will be feed in network for training
        :param cur_train_epoch: the current training epoch
        :param cur_train_itr: the current training iteration
        :return: recorded dict for logger
        """
        return None

    def _valid_loop(self, valid_loader: dataloader, cur_train_epoch, cur_train_itr) -> dict or None:
        """
        Validate the training process by providing multiple validating samples.
        :param valid_loader: subset of validation set.
        :param cur_train_epoch: the current training epoch
        :param cur_train_itr: the current training iteration
        :return: recorded dict for logger
        """
        if self.model is not None:
            self.model.eval()
        return None

    def train_loop(self, train_loader, valid_loader=None):

        # prepare the training process (e.g. adding more dict keys)
        self._prepare_train()

        epoch, itr = 0, 0
        self.train_start_time = datetime.datetime.now()
        print('[Running] -----------------------------------------------------------------------------------------')

        try:
            for epoch in range(0, self.train_params.MAX_EPOCHS):
                progress = tqdm(total=len(train_loader), ncols=100, leave=False)

                for train_batch_idx, train_sample in enumerate(train_loader):

                    progress.update(1)
                    progress.set_description('[Train] epoch = %d, lr=%f' % (epoch,
                                                                          dl_util.get_learning_rate(self.optimizer)))

                    # prepare feeding the samples
                    self.model.train()
                    self.optimizer.zero_grad()

                    # update optimizer
                    self._optimizer_update()

                    # forward and backward
                    log_dict = self._train_feed(train_sample, epoch, itr)

                    # optimize the parameters
                    self.optimizer.step()

                    # log the training information
                    if log_dict is not None:
                        log_dict['Iteration'] = itr + 1
                        log_dict['Epoch'] = epoch
                        log_dict['Event'] = 'Training'
                        self.logger.log(log_dict)

                    # save the training checkpoints every 'checkpoint_steps'
                    if self.check_checkpoint_step(itr):
                        self.save_checkpoint(epoch, itr)

                    # do validation
                    if self.check_valid_step(itr) and valid_loader is not None:
                        progress.set_description('[Valid]')

                        valid_log_dict = self._valid_loop(valid_loader, epoch, itr)

                        # log the validation
                        if valid_log_dict is not None:
                            valid_log_dict['Iteration'] = itr + 1
                            valid_log_dict['Epoch'] = epoch
                            valid_log_dict['Event'] = 'Validating'
                            self.logger.log(valid_log_dict)
                    itr += 1

                # save the checkpoint
                self.save_checkpoint(epoch, itr)
                progress.close()

        except Exception as e:
            import traceback
            print(traceback.format_exc())

            print('[Exception]: ' + str(e))
            self.save_checkpoint(epoch, itr)

    def check_log_step(self, itr):
        return (itr + 1) % self.train_params.LOG_STEPS == 0

    def check_checkpoint_step(self, itr):
        return self.train_params.CHECKPOINT_STEPS > 0 and (itr + 1) % self.train_params.CHECKPOINT_STEPS == 0

    def check_valid_step(self, itr):
        return self.train_params.VALID_STEPS > 0 and (itr + 1) % self.train_params.VALID_STEPS == 0

    def check_visualization_step(self, itr):
        return self.train_params.VIS_STEPS > 0 and (itr+1) % self.train_params.VIS_STEPS == 0

    def save_checkpoint(self, epoch, itr):
        if self.logger is not None:
            checkpoint_file_name = os.path.join(self.model_checkpoint_dir, 'iter_%06d.pth.tar' % (itr + 1))
            if self.logger is not None:
                self.logger.log({
                    'Iteration': itr + 1,
                    'Epoch': epoch,
                    'Event': "Check point saved to %s" % checkpoint_file_name
                })

            dl_util.save_checkpoint({
                'log_dir': self.logger.log_base_dir,
                'iteration': itr + 1,
                'epoch': epoch + 1,
                'net_instance': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, filename=checkpoint_file_name, is_best=False)

            if self.verbose_mode:
                print('[Checkpoints] Save checkpoint to ' + checkpoint_file_name)

        self.logger.save_meta_info(add_log_dict={'history': 'Save checkpoint to %s' % checkpoint_file_name})

    def report(self):

        self.print_protected_god_animal()

        print("[Training Parameters Overview] ------------------------------------------------------------------------")
        self.train_params.report()

        print("[Optimizer Overview] ----------------------------------------------------------------------------------")
        if self.optimizer is not None:
            print("[%s] Start learning rate: %f" % (type(self.optimizer), dl_util.get_learning_rate(self.optimizer)))

    @staticmethod
    def print_protected_god_animal():
        print(Back.GREEN + " ## ┏━━ FORGIVE HAT ━━┓##" + Style.RESET_ALL)
        god = " #    ┏┓     ┏┓  \n" \
              " #   ┏┛┻━━━━━┛┻┓ \n" \
              " #   ┃         ┃ \n" \
              " #   ┃    ━    ┃ \n" \
              " #   ┃ ┳┛   ┗┳ ┃ \n" \
              " #   ┃         ┃ \n" \
              " #   ┃    ┻    ┃ \n" \
              " #   ┃         ┃ \n" \
              " #   ┗━┓     ┏━┛ \n" \
              " #     ┃     ┃   \n" \
              " #     ┃     ┃   \n" \
              " #     ┃     ┗━━━┓  \n" \
              " #     ┃         ┣┓ \n" \
              " #     ┃         ┏┛ \n" \
              " #     ┗┓┓┏━━┳┓┏━┛  \n" \
              " #      ┃┫┫  ┃┫┫    \n" \
              " #      ┗┻┛  ┗┻┛    \n";
        print(god)
        print(Back.GREEN + " ## This code is far away from bug with the animal protecting ##" + Style.RESET_ALL)
