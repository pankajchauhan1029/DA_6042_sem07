from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--earlystop_epoch', type=int, default=15)
        parser.add_argument('--data_aug', action='store_true')
        parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'])
        parser.add_argument('--new_optim', action='store_true')
        parser.add_argument('--loss_freq', type=int, default=100)
        parser.add_argument('--save_latest_freq', type=int, default=2000)
        parser.add_argument('--save_epoch_freq', type=int, default=1)
        parser.add_argument('--continue_train', action='store_true')
        parser.add_argument('--epoch_count', type=int, default=1)
        parser.add_argument('--last_epoch', type=int, default=-1)
        parser.add_argument('--train_split', type=str, default='train')
        parser.add_argument('--val_split', type=str, default='val')
        parser.add_argument('--niter', type=int, default=85)
        parser.add_argument('--beta1', type=float, default=0.9)
        parser.add_argument('--lr', type=float, default=0.001)
        parser.add_argument('--seed', type=int, default=1029)
        parser.add_argument('--fcr_enabled', action='store_true')
        parser.add_argument('--fcr_lambda', type=float, default=0.05)
        parser.add_argument('--patience', type=int, default=12)
        self.isTrain = True
        return parser
