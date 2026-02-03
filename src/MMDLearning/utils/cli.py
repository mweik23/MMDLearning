import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser(description='Top tagging')
    parser.add_argument('--exp_name', type=str, default='', metavar='N',
                        help='experiment_name')
    parser.add_argument('--test_mode', action='store_true', default=False,
                        help = 'test best model')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',   
                        help='input batch size for training')
    parser.add_argument('--num_data', type=int, default=-1, metavar='N',
                        help='number of samples')
    parser.add_argument('--epochs', type=int, default=35, metavar='N',
                        help='number of training epochs')
    parser.add_argument('--model_config', type=str, default='', metavar='N',
                        help='model config file')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='number of warm-up epochs')             
    parser.add_argument('--seed', type=int, default=99, metavar='N',
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--mmd_interval', type=int, default=-1, metavar='N',
                        help='how many batches to wait before calculating the null MMD')
    parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                        help='how many epochs to wait before validation')
    parser.add_argument('--datadir', nargs='+', default='data/top', metavar='N',
                        help='data directories')
    parser.add_argument('--logdir', type=str, default='logs/top', metavar='N',
                        help='folder to output logs')
    parser.add_argument('--peak_lr', type=float, default=1e-3, metavar='N',
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=None, metavar='N',
                        help='number of workers for the dataloader')
    parser.add_argument('--patience', type=int, default=10, metavar='N',
                        help='learning rate scheduler')
    parser.add_argument('--threshold', type=float, default=1e-4, metavar='N',
                        help='threshold for lr scheduler to measure new optimum')
    parser.add_argument('--reduce_factor', type=float, default=0.1, metavar='N',
                        help='factor for LR scheduler if reduce')
    parser.add_argument('--start_lr', type=float, default=1e-4, metavar='N',
                        help='starting learning rate factor for warmup')
    parser.add_argument('--MMDturnon_epoch', type=int, default=5, metavar='N',
                        help='epoch when MMD turns on')
    parser.add_argument('--MMD_frac', type=float, default=0, metavar='N',
                        help='prefactor for the MMD loss term')
    parser.add_argument('--MMDturnon_width', type=int, default=5, metavar='N',
                        help='the number of epochs it takes for MMD to smoothly turnon')
    parser.add_argument('--pretrained', type=str, default='', metavar='N',
                        help='directory with model to start the run with')
    parser.add_argument('--ld_optim_state', action='store_true', default=False,
                        help='want to load the optimizer state from pretrained run?')
    parser.add_argument('--n_kernels', type=int, default=5, metavar='N', 
                        help='number of kernels summed for MMD kernel')
    parser.add_argument('--use_tar_labels', action='store_true', default=False,
                        help = 'Use target labels for MMD')
    parser.add_argument('--model_name', type=str, default='LorentzNet', metavar='N',
                        help='model name')
    parser.add_argument('--target_model_groups', nargs='+', default=None, metavar='N',
                        help='use a separate model for target data for these groups (ParticleNet only)')
    parser.add_argument('--mode', type=str, default='qt_classifier', metavar='N',
                        help='mode of operation')
    parser.add_argument('--frozen_groups', type=json.loads, default='{}', metavar='N',
                        help='list of model groups to freeze')
    ############################################################                    
    parser.add_argument('--local_rank', type=int, default=0,
                        help='for compatibility with torch.distributed.launch')
    
    return parser