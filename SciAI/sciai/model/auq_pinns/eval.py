"""auq pinns eval"""
from mindspore import amp
from sciai.context import init_project
from sciai.utils.python_utils import print_time

from src.network import get_all_networks
from src.process import ValDataset, TrainDataset, post_process, prepare


@print_time("eval")
def main(args):
    args.load_ckpt = True
    _, decoder, _, _, _ = get_all_networks(args)
    decoder = amp.auto_mixed_precision(decoder, args.amp_level)
    train_dataset = TrainDataset(args)
    val_dataset = ValDataset(args)
    post_process(args, decoder, train_dataset, val_dataset)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
