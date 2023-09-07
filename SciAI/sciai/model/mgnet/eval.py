"""mgnet eval"""
import mindspore as ms
from mindspore import nn, Model

from sciai.context import init_project
from sciai.utils import print_log
from sciai.utils.python_utils import print_time

from src.network import MgNet
from src.process import load_data, prepare


@print_time("eval")
def main(args):
    print_log(args)
    _, test_set, num_classes = load_data(args.load_data_path, args.batch_size, args.dataset)
    net = MgNet(args, ms.float32, num_classes=num_classes)
    ms.load_checkpoint(args.load_ckpt_path, net)
    criterion = nn.CrossEntropyLoss()
    model = Model(network=net, loss_fn=criterion, metrics={'accuracy'})
    test_accuracy = model.eval(test_set)
    print_log("test accuracy:{}".format(test_accuracy))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
