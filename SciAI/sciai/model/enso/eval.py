"""enso eval"""
import mindspore as ms

from sciai.context import init_project
from sciai.utils import data_type_dict_amp
from sciai.utils.python_utils import print_time
from src.network import ENSO
from src.plot import evaluate
from src.process import fetch_dataset_nino34, prepare


@print_time("eval")
def main(args):
    dtype = data_type_dict_amp.get(args.amp_level, ms.float32)
    net = ENSO()
    if dtype == ms.float16:
        net.to_float(ms.float16)
    _, _, ip_var, nino34_var, _, _ = fetch_dataset_nino34(args.load_data_path)
    ms.load_checkpoint(args.load_ckpt_path, net)
    evaluate(args, net, ip_var, nino34_var)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
