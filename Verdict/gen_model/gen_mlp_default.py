#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import sys

sys.path.append("..")
sys.path.append("../nnscaler")
sys.setrecursionlimit(10000)

import os
import shutil
import torch

from Verdict.gen_model.model.mlp import MLP


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='MLP example')
    parser.add_argument('--policy', type=str, help='policy choice, starting with "PAS"')
    parser.add_argument('--dim', type=int, default=1024, help='model hidden size')
    parser.add_argument('--layers', type=int, default=16, help='number of linear layers')
    parser.add_argument('--gbs', type=int, default=4, help='global batch size')
    parser.add_argument('--mbs', type=int, default=4, help='micro batch size')
    parser.add_argument('--fp16', action='store_true', default=False, help='use fp16 for the training')
    parser.add_argument('--dp_size', type=int, default=1, help='size of data parallelism')
    parser.add_argument('--pp_size', type=int, default=1, help='size of pipeline parallelism')
    parser.add_argument('--tp_size', type=int, default=1, help='size of tensor parallelism')
    parser.add_argument('--zero', action='store_true', default=False, help='use zero1 for the training')
    args = parser.parse_args()
    return args



def create_path(args):
    dp = args.dp_size
    pp = args.pp_size
    tp = args.tp_size
    gbs = args.gbs
    mbs = args.mbs
    layers = args.layers
    hi = args.dim
    nm = gbs // dp // mbs if args.policy in ["hybrid"] else 1
    strategy = "default"
    fname = f"mlp_{strategy}_dp{dp}_pp{pp}_tp{tp}_nm{nm}_gbs{gbs}_ly{layers}_h{0}_hi{hi}_sq{0}"
    dst = f"gen_model/mgeners/{fname}.pkl"
    return dst


def gen_model(args, dst):
    import nnscaler
    from nnscaler.parallel import parallelize, ComputeConfig
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel

    dp = args.dp_size
    pp = args.pp_size
    tp = args.tp_size
    gbs = args.gbs
    mbs = args.mbs
    nm = gbs // dp // mbs if args.policy in ["hybrid"] else 1
    ngpus = dp * pp * tp
    policy_name = "pas_" + args.policy

    # initialize
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
    initialize_model_parallel(model_parallel_size)

    # sanity check
    nnscaler.init()
    # if torch.distributed.get_world_size() != ngpus:
    #     raise ValueError("world size should be equal to dp_size * pp_size * tp_size")
    if gbs % mbs != 0:
        raise ValueError("global batch size should be divisible by micro batch size")

    # model
    model = MLP(dim=args.dim, nlayers=args.layers)
    model = model if not args.fp16 else model.half()

    # dummy_input
    def dummy_data():
        return torch.randn(
            args.mbs, args.dim, device=torch.cuda.current_device())
    dummy_input = {"data": dummy_data()}

    # get policy
    policy = args.policy  # use the builtin policies

    # compute_config
    compute_config = ComputeConfig(
        plan_ngpus=pp * tp,
        # runtime_ngpus=torch.distributed.get_world_size(),
        runtime_ngpus=ngpus,
        use_zero=args.zero,
        use_end2end=True,
        constant_folding=True,
        use_pipeline=pp > 1,
        pipeline_nmicros=nm,
        pipeline_nstages=pp,
        pas_config={
            # customized settings that can affect code generation.
            "_pas_name": args.policy,
            "_gbs": gbs,
            "_pp_size": pp,
            "_tp_size": tp,
            "_dp_size": dp,
        },
        user_config={
            "mbs": mbs,
        },
    )
    # print(compute_config)
    # print(input_ids.shape)
    # exit(0)

    # parallelization
    pmodel = parallelize(
        module_or_module_class=model,
        dummy_input=dummy_input,
        pas_policy=policy,
        compute_config=compute_config,
        gen_savedir="./.nnscaler",
        reuse="override",
        # instance_name: Optional[str] = None,
        load_module=False,
        # module_dtype:  Optional[torch.dtype] = None,
        # module_fn: Optional[Callable[[], torch.nn.Module]] = None,
        # init_module_params: bool = True,
        # broadcast_strategy: Union[str, BroadcastGenFilesStrategy] = 'none',
    )

    file = "mgener.pkl"
    try:
        shutil.move(file, dst)
    except:
        pass


if __name__ == "__main__":
    args = parse_arguments()
    dst = create_path(args)

    print(f"👉 Model destination: {dst}")
    if os.path.exists(dst):
        print(f"💾 Using cached model.")
    else:
        print(f"🏎️ Start generating NNScaler model...")
        gen_model(args, dst)
    print(f"✅ Model ready.")
