"""Adapt jeremy's adme_pretrain MPNN checkpoints to ft_ext's --pretrained schema.

jeremy's /tmp/.../checkpoints/{chemprop_chemeleon,chemprop_medium}.pt save a
FULL MPNN: state_dict keys are message_passing.* + predictor.*, hyper_parameters
is the MPNN-level dict. ft_ext.py --pretrained expects the bare-encoder schema
(hyper_parameters = BondMessagePassing kwargs, state_dict keys mp.*).

This strips predictor.* (fine-tune uses a fresh 12-head head), renames
message_passing.* -> mp.*, and lifts the message_passing sub-dict of
hyper_parameters. Aggregation is NOT stored: ft_ext always uses MeanAggregation,
note chemprop_medium was trained with NormAggregation(norm=100) - deviation
logged in NOTES.

Usage: python src/adapt_adme_ckpt.py SRC.pt DST.pt
"""
import copy
import sys

import torch

MP_KEYS = {"activation", "bias", "d_e", "d_h", "d_v", "d_vd", "depth", "dropout",
           "graph_transform", "undirected", "V_d_transform"}


def main(src, dst):
    st = torch.load(src, map_location="cpu", weights_only=False)
    hp = st["hyper_parameters"]
    mp_hp = {k: copy.deepcopy(v) for k, v in hp["message_passing"].items()
             if k in MP_KEYS and k != "cls"}
    sd = {}
    for k, v in st["state_dict"].items():
        if k.startswith("message_passing."):
            sd[k[len("message_passing."):]] = v
    missing = [k for k in st["state_dict"] if not k.startswith("message_passing.")]
    torch.save({"hyper_parameters": mp_hp, "state_dict": sd}, dst)
    print(f"wrote {dst}: {len(sd)} mp tensors (dropped {len(missing)} predictor tensors)")
    print("hp:", {k: v for k, v in mp_hp.items() if k != "graph_transform"})


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
