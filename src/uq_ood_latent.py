"""UQ latent-embedding OOD (Oct 25 directive part 2): flag issue-causing
TEST molecules via MPNN fingerprint-space distance to training chemistry.

Encoder = frozen CheMeLeon init (no training needed, ~1 min GPU).
Per test molecule: mean distance to k nearest TRAIN fingerprint neighbors
(cosine on standardized 300-d molspace), per isoform restricted to rows
with labels for that iso. Report + write cache/uq_ood_test.csv:
 - high-OOD decile flags per isoform
 - validation: on LABELED train rows (OOF-free proxy), does latent kNN
   distance correlate with |OOF residual| (cross_family UQ as target)?
"""
import os
import sys

import numpy as np
import pandas as pd
import torch
from chemprop import data, models, nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loss_zoo as L  # noqa: E402
import ft_ext  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CACHE = L.CACHE
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
K = 25


def encode(model, smiles_list, device, featurizer=None):
    featurizer = featurizer or ft_ext.featurizers.SimpleMoleculeMolGraphFeaturizer()
    model.eval()
    ok, outs = [], []
    with torch.no_grad():
        for s in smiles_list:
            try:
                p = data.MoleculeDatapoint.from_smi(s)
                mg = featurizer(p.mol)
                bmg = data.BatchMolGraph([mg])
                bmg.to(device)
                z = model.fingerprint(bmg, None, None)
                outs.append(z.cpu().numpy()[0])
                ok.append(s)
            except Exception:
                pass
    return np.stack(outs), ok


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = torch.load(ft_ext.WEIGHTS, map_location="cpu", weights_only=False)
    mp = nn.BondMessagePassing(**st["hyper_parameters"])
    mp.load_state_dict(st["state_dict"])
    model = models.MPNN(mp, nn.MeanAggregation(),
                        nn.RegressionFFN(input_dim=mp.output_dim),
                        batch_norm=False)
    model.to(device)
    ft, test, allrows = ft_ext.load_tables()
    tr_smi = ft["SMILES"].values
    te_smi = test["SMILES"].values
    Ztr, tr_smi = encode(model, list(tr_smi), device)
    Zte, te_smi = encode(model, list(te_smi), device)
    tr_smi = np.array(tr_smi); te_smi = np.array(te_smi)
    print("latents", Ztr.shape, Zte.shape, flush=True)
    sc = StandardScaler().fit(Ztr)
    A = sc.transform(Ztr); B = sc.transform(Zte)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    # distances
    def knn_dist(Q, P):
        S = Q @ P.T
        d = 1.0 - S
        return np.sort(d, axis=1)[:, :K].mean(axis=1)
    d_te = knn_dist(B, A)
    S = A @ A.T
    np.fill_diagonal(S, -np.inf)
    d_tr = np.sort(1.0 - S, axis=1)[:, :K].mean(axis=1)

    out = pd.DataFrame({"SMILES": te_smi, "ood_knn": d_te})
    out["ood_flag"] = (out.ood_knn >= out.ood_knn.quantile(0.9)).astype(int)
    out.to_csv(os.path.join(CACHE, "uq_ood_test.csv"), index=False)
    print(f"test n={len(out)} flagged(top10%)={int(out.ood_flag.sum())}",
          flush=True)

    # validation on train rows vs OOF cross-family UQ + residual
    cs = pd.read_csv(os.path.join(CACHE, "uq_cross_spread.csv"), index_col=0)
    lab = ft.drop_duplicates("SMILES").set_index("SMILES")
    from scipy.stats import spearmanr
    for iso in ISO:
        s = pd.Series(d_tr, index=tr_smi)
        u = cs[f"{iso}_uq"].reindex(s.index)
        y = lab[iso].reindex(s.index).astype(float)
        oof = pd.read_csv(os.path.join(CACHE, "ft_oof_ext.csv")) \
              .drop_duplicates("SMILES").set_index("SMILES")[iso]
        o = oof.reindex(s.index)
        m = u.notna() & y.notna() & o.notna()
        r = (y - o).abs()
        rho_u = spearmanr(s[m], u[m]).statistic
        rho_r = spearmanr(s[m], r[m]).statistic
        print(f"{iso}: corr(ood_knn, crossUQ)={rho_u:.3f} "
              f"corr(ood_knn, |resid|)={rho_r:.3f} n={int(m.sum())}",
              flush=True)


if __name__ == "__main__":
    main()
