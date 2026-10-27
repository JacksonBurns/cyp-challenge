"""ADME2 data acquisition: TDC (Therapeutics Data Commons) ADME + tox sets.

Endpoints for assay-physics relevance to CYP pIC50 prediction:
  direct CYP labels (binary substrate/inhibitor, ~15k each):
    CYP2C9_Veith, CYP2D6_Veith, CYP3A4_Veith, CYP1A2_Veith
  pKa (ABSENT from the challenge stack): PKa_AstraZeneca
  solubility: Solubility_AqSolDB
  permeability: Caco2_Wang, HIA_Hou, Pgp_Broccatelli, BBB_Martins
  distribution: VDss_Lombardo, PPBR_AZ (protein binding)
  metabolism: Half_Life_Obach, Clearance_Hepatocyte, Clearance_Microsome
  lipophilicity: Lipophilicity_AstraZeneca
Writes one clean CSV per dataset under external/user_supplied/tdc/.
"""
import os

import pandas as pd
from pytdc.ground_adme.benchmark import benchmark
from pytdc.tdc_dataset import get_multi_valid

OUT = "/home/jackson/cyp-challenge/external/user_supplied/tdc"
os.makedirs(OUT, exist_ok=True)

# (dataset name, TDC group function, output filename)
SPEC = [
    ("CYP2C9_Veith", "toxicology", "cyp2c9_veith.csv"),
    ("CYP2D6_Veith", "toxicology", "cyp2d6_veith.csv"),
    ("CYP3A4_Veith", "toxicology", "cyp3a4_veith.csv"),
    ("CYP1A2_Veith", "toxicology", "cyp1a2_veith.csv"),
    ("CYP2C9_Substrate_CarbonMangels", "toxicology", "cyp2c9_sub.csv"),
    ("CYP2D6_Substrate_CarbonMangels", "toxicology", "cyp2d6_sub.csv"),
    ("CYP3A4_Substrate_CarbonMangels", "toxicology", "cyp3a4_sub.csv"),
    ("PKa_AstraZeneca", "admet_group", "pka_az.csv"),
    ("Solubility_AqSolDB", "admet_group", "aqsoldb.csv"),
    ("Caco2_Wang", "admet_group", "caco2_wang.csv"),
    ("HIA_Hou", "admet_group", "hia_hou.csv"),
    ("Pgp_Broccatelli", "admet_group", "pgp_broccatelli.csv"),
    ("BBB_Martins", "admet_group", "bbb_martins.csv"),
    ("VDss_Lombardo", "admet_group", "vdss_lombardo.csv"),
    ("PPDB_BRONZE", "admet_group", "ppdb_bronze.csv"),
    ("Half_Life_Obach", "admet_group", "half_life_obach.csv"),
    ("Clearance_Hepatocyte_AZ", "admet_group", "cl_hepatocyte.csv"),
    ("Clearance_Microsome_AZ", "admet_group", "cl_microsome.csv"),
    ("Lipophilicity_AstraZeneca", "admet_group", "lipophilicity_az.csv"),
]


def save(df, fn):
    path = os.path.join(OUT, fn)
    df.to_csv(path, index=False)
    print(f"  {fn}: {len(df)} rows, cols {list(df.columns)[:4]}")


def main():
    for name, group, fn in SPEC:
        try:
            if group == "toxicology":
                from pytdc.tdc_ockernel import get_cyp_data
                pass
            import pytdc
            # generic: single_dataset loader
            if group == "toxicology":
                from pytdc.oxology.admt_cyp import get_dataset
            else:
                from pytdc.admet.benchmark import get_dataset as get_admet
                get_dataset = get_admet
            d = get_dataset(name)
            save(pd.DataFrame(d), fn)
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
