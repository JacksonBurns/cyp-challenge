"""Parse zoo13/zoo14 gate outputs -> per-isoform cv gains, real-minus-shuf."""
import json
import re
import os

CACHE = "/home/jackson/cyp-challenge/cache"
FILES = {
    "zoo13_real": "zoo13_real_out.txt",
    "zoo13_shuf": "zoo13_shuf_out.txt",
    "zoo14_ext": "zoo14_ext_out.txt",
    "zoo14_extshuf": "zoo14_extshuf_out.txt",
    "zoo14_adme": "zoo14_adme_out.txt",
    "zoo14_admeshuf": "zoo14_admeshuf_out.txt",
    "zoo14_both": "zoo14_both_out.txt",
    "zoo14_bothshuf": "zoo14_bothshuf_out.txt",
}

def parse(fn):
    p = os.path.join(CACHE, fn)
    if not os.path.exists(p):
        return None
    out = {}
    for ln in open(p):
        m = re.match(r"(CYP\w+) (\{.*\})", ln.strip())
        if m:
            d = json.loads(m.group(2))
            out[m.group(1)] = (d["cv_mean_gain"], d["L4"]["delta_vs_incumbent"])
    return out

res = {k: parse(v) for k, v in FILES.items()}
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
for tag in ["zoo13", "zoo14_ext", "zoo14_adme", "zoo14_both"]:
    r = res.get(tag)
    s = res.get(tag + "_shuf") if tag != "zoo14_ext" else res.get("zoo14_extshuf")
    if not r:
        continue
    print(f"== {tag} ==")
    for iso in ISO:
        if iso in r:
            line = f"{iso}: cv_gain {r[iso][0]:+.4f} L4 {r[iso][1]:+.4f}"
            if s and iso in s:
                line += f" | shuf cv {s[iso][0]:+.4f} -> real-minus-shuf {r[iso][0]-s[iso][0]:+.4f}"
            print(line)
