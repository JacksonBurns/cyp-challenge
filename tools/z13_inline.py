import json
import re

def parse(fn):
    out = {}
    for ln in open('/home/jackson/cyp-challenge/cache/' + fn):
        m = re.match(r"(CYP\w+) (\{.*\})", ln.strip())
        if m:
            d = json.loads(m.group(2))
            out[m.group(1)] = (d["cv_mean_gain"], d["L4"]["delta_vs_incumbent"])
    return out

r = parse('zoo13_real_out.txt')
s = parse('zoo13_shuf_out.txt')
for iso in ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]:
    a = r.get(iso)
    b = s.get(iso)
    if a and b:
        print(f"{iso}: real cv {a[0]:+.4f} L4 {a[1]:+.4f} | shuf cv {b[0]:+.4f} -> rms {a[0]-b[0]:+.4f}")
    else:
        print(iso, "missing", bool(a), bool(b))
