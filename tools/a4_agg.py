"""Aggregate A4 gate seeds: A4-on-A2 delta per isoform, mean +- sd."""
import re, sys, statistics as st
ISO = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
rows = {}
for f in sys.argv[1:]:
    d = {}
    for ln in open(f):
        m = re.match(r"(CYP\d[A-Z]\d): base .* A4-on-A2 delta ([+-][\d.]+)", ln)
        if m: d[m.group(1)] = float(m.group(2))
    if len(d) == 4: rows[f] = d
print(f"seeds loaded: {len(rows)}")
for iso in ISO:
    vs = [d[iso] for d in rows.values()]
    print(f"{iso}: " + " ".join(f"{v:+.4f}" for v in vs) +
          f"  mean {st.mean(vs):+.4f} sd {st.stdev(vs):.4f}")
m = [st.mean([d[i] for i in ISO]) for d in rows.values()]
print(f"MACRO mean {st.mean(m):+.4f}")
