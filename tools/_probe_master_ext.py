import pandas as pd
m = pd.read_csv('cache/umtm_master.csv')
ext = m[m.kind == 'ext']
heads = [c for c in m.columns if c.endswith('_chembl') or c.endswith('_pubchem')]
print('ext rows', len(ext))
print({c: int(ext[c].notna().sum()) for c in heads})
sc = m[m.kind == 'sc_only']
print('sc_only rows:', len(sc),
      sc[['CYP1A2_log2fc', 'CYP2C9_log2fc', 'CYP2D6_log2fc', 'CYP3A4_log2fc']].notna().sum().to_dict())
print('ext log2fc cols present:', [c for c in ext.columns if 'log2fc' in c], 'notna:',
      int(ext[[c for c in ext.columns if 'log2fc' in c]].notna().sum().sum()))
