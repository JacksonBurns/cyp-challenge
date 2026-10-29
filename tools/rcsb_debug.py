import json
import urllib.request

for pid in ["3ac2", "1eu9", "4uou"]:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pid.upper()}"
    try:
        d = json.loads(urllib.request.urlopen(url, timeout=30).read())
        print(pid, "keys:", [k for k in d.keys() if "poly" in k.lower()])
        npe = d.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids")
        print(pid, "npe:", npe)
        npe = d.get("rcsb_entry_container_identifiers", {}).get("nonpolymer_entity_ids")
        print(pid, "npe2:", npe)
        ents = d.get("nonpolymer_entities")
        print(pid, "nonpolymer_entities:", json.dumps(ents)[:200] if ents else None)
    except Exception as e:
        print(pid, "ERR", e)
