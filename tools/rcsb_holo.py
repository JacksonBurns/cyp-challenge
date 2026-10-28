"""Find holo CYP structures (heme + inhibitor) via RCSB search API."""
import json
import urllib.request

UP = {"CYP1A2": "P05177", "CYP2C9": "P11712", "CYP2D6": "P10635", "CYP3A4": "P08684"}

Q = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_polymer_entity_align.source_db_accession",
                            "operator": "exact_match", "value": "P05177"}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_containing_components",
                            "operator": "exact_match", "value": "HEM"}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_container_identifiers.nonpolymer_entity_ids",
                            "operator": "exists", "value": True}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_entry_info.resolution_combined",
                            "operator": "less", "value": 2.8}},
        ],
    },
    "return_type": "entry",
    "request_options": {"paginate": {"start": 0, "rows": 8},
                        "sort": [{"sort_by": "rcsb_accession", "direction": "asc"}]},
}


def search(up):
    Q["query"]["nodes"][0]["parameters"]["value"] = up
    body = json.dumps(Q).encode()
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=40).read())
    return [x["data"]["id"] for x in d.get("result_set", [])]


for iso, up in UP.items():
    ids = search(up)
    print(iso, up, "->", ids, flush=True)
