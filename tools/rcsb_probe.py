import json
import urllib.request

UP = {"CYP1A2": "P05177", "CYP2C9": "P11712", "CYP2D6": "P10635", "CYP3A4": "P08684"}
q = {
    "query": {
        "type": "group",
        "logical_operator": "and",
        "nodes": [
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_polymer_instance_annotation_type.annotation_id",
                            "operator": "equal", "value": "uniprot_P05177"}},
            {"type": "terminal", "service": "text",
             "parameters": {"attribute": "rcsb_pubmed_central_id", "operator": "exists", "value": True}},
        ],
    },
    "return_type": "polymer_entity",
    "request_options": {
        "paginate": {"start": 0, "rows": 3},
        "facets": [{"name": "pf", "aggregation_type": "terms",
                    "terms": {"attribute": "rcsb_polymer_entity.pdb_id", "missing": False}}],
    },
}

def search(up):
    q["query"]["nodes"][0]["parameters"]["value"] = f"uniprot_{up}"
    body = json.dumps(q).encode()
    req = urllib.request.Request("https://search.rcsb.org/rcsbsearch/v2/query",
                                 data=body, headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=30).read())
    return [x["result_identifier"] for x in d.get("result_set", [])]

for iso, up in UP.items():
    rs = search(up)
    print(iso, up, len(rs))
