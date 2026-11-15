"""Send the example patient to the prediction service.

Usage:
    python test_predict.py                                  # http://localhost:9696
    python test_predict.py http://<host>:<port>             # another URL

Only the Python standard library is used, so it runs without the project environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


base_url = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:9696"
patient = json.loads(Path(__file__).with_name("examples").joinpath("patient.json").read_text())

with urlopen(f"{base_url}/health", timeout=10) as response:
    print("health:", json.load(response))

request = Request(
    f"{base_url}/predict",
    data=json.dumps(patient).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(request, timeout=10) as response:
    result = json.load(response)

print("prediction:", result)
assert 0 <= result["readmission_probability"] <= 1
assert isinstance(result["high_risk"], bool)
