#!/usr/bin/env python3
"""Idempotently provision the study dashboards into OpenObserve.

Called by run_all after the OpenObserve backend starts. Creates any dashboard from
`dashboards/*.json` whose title isn't already present, so a fresh box (or a wiped data
dir) comes up with the dashboards ready — no manual import. Existing ones are left alone.

Env: O2_URL (default http://127.0.0.1:5080/logs), O2_ORG (default), and credentials from
O2_USER/O2_PASSWORD or ZO_ROOT_USER_EMAIL/ZO_ROOT_USER_PASSWORD.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import ssl
import time
import urllib.error
import urllib.request

BASE = os.environ.get("O2_URL", "http://127.0.0.1:5080/logs").rstrip("/")
ORG = os.environ.get("O2_ORG", "default")
USER = os.environ.get("O2_USER") or os.environ.get("ZO_ROOT_USER_EMAIL", "admin@example.com")
PW = os.environ.get("O2_PASSWORD") or os.environ.get("ZO_ROOT_USER_PASSWORD", "ChangeMe123")
DIR = os.environ.get("DASH_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboards"))

_AUTH = base64.b64encode(f"{USER}:{PW}".encode()).decode()
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(f"{BASE}{path}", data=data, method=method,
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Basic " + _AUTH})
    try:
        return json.load(urllib.request.urlopen(r, context=_CTX, timeout=20))
    except urllib.error.HTTPError as e:
        return {"_err": e.code, "_body": e.read().decode()[:200]}
    except Exception as e:  # noqa: BLE001
        return {"_err": str(e)}


def main() -> None:
    files = sorted(glob.glob(os.path.join(DIR, "*.json")))
    if not files:
        print(f"[provision] no dashboard files in {DIR}")
        return
    # wait for the API to answer (start already waited on the port, give it a beat more)
    listing = {}
    for _ in range(30):
        listing = _req("GET", f"/api/{ORG}/dashboards?folder=default")
        if isinstance(listing, dict) and "dashboards" in listing:
            break
        time.sleep(1)
    existing = {d.get("title") for d in listing.get("dashboards", [])} if isinstance(listing, dict) else set()
    for f in files:
        try:
            d = json.load(open(f))
        except Exception as e:  # noqa: BLE001
            print(f"[provision] skip {f}: {e}")
            continue
        title = d.get("title", "")
        if title in existing:
            print(f"[provision] exists: {title}")
            continue
        d.setdefault("owner", USER)
        d.pop("dashboardId", None)
        r = _req("POST", f"/api/{ORG}/dashboards?folder=default", d)
        ok = isinstance(r, dict) and (r.get("v5") or r.get("v3") or r.get("v1"))
        print(f"[provision] {'created' if ok else 'FAILED'}: {title}" + ("" if ok else f"  {r}"))


if __name__ == "__main__":
    main()
