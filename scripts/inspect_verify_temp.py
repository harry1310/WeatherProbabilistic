"""Find the 96h vs 120h asymmetry in the live verify_temperature_2026-05-07.json."""
import json
from collections import defaultdict
from pathlib import Path

p = Path("C:/Users/rhcsl/AppData/Local/Temp/verify_temperature_2026-05-07.json")
d = json.loads(p.read_text())
print(f"target={d.get('target')} asOfUtc={d.get('asOfUtc')} windowDays={d.get('windowDays')}")
print(f"latencyDays={d.get('latencyDays')} metric={d.get('metricLabel')}")
print(f"rows: {len(d['rows'])}")
print()

leads_seen = sorted({r.get("leadHours") for r in d["rows"] if "leadHours" in r})
print(f"Leads present: {leads_seen}")
print()

by_lead = defaultdict(list)
for r in d["rows"]:
    by_lead[r.get("leadHours")].append(r)

for lead in sorted(by_lead.keys()):
    rows = by_lead[lead]
    valid_times = sorted({r.get("validTimeUtc") or r.get("ValidTimeUtc") or r.get("windowEndUtc")
                          for r in rows})
    print(f"Lead {lead}h — {len(rows)} rows, {len(valid_times)} unique valid_times")
    if valid_times:
        print(f"  earliest: {valid_times[0]}")
        print(f"  latest:   {valid_times[-1]}")
    keys = list(rows[0].keys()) if rows else []
    print(f"  keys: {keys}")
    print()
