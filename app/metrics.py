import json
from collections import Counter

from . import config


def read_log():
    if not config.LOG_FILE.exists():
        return []
    out = []
    for line in config.LOG_FILE.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        out.append({"scenario": r.get("scenario_label") or "unlabeled", "decision": r["decision"],
                    "total_ms": r.get("total_ms", 0.0)})
    return out


def summarize(records):
    by = {}
    for r in records:
        by.setdefault(r["scenario"], []).append(r)
    rows = []
    for sc, rs in sorted(by.items()):
        n = len(rs)
        c = Counter(r["decision"] for r in rs)
        ms = sorted(r["total_ms"] for r in rs)
        row = {"scenario": sc, "kind": "genuine" if sc == "genuine" else "attack", "n": n,
               "allow": c["ALLOW"], "request": c["REQUEST_VERIFICATION"], "deny": c["DENY"],
               "allow_rate": c["ALLOW"] / n, "request_rate": c["REQUEST_VERIFICATION"] / n,
               "deny_rate": c["DENY"] / n,
               "median_ms": ms[n // 2], "p95_ms": ms[min(n - 1, int(0.95 * n))]}
        # attacks: allow_rate = attack acceptance (lower is better); genuine: deny_rate = false rejection
        if sc == "unlabeled":
            row["key_metric"] = "n/a (no scenario label)"
        elif row["kind"] == "genuine":
            row["key_metric"] = "false_reject=%.1f%%" % (100 * row["deny_rate"])
        else:
            row["key_metric"] = "attack_accepted=%.1f%%" % (100 * row["allow_rate"])
        rows.append(row)
    return rows


def format_table(rows):
    head = f"{'scenario':<16}{'n':>4}{'ALLOW':>7}{'REQ':>6}{'DENY':>6}  {'key metric':<26}{'med ms':>8}{'p95 ms':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['scenario']:<16}{r['n']:>4}{r['allow']:>7}{r['request']:>6}{r['deny']:>6}  "
                     f"{r['key_metric']:<26}{r['median_ms']:>8.0f}{r['p95_ms']:>8.0f}")
    return "\n".join(lines)
