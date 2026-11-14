"""One CLI for the whole data -> train -> evaluate loop.

  python tools.py collect  --person nithya --scenario genuine --cond normal --round train
  python tools.py enroll   [--dir data/enroll]
  python tools.py train    [--model lr|gbm]
  python tools.py evaluate [--root data/test]

Folder layout (person names must not contain '_'):
  data/enroll/<person>/<cond>_<k>/frame_00.jpg ...     (selfies for enrolment, collect --scenario enroll)
  data/train/<scenario>/<person>_<cond>_<k>/frame_00.jpg ...
  data/test/<scenario>/<person>_<cond>_<k>/frame_00.jpg ...
scenario: genuine | print | screen_photo | video_replay | mask   ('genuine' = live person, rest = attacks)
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from app import config
from app.detect import prep_frame
from app.metrics import format_table, summarize

IMG_EXT = {".jpg", ".jpeg", ".png"}
DATA = config.DATA_DIR


def read_images(paths):
    out = []
    for f in paths:
        im = cv2.imread(str(f))
        if im is not None:
            out.append(prep_frame(im))
    return out


def list_samples(root: Path):
    for sc in sorted(d for d in root.iterdir() if d.is_dir()):
        for item in sorted(sc.iterdir()):
            if item.is_dir() or item.suffix.lower() in IMG_EXT:
                name = item.name if item.is_dir() else item.stem
                yield sc.name, name.split("_")[0], item


def sample_frames(item: Path):
    if item.is_file():
        return read_images([item])
    return read_images(sorted(f for f in item.iterdir() if f.suffix.lower() in IMG_EXT))


# ------------------------------------------------------------------ collect
def cmd_collect(a):
    cap = cv2.VideoCapture(a.cam)
    if not cap.isOpened():
        raise SystemExit("cannot open camera")
    if a.scenario == "enroll":
        base, prefix = DATA / "enroll" / a.person, f"{a.cond}_"
    else:
        base, prefix = DATA / a.round / a.scenario, f"{a.person}_{a.cond}_"
    base.mkdir(parents=True, exist_ok=True)
    k = len(list(base.glob(prefix + "*")))
    print("SPACE = capture a burst, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"{a.scenario}/{a.person}/{a.cond}  saved={k}  [SPACE] capture  [q] quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("collect", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            out_dir = base / f"{prefix}{k:03d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(a.frames):
                ok, fr = cap.read()
                if ok:
                    cv2.imwrite(str(out_dir / f"frame_{i:02d}.jpg"), fr)
                cv2.waitKey(max(1, int(1000 * a.span / a.frames)))
            k += 1
            print("saved", out_dir)
    cap.release()
    cv2.destroyAllWindows()


# ------------------------------------------------------------------- enroll
def cmd_enroll(a):
    from app.pipeline import Gate
    gate, root = Gate(), Path(a.dir)
    for pdir in sorted(d for d in root.iterdir() if d.is_dir()):
        imgs = read_images(sorted(f for f in pdir.rglob("*") if f.suffix.lower() in IMG_EXT))
        try:
            n = gate.enroll_images(pdir.name, imgs)
            print(f"enrolled {pdir.name}: {n} good frames (from {len(imgs)} images)")
        except Exception as e:
            print(f"could not enroll {pdir.name}: {e}")


# -------------------------------------------------------------------- train
def cmd_train(a):
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from app.pipeline import Gate

    gate = Gate()
    X, y, groups, scen, skipped = [], [], [], [], 0
    for sc, person, item in list_samples(DATA / "train"):
        res = gate.analyze(sample_frames(item), need_identity=False)
        if not res["ok"]:
            skipped += 1
            continue
        X.append(res["feats"]); y.append(1 if sc == "genuine" else 0); groups.append(person); scen.append(sc)
    y = np.array(y)
    print(f"{len(y)} usable samples ({int((y == 1).sum())} genuine, {int((y == 0).sum())} attacks), "
          f"{skipped} skipped by the quality gate")
    if min((y == 1).sum(), (y == 0).sum()) < 6:
        raise SystemExit("need at least ~6 genuine and ~6 attack samples; collect more")

    names = sorted({k for f in X for k in f})
    M = np.array([[f.get(n, 0.0) for n in names] for f in X])
    clf = (HistGradientBoostingClassifier(max_depth=3, learning_rate=0.1, max_iter=150)
           if a.model == "gbm" else LogisticRegression(C=a.C, class_weight="balanced", max_iter=3000))
    pipe = make_pipeline(StandardScaler(), clf)

    persons = sorted(set(groups))
    minc = int(min((y == 1).sum(), (y == 0).sum()))
    try:
        if len(persons) >= 3:
            cv = GroupKFold(n_splits=min(5, len(persons)))
            oof = cross_val_predict(pipe, M, y, groups=groups, cv=cv, method="predict_proba")[:, 1]
            print(f"cross-validation: leave-person-out style ({cv.n_splits} folds)")
        else:
            cv = StratifiedKFold(n_splits=min(5, minc), shuffle=True, random_state=0)
            oof = cross_val_predict(pipe, M, y, cv=cv, method="predict_proba")[:, 1]
            print("cross-validation: stratified (fewer than 3 people, so this is optimistic)")
    except Exception as e:
        raise SystemExit(f"cross-validation failed: {e}")

    grid = np.linspace(0.05, 0.95, 91)
    att, gen = oof[y == 0], oof[y == 1]
    t_allow = next((float(t) for t in grid if (att >= t).mean() <= a.max_far), config.T_ALLOW_DEFAULT)
    ok_deny = [float(t) for t in grid if (gen < t).mean() <= a.max_frr]
    t_deny = max(ok_deny) if ok_deny else config.T_DENY_DEFAULT
    if t_deny >= t_allow:
        t_deny = max(0.05, t_allow - 0.2)
    print(f"thresholds: DENY below {t_deny:.2f}, ALLOW at/above {t_allow:.2f}, request verification between")

    print("\nliveness-only, out-of-fold (not a final result; use `evaluate` on data/test):")
    for sc in sorted(set(scen)):
        p = oof[[s == sc for s in scen]]
        print(f"  {sc:<14} n={len(p):<4} live-accepted={100 * (p >= t_allow).mean():5.1f}%  "
              f"denied={100 * (p < t_deny).mean():5.1f}%  step-up={100 * ((p >= t_deny) & (p < t_allow)).mean():5.1f}%")

    pipe.fit(M, y)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": names, "t_deny": t_deny, "t_allow": t_allow,
                 "meta": {"n": int(len(y)), "model": a.model, "persons": persons}}, config.LIVENESS_PATH)
    print("saved", config.LIVENESS_PATH, "- restart the API to load it")


# ----------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    from app.pipeline import Gate
    gate, recs = Gate(), []
    for i, (sc, person, item) in enumerate(list_samples(Path(a.root))):
        frames = sample_frames(item)
        if not frames:
            continue
        r = gate.verify_images(frames, claimed_id=person if person in gate.store.tpl else None,
                               device_id=f"eval-{i}", session_id=f"eval-{i}", scenario_label=sc, log=False)
        recs.append({"scenario": sc, "person": person, "sample": item.name, "decision": r["decision"],
                     "reason": r["reason_code"], "p_live": r["liveness_score"],
                     "similarity": r["identity"]["similarity"], "total_ms": r["timings_ms"]["total_server"]})
    if not recs:
        raise SystemExit(f"no samples under {a.root}")
    rows = summarize(recs)
    print(format_table(rows))
    print("\nNotes: attacks -> 'attack_accepted' is the share wrongly ALLOWed; genuine -> 'false_reject' is the share DENYed "
          "(REQ = sent to extra verification). Times are server-side only (no camera capture or network).")
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name, data in (("samples", recs), ("summary", rows)):
        with open(out / f"eval_{name}_{stamp}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--person", required=True)
    c.add_argument("--scenario", required=True,
                   help="enroll | genuine | print | screen_photo | video_replay | mask")
    c.add_argument("--cond", default="normal")
    c.add_argument("--round", default="train", choices=["train", "test"])
    c.add_argument("--cam", type=int, default=0)
    c.add_argument("--frames", type=int, default=8)
    c.add_argument("--span", type=float, default=0.8)
    c.set_defaults(fn=cmd_collect)
    e = sub.add_parser("enroll")
    e.add_argument("--dir", default=str(DATA / "enroll"))
    e.set_defaults(fn=cmd_enroll)
    t = sub.add_parser("train")
    t.add_argument("--model", default="lr", choices=["lr", "gbm"])
    t.add_argument("--C", type=float, default=1.0)
    t.add_argument("--max-far", type=float, default=0.02, dest="max_far")
    t.add_argument("--max-frr", type=float, default=0.05, dest="max_frr")
    t.set_defaults(fn=cmd_train)
    v = sub.add_parser("evaluate")
    v.add_argument("--root", default=str(DATA / "test"))
    v.set_defaults(fn=cmd_evaluate)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
