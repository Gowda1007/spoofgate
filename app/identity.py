import json
import secrets
import threading

import cv2
import numpy as np

from . import config
from .detect import crop_box


class Embedder:
    """SFace (OpenCV Zoo, Apache-2.0) -> 128-d L2-normalised embedding."""

    def __init__(self):
        self.rec = None
        self.lock = threading.Lock()
        if config.SFACE_PATH.exists():
            self.rec = cv2.FaceRecognizerSF.create(str(config.SFACE_PATH), "")

    @property
    def ready(self):
        return self.rec is not None

    def embed(self, img, face):
        with self.lock:
            if face.row is not None:
                aligned = self.rec.alignCrop(img, face.row)
            else:
                aligned = cv2.resize(crop_box(img, face.box, 0.15), (112, 112))
            v = self.rec.feature(aligned).flatten().astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-9)


def _orthogonal(seed, dim=128):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


class IdentityStore:
    """Cancelable templates: we store Q_k @ embedding, never the raw embedding. Q_k is orthogonal, so
    cosine similarity is preserved. 'Revoking' = delete template+seed and enroll again with a new seed.
    (Demo-grade: JSON files, not a hardened vault.)"""

    def __init__(self):
        config.STORE_DIR.mkdir(parents=True, exist_ok=True)
        config.KEYS_DIR.mkdir(parents=True, exist_ok=True)
        self.tpl_path = config.STORE_DIR / "templates.json"
        self.key_path = config.KEYS_DIR / "keys.json"
        self.lock = threading.Lock()
        self.tpl = self._load(self.tpl_path)
        self.keys = self._load(self.key_path)
        self._qcache = {}

    @staticmethod
    def _load(p):
        return json.loads(p.read_text()) if p.exists() else {}

    def _q(self, seed):
        if seed not in self._qcache:
            self._qcache[seed] = _orthogonal(seed)
        return self._qcache[seed]

    def reload(self):
        with self.lock:
            self.tpl = self._load(self.tpl_path)
            self.keys = self._load(self.key_path)

    def enroll(self, user_id, embeddings):
        e = np.mean(embeddings, axis=0)
        e = e / (np.linalg.norm(e) + 1e-9)
        seed = secrets.randbits(63)
        with self.lock:
            self.keys[user_id] = seed
            self.tpl[user_id] = (self._q(seed) @ e).tolist()
            self.tpl_path.write_text(json.dumps(self.tpl))
            self.key_path.write_text(json.dumps(self.keys))

    def revoke(self, user_id):
        with self.lock:
            self.tpl.pop(user_id, None)
            self.keys.pop(user_id, None)
            self.tpl_path.write_text(json.dumps(self.tpl))
            self.key_path.write_text(json.dumps(self.keys))

    def identify(self, emb, claimed_id=None):
        """Returns (user_id or None, cosine similarity). 1:1 if claimed_id else 1:N."""
        cands = [claimed_id] if claimed_id else list(self.tpl)
        best = (None, -1.0)
        for u in cands:
            if u not in self.tpl:
                continue
            sim = float(np.dot(self._q(self.keys[u]) @ emb, np.array(self.tpl[u])))
            if sim > best[1]:
                best = (u, sim)
        return best
