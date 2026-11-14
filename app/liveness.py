import joblib
import numpy as np

from . import config


class LivenessModel:
    """Wraps the trained fusion classifier. Until you run `python tools.py train`, it returns 0.5
    (uncertain) so the whole pipeline still runs end to end."""

    def __init__(self, path=config.LIVENESS_PATH):
        self.trained = False
        self.pipe = None
        self.names = []
        self.meta = {}
        self.t_deny = config.T_DENY_DEFAULT
        self.t_allow = config.T_ALLOW_DEFAULT
        if path.exists():
            d = joblib.load(path)
            self.pipe, self.names, self.meta = d["pipeline"], d["feature_names"], d.get("meta", {})
            self.t_deny, self.t_allow = d["t_deny"], d["t_allow"]
            self.trained = True
        if config.T_DENY:
            self.t_deny = float(config.T_DENY)
        if config.T_ALLOW:
            self.t_allow = float(config.T_ALLOW)

    def score(self, feats: dict) -> float:
        if not self.trained:
            return 0.5
        x = np.array([[feats.get(n, 0.0) for n in self.names]], dtype=np.float64)
        return float(self.pipe.predict_proba(x)[0, 1])   # classes_ = [0 spoof, 1 live]
