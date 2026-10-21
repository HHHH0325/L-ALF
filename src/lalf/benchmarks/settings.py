"""Fixed manuscript protocol; scientific defaults follow the main package."""

import json
from pathlib import Path
from lalf.models import Thresholds

P = json.loads(Path(__file__).with_name("paper_protocol.json").read_text(encoding="utf-8"))
S1_LOW = Thresholds().s1_low
S1_HIGH = Thresholds().s1_high
EST_MAX = Thresholds().est_max
F_MIN, RATIO_MIN = P["relaxed"]["f"], P["relaxed"]["ratio"]
F_STRICT_MIN, RATIO_STRICT_MIN = P["strict"]["f"], P["strict"]["ratio"]
assert P["constraints"] == {"S1_low": S1_LOW, "S1_high": S1_HIGH, "EST_max": EST_MAX}
