"""S2 checks: stereo VO port (frame-to-keyframe tracking, lost-frame flagging).

The 5-run full-sequence evaluation is eval/run_s2_stereo_vo.py (the ORB-SLAM3 extractor is
not deterministic, so accuracy is judged over repeated runs there).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pysteam_vo.config import load_config
from pysteam_vo.data.euroc import invert_se3
from pysteam_vo.data.stereo import StereoSequence
from pysteam_vo.frontend.stereo_vo import StereoVO

CFG = load_config()
S2 = CFG["s2_check"]
SEQ_DIR = Path(CFG["data"]["euroc_root"]) / CFG["data"]["sequence"]
needs_data = pytest.mark.skipif(not (SEQ_DIR / "mav0").is_dir(), reason=f"{SEQ_DIR} missing")


@pytest.fixture(scope="module")
def seq() -> StereoSequence:
    return StereoSequence.from_config(CFG)


@needs_data
def test_lost_frame_is_flagged_and_tracking_resumes(seq: StereoSequence):
    """A blank image cannot be tracked: it must be flagged (not silently trusted), get the
    motion-model pose, and tracking must resume within a couple of frames."""
    vo = StereoVO(seq.rig, CFG)
    blank = 1015
    for i in range(1000, 1040):
        l, r = seq.load_rectified(i)
        if i == blank:
            l, r = np.zeros_like(l), np.zeros_like(r)
        if i == blank:
            T_pred = vo._predict()
        res = vo.process(i, int(seq.image_ts_ns[i]), l, r)
        if i == blank:
            assert not res.tracked and res.n_inliers == 0
            np.testing.assert_allclose(res.T_rw, T_pred, atol=1e-12)
    f, T, tracked = vo.trajectory()
    lost = f[~tracked]
    assert blank in lost
    assert len(lost) <= 3 and lost.max() <= blank + 2          # resumes quickly
    assert tracked[f > blank + 2].all()


@needs_data
@pytest.mark.slow
def test_s2_segment_follows_gt(seq: StereoSequence):
    from pysteam_vo.eval.run_s2_stereo_vo import run_once, score
    a, b = S2["segment"]
    vo, secs = run_once(seq, CFG, a, b)
    r = score(vo, seq)
    print(f"\n[S2] frames {a}-{b}: tracked {r['tracked']}/{r['frames']}  lost {r['lost']}  keyframes {r['keyframes']}  "
          f"ATE SE3 {r['se3']['rmse']:.3f} m ({100 * r['se3']['rmse'] / r['path_m']:.2f}% of {r['path_m']:.1f} m)  "
          f"Sim3 scale {r['sim3']['scale']:.3f}  {secs:.0f} s")
    assert r["tracked"] / r["frames"] >= S2["min_tracked_fraction"]
    assert r["se3"]["rmse"] <= S2["max_ate_fraction_of_path"] * r["path_m"]
