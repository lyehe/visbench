"""Protocol constants shared by the metric primitives and the harness (one source of truth)."""

from __future__ import annotations

# cv2 algorithm minimums (these are baked into the cv2 estimators, not tunable).
MIN_POSE_MATCHES = 5      # cv2.findEssentialMat — 5-point algorithm
MIN_FUND_MATCHES = 8      # cv2.findFundamentalMat — 8-point algorithm
MIN_HOMOG_MATCHES = 4     # cv2.findHomography

# RANSAC confidence levels used across estimators. The 0.99999 used for pose
# and homography matches the RoMa-canonical protocol; the lower 0.999 for the
# fundamental matrix matches LoMa/WxBS practice.
POSE_RANSAC_CONF = 0.99999
HOMOG_RANSAC_CONF = 0.99999
FUND_RANSAC_CONF = 0.999

# Iteration caps for non-MAGSAC paths.
HOMOG_RANSAC_MAX_ITERS = 10_000
FUND_RANSAC_MAX_ITERS = 100_000

# Pose/homography metric fallbacks.
POSE_FAIL_DEG = 90.0
