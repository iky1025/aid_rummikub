"""Reproduction: HiGHS 1.13.1 MIP presolve returns a suboptimal solution
declared kOptimal (gap 0.0) on repro.mps. presolve=off finds the true optimum.

Usage: python reproduce.py   (requires highspy, repro.mps in the same dir)
"""

import os
import highspy

MPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "repro.mps")

print("HiGHS version:", highspy.Highs().version())
for presolve in ("on", "off"):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("presolve", presolve)
    h.setOptionValue("threads", 1)
    h.readModel(MPS)
    h.run()
    info = h.getInfo()
    print(
        f"presolve={presolve:3s}: status={h.getModelStatus()}, "
        f"objective={h.getObjectiveValue()}, mip_gap={info.mip_gap}"
    )

print()
print("Expected: both report objective -59 (verified by CBC and by an")
print("independent exact DP solver for this problem class).")
print("Observed on 1.13.1: presolve=on -> -58 (suboptimal, declared optimal,")
print("gap 0.0); presolve=off -> -59 (correct).")
print("Not reproducible on 1.14.0 (both -59).")
