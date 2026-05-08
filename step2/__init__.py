"""Step 2: segmented CPS optimization for traveling-wave Mach-Zehnder modulator.

Outer loop: 10 C values (linearly spaced over Step-1's range), independent CPS
optimization at each C.  Inner loop per C: 8 LHS + 12 BO (soft cap 20, hard
cap 40), with reviews at LHS-end and every 4 BO runs.

See PLAN.md for the full specification.
"""

__version__ = "0.1.0"
