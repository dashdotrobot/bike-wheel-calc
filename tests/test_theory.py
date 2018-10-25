import pytest
from bikewheelcalc import *


# -----------------------------------------------------------------------------
# Buckling tests
# -----------------------------------------------------------------------------

def test_Tc_linear(std_ncross):
	'Linear approximation'

	w = std_ncross(0)

	Tc = calc_buckling_tension(w, approx='linear', N=20)

	print(Tc)

	assert False
