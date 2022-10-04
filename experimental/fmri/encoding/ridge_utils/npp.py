"""This module contains one line functions that should, by all rights, be in numpy.
"""
import numpy as np

## Demean -- remove the mean from each column
demean = lambda v: v-v.mean(0)
demean.__doc__ = """Removes the mean from each column of [v]."""
dm = demean

## Z-score -- z-score each column
def zscore(v):
	s = v.std(0)
	m = v - v.mean(0)
	for i in range(len(s)):
		if s[i] != 0.:
			m[:, i] /= s[i]
	return m

# zscore = lambda v: (v-v.mean(0))/v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Rescale -- make each column have unit variance
rescale = lambda v: v/v.std(0)
rescale.__doc__ = """Rescales each column of [v] to have unit variance."""
rs = rescale

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zs(c1)*zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""

## Cross corr -- find corr. between each row of c1 and EACH row of c2
xcorr = lambda c1,c2: np.dot(zs(c1.T).T,zs(c2.T)) / (c1.shape[1])
xcorr.__doc__ = """Cross-column correlation. Finds the correlation between each row of [c1] and each row of [c2]."""
