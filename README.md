# DBM-TGP-F0

It's a method for synthesizing F0 contours of Bengali readout speech from the textual features of input text using Deep Boltzmann Machine (DBM) and Twin Gaussian Process (TGP) hybrid model. DBM will capture the high-level linguistic structure of input text and improve the prediction accuracy when plug into the TGP model. Unlike Gaussian Process (GP) models which only focus on the prediction of a single output (F0), TGP can generalize across multiple outputs (F0, delta F0, delta-delta F0) by encoding relations between both inputs and outputs with GP priors.

----
ROOT.m  => Main program
data_preprocess_class.m  => Preprosessing of data when output F0 is considered as class ranging from 75Hz-500Hz
data_preprocess_reg.m  => Preprosessing of data when output F0 is considered as a single value
result_DBMTGP.m => DBM-TGP combined algo


SYNTHESIS.m  => synthesis of generated F0 contur with MLSA filter
