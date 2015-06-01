# DBM-TGP-F0

It's a method for synthesizing F0 contours of Bengali readout speech from the textual features of input text using Deep Boltzmann Machine (DBM) and Twin Gaussian Process (TGP) hybrid model. DBM will capture the high-level linguistic structure of input text and improve the prediction accuracy when plug into the TGP model. Unlike Gaussian Process (GP) models which only focus on the prediction of a single output (F0), TGP can generalize across multiple outputs (F0, delta F0, delta-delta F0) by encoding relations between both inputs and outputs with GP priors.

----
ROOT.m  => Main program

data_preprocess_class.m  => Preprosessing of data when output F0 is considered as class ranging from 75Hz-500Hz

data_preprocess_reg.m  => Preprosessing of data when output F0 is considered as a single value

result_DBMTGP.m => DBM-TGP combined algo

SYNTHESIS.m  => synthesis of generated F0 contur with MLSA filter

----
Please cite this paper if you use the tool.

http://www.isca-speech.org/archive/interspeech_2014/i14_2445.html

S. Mukherjee, S. K. Das Mandal, Generation of F0 Contour using Deep Boltzmann Machine and Twin Gaussian Process Hybrid Model for Bengali Language, Interspeech-2014.

bibtex
```
@inproceedings{DBLP:conf/interspeech/MukherjeeM14,
  author    = {Sankar Mukherjee and
               Shyamal Kumar Das Mandal},
  title     = {Generation of {F0} contour using deep boltzmann machine and twin Gaussian
               process hybrid model for bengali language},
  booktitle = {{INTERSPEECH} 2014, 15th Annual Conference of the International Speech
               Communication Association, Singapore, September 14-18, 2014},
  pages     = {2445--2449},
  year      = {2014},
  crossref  = {DBLP:conf/interspeech/2014},
  url       = {http://www.isca-speech.org/archive/interspeech_2014/i14_2445.html},
  timestamp = {Wed, 18 Feb 2015 08:38:47 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/conf/interspeech/MukherjeeM14},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```
