# Predaq repo
Implementation of plan feature extraction in the paper "Predicting gamma passing rates for portal dosimetry based IMRT QA using machine learning", Medical Physics, 2019. Following features are calculated:


- BA	Beam aperture area weighted by MU
- BI	Beam irregularity
- BM	Fraction of BA normalized by UAA
- UAA	Union area of aperture (UAA)
- MFAS2,5,10,20	Mean of fraction of aperture smaller (MFAS) than 2, 5, 10, 20 mm
- MaxFAS2,5,10,20	Max of fraction of aperture smaller (MaxFAS) than 2, 5, 10, 20 mm
- MAA	Mean aperture area
- MAD	Maximum distance of the mid‐point between any open leaf‐pair in a beam
- MUCP	Mean of MUs per control point in a beam
- MLO1,2,3,4,5	Moment order of 1, 2, 3, 4, 5 of leaf openings
- minAP_h	Minimum aperture perimeter in horizontal direction
- maxAP_h	Maximum aperture perimeter in horizontal direction
- minAP_v	Minimum aperture perimeter in vertical direction
- maxAP_v	Maximum aperture perimeter in vertical direction
- maxRegs	Maximum number of regions in the beam
- AAJA	Ratio of the average area of an aperture over the area defined by jaws
- MAXJ	Maximum of x‐y jaw positions
- MCS	Modulation complexity score
- EM	Edge metric: ratio of MLC side‐length to aperture area

# Usage:
python plan_complexity -i RP.dcm -o output.csv

The program takes a plan as input and outputs a csv file in which each line is extracted features for each beam in the plan  

# Installation
I recommend Anaconda as Python package manager. It comes with numpy. The program needs pydicom package which can be installed by pip: pip install pydicom.

Please cite our paper if you find it useful for your research.  
@article{lam2019predicting,  
  title={Predicting gamma passing rates for portal dosimetry based IMRT QA using machine learning},  
  author={Lam, Dao and Zhang, Xizhe and Li, Harold and Yang, Deshan and Schott, Brayden and Zhao, Tianyu and Zhang, Weixiong and Mutic, Sasa and Sun, Baozhou},  
  journal={Medical physics},  
  year={2019},  
  publisher={Wiley Online Library}  
}
