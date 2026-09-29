# References

- **Kathpalia A, Karabiyik Y, Eik-Nes SH, Tegnander E, Ekroll IK, Kiss G, Torp H.** Adaptive spectral
  envelope estimation for Doppler ultrasound. *IEEE Trans Ultrason Ferroelectr Freq Control*
  2016;63(11):1825-1838. doi:10.1109/TUFFC.2016.2587381.
  Source for the definitions of GM, MGM and SNSI used here (section II-A) and for the adaptive SNSI
  method `asnsi` (sections II-A, II-B).
- **Li J, Zhang Y, Liu X, Liu P, Yin H, Liu DC.** A robust automatic ultrasound spectral envelope
  estimation. *Information* 2019;10(6):199. doi:10.3390/info10060199. Source for QIA (`qia`).
- **Latham J, Hicks Y, Yang X, Setchi R, Rainer T.** Stable automatic envelope estimation for noisy
  Doppler ultrasound. *IEEE Trans Ultrason Ferroelectr Freq Control* 2021;68(3):465-481.
  doi:10.1109/TUFFC.2020.3011823. Source for OMM (`latham`: section II-B..D, Fig. 5).
- **Sulas E, et al.** Impact of pulsed-wave-Doppler
  velocity-envelope tracing techniques on classification of complete fetal cardiac cycles.
  *PLoS ONE* 2021;16(4):e0248114. doi:10.1371/journal.pone.0248114. Stage decomposition of image-based
  envelope tracing (`segmentation`).
- **Marasek K, Nowicki A.** Comparison of the performance of three maximum Doppler frequency estimators
  coupled with different spectral estimation methods. *Ultrasound Med Biol* 1994;20:629-638. (GM)
- **Moraes R, Aydin N, Evans DH.** The performance of three maximum frequency envelope detection
  algorithms for Doppler signals. *J Vasc Invest* 1995;1:126-134. (MGM)
- **Steinman AH, Tavakkoli J, Myers JG, Cobbold RSC, Johnston KW.** A new approach for determining
  maximum frequency in clinical Doppler ultrasound spectral estimates. *Proc 22nd IEEE EMBS* 2000:2640-2643. (SNSI)
- **Otsu N.** A threshold selection method from gray-level histograms. *IEEE Trans Syst Man Cybern*
  1979;9(1):62-66.
- **Pan J, Tompkins WJ.** A real-time QRS detection algorithm. *IEEE Trans Biomed Eng* 1985;32(3):230-236.
  (R-peak detector, via the MATLAB `DetectRpeaks.m`.)
- **Felzenszwalb PF, Huttenlocher DP.** Distance transforms of sampled functions. *Theory of Computing*
  2012;8:415-428. (O(n) L1 min-convolution used in the DP.)

## Image-domain caveat

All spectrum-based methods (percentile, GM, MGM, SNSI, adaptive SNSI, QIA) are applied to the
**rendered, log-compressed display**, de-compressed with an assumed dynamic range. They are image-domain
adaptations, not reproductions of the signal-domain algorithms. Steps that need the Doppler signal
(Latham's audio filtering and per-window STFTs, Kathpalia's manual baseline shift for aliasing) are
replaced or omitted as documented in the code.
