# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Unittest for atmospheric/ionosphere.py core functionalities"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime

from arepyextras.perturbations.atmospheric import ionosphere as iono

TEST_IONOSPHERE_MAP = """
     1.0            IONOSPHERE MAPS     GNSS                IONEX VERSION / TYPE
ADDNEQ2 V5.3        AIUB                09-JAN-19 07:03     PGM / RUN BY / DATE 
CODE'S RAPID IONOSPHERE MAPS FOR DAY 008, 2019              COMMENT             
Global ionosphere maps (GIM) are generated on a daily basis DESCRIPTION         
at CODE using data from about 120 GNSS sites of the IGS and DESCRIPTION         
other institutions. The vertical total electron content     DESCRIPTION         
(VTEC) is modeled in a solar-geomagnetic reference frame    DESCRIPTION         
using a spherical harmonics expansion up to degree and      DESCRIPTION         
order 15. Piece-wise linear functions are used for          DESCRIPTION         
representation in the time domain. The time spacing of      DESCRIPTION         
their vertices is 1 hour, conforming with the epochs of the DESCRIPTION         
VTEC maps.                                                  DESCRIPTION         
To convert line-of-sight TEC into vertical TEC, a modified  DESCRIPTION         
single-layer model mapping function approximating the JPL   DESCRIPTION         
extended slab model mapping function is adopted. The        DESCRIPTION         
mapping function is evaluated at geodetic satellite         DESCRIPTION         
elevation angles. For the computation of the ionospheric    DESCRIPTION         
pierce points, a spherical layer with a radius of 6821 km   DESCRIPTION         
is assumed, implying geocentric, not geodetic IONEX         DESCRIPTION         
latitudes.                                                  DESCRIPTION         
Instrumental biases with respect to each involved GNSS code DESCRIPTION         
observable are estimated as constant values for each        DESCRIPTION         
day. They are modeled with one set of bias parameters for   DESCRIPTION         
each satellite and each station in case of GPS and for each DESCRIPTION         
satellite-station link in case of GLONASS. The code bias    DESCRIPTION         
datum is defined by zero-mean conditions imposed on the     DESCRIPTION         
satellite bias estimates. In this file, just GPS C1W-C2W    DESCRIPTION         
satellite and station bias values are included. The         DESCRIPTION         
complete set of GNSS code bias results is provided in an    DESCRIPTION         
additional Bias-SINEX file as pseudo-absolute values        DESCRIPTION         
specific to each observable.                                DESCRIPTION         
Note: These GIM results correspond to the results for the   DESCRIPTION         
last day of a 3-day combination analysis solving for 73     DESCRIPTION         
times 256, or 18688 VTEC parameters and 3 daily sets of     DESCRIPTION         
GNSS code bias parameters.                                  DESCRIPTION         
Contact address: code(at)aiub.unibe.ch                      DESCRIPTION         
Data archive:    ftp.aiub.unibe.ch/CODE/                    DESCRIPTION         
                 www.aiub.unibe.ch/download/CODE/           DESCRIPTION         
  2019     1     8     0     0     0                        EPOCH OF FIRST MAP  
  2019     1     9     0     0     0                        EPOCH OF LAST MAP   
  3600                                                      INTERVAL            
    25                                                      # OF MAPS IN FILE   
  NONE                                                      MAPPING FUNCTION    
    10.0                                                    ELEVATION CUTOFF    
One-way carrier phase leveled to code                       OBSERVABLES USED    
   118                                                      # OF STATIONS       
    53                                                      # OF SATELLITES     
  6371.0                                                    BASE RADIUS         
     2                                                      MAP DIMENSION       
   450.0 450.0   0.0                                        HGT1 / HGT2 / DHGT  
    87.5 -87.5  -2.5                                        LAT1 / LAT2 / DLAT  
  -180.0 180.0   5.0                                        LON1 / LON2 / DLON  
    -1                                                      EXPONENT            
TEC/RMS values in 0.1 TECU; 9999, if no value available     COMMENT             
Peak TEC values for the included maps:                      COMMENT             
 356  344  331  321  312  303  316  322  315  295  276  261 COMMENT             
 258  262  266  266  259  253  256  266  279  302  323  343 COMMENT             
 357                                                        COMMENT             
List of stations:                                           COMMENT             
albh algo alic alrt ankr areq auck badg bake bako bjco bjfs COMMENT             
brmu brst brux bshm cas1 chpi chti chur coco cord cro1 darw COMMENT             
dav1 dubo faa1 fair flin flrs frdn func ganp glps glsv gode COMMENT             
godz gope gras guam guat hert hob2 hofn hrao hyde iisc iqal COMMENT             
isba joz2 kerg kir0 kit3 kokb kokv kour lama lhaz lpal lpgs COMMENT             
mac1 mas1 mat1 mate maw1 mcm4 mdvj mizu mkea mobj mobs mtka COMMENT             
nano nklg not1 nril ntus ohi3 onsa park pdel penc pert pfa2 COMMENT             
pimo pots qaq1 rabt rbay reun reyk rio2 sant sch2 scub sofi COMMENT             
stjo str2 suth sydn tabv tash thti thu2 tid1 tow2 ulab unb3 COMMENT             
unbj unsa vald whit wind xmis yarr yssk zim2 zimm           COMMENT             
DIFFERENTIAL CODE BIASES                                    START OF AUX DATA   
Reference observables for GPS    : C1W-C2W                  COMMENT             
   G01    -7.722     0.020                                  PRN / BIAS / RMS    
   G02     9.359     0.020                                  PRN / BIAS / RMS    
   G03    -5.355     0.020                                  PRN / BIAS / RMS    
   G05     3.329     0.020                                  PRN / BIAS / RMS    
   G06    -6.929     0.020                                  PRN / BIAS / RMS    
   G07     3.269     0.019                                  PRN / BIAS / RMS    
   G08    -7.416     0.020                                  PRN / BIAS / RMS    
   G09    -4.991     0.020                                  PRN / BIAS / RMS    
   G10    -5.253     0.019                                  PRN / BIAS / RMS    
   G11     3.834     0.020                                  PRN / BIAS / RMS    
   G12     4.008     0.020                                  PRN / BIAS / RMS    
   G13     3.224     0.020                                  PRN / BIAS / RMS    
   G14     2.238     0.019                                  PRN / BIAS / RMS    
   G15     3.197     0.020                                  PRN / BIAS / RMS    
   G16     3.037     0.020                                  PRN / BIAS / RMS    
   G17     3.247     0.019                                  PRN / BIAS / RMS    
   G18    -0.397     0.020                                  PRN / BIAS / RMS    
   G19     6.079     0.019                                  PRN / BIAS / RMS    
   G20     1.737     0.019                                  PRN / BIAS / RMS    
   G21     2.671     0.019                                  PRN / BIAS / RMS    
   G22     7.787     0.020                                  PRN / BIAS / RMS    
   G23     9.044     0.019                                  PRN / BIAS / RMS    
   G24    -5.897     0.020                                  PRN / BIAS / RMS    
   G25    -7.785     0.020                                  PRN / BIAS / RMS    
   G26    -8.833     0.020                                  PRN / BIAS / RMS    
   G27    -5.175     0.020                                  PRN / BIAS / RMS    
   G28     3.185     0.019                                  PRN / BIAS / RMS    
   G29     2.638     0.020                                  PRN / BIAS / RMS    
   G30    -6.437     0.019                                  PRN / BIAS / RMS    
   G31     4.709     0.020                                  PRN / BIAS / RMS    
   G32    -4.401     0.019                                  PRN / BIAS / RMS    
Reference observables for ALBH 40129M003 (G): C1W-C2W       COMMENT             
   G  ALBH 40129M003          13.899     0.079              STATION / BIAS / RMS
Reference observables for ALGO 40104M002 (G): C1W-C2W       COMMENT             
   G  ALGO 40104M002           2.226     0.066              STATION / BIAS / RMS
Reference observables for ALIC 50137M001 (G): C1C-C2W       COMMENT             
   G  ALIC 50137M001          21.005     0.071              STATION / BIAS / RMS
Reference observables for ALRT 40162M001 (G): C1W-C2W       COMMENT             
   G  ALRT 40162M001           1.129     0.109              STATION / BIAS / RMS
Reference observables for ANKR 20805M002 (G): C1C-C2W       COMMENT             
   G  ANKR 20805M002           7.971     0.061              STATION / BIAS / RMS
Reference observables for AREQ 42202M005 (G): C1W-C2W       COMMENT             
   G  AREQ 42202M005           7.860     0.084              STATION / BIAS / RMS
Reference observables for AUCK 50209M001 (G): C1C-C2W       COMMENT             
   G  AUCK 50209M001          -8.404     0.081              STATION / BIAS / RMS
Reference observables for BADG 12338M002 (G): C1W-C2W       COMMENT             
   G  BADG 12338M002           7.079     0.081              STATION / BIAS / RMS
Reference observables for BAKE 40152M001 (G): C1W-C2W       COMMENT             
   G  BAKE 40152M001           7.096     0.075              STATION / BIAS / RMS
Reference observables for BAKO 23101M002 (G): C1C-C2W       COMMENT             
   G  BAKO 23101M002           3.477     0.078              STATION / BIAS / RMS
Reference observables for BJCO 32701M001 (G): C1C-C2W       COMMENT             
   G  BJCO 32701M001         -17.428     0.092              STATION / BIAS / RMS
Reference observables for BJFS 21601M001 (G): C1C-C2W       COMMENT             
   G  BJFS 21601M001         -13.916     0.075              STATION / BIAS / RMS
Reference observables for BRMU 42501S004 (G): C1C-C2W       COMMENT             
   G  BRMU 42501S004          16.677     0.067              STATION / BIAS / RMS
Reference observables for BRST 10004M004 (G): C1C-C2W       COMMENT             
   G  BRST 10004M004          -9.236     0.061              STATION / BIAS / RMS
Reference observables for BRUX 13101M010 (G): C1W-C2W       COMMENT             
   G  BRUX 13101M010           5.873     0.060              STATION / BIAS / RMS
Reference observables for BSHM 20705M001 (G): C1W-C2W       COMMENT             
   G  BSHM 20705M001           6.432     0.069              STATION / BIAS / RMS
Reference observables for CAS1 66011M001 (G): C1C-C2W       COMMENT             
   G  CAS1 66011M001         -20.483     0.122              STATION / BIAS / RMS
Reference observables for CHPI 41609M003 (G): C1W-C2W       COMMENT             
   G  CHPI 41609M003          -2.273     0.084              STATION / BIAS / RMS
Reference observables for CHTI 50242M001 (G): C1C-C2W       COMMENT             
   G  CHTI 50242M001          -9.987     0.087              STATION / BIAS / RMS
Reference observables for CHUR 40128M002 (G): C1W-C2W       COMMENT             
   G  CHUR 40128M002          10.723     0.071              STATION / BIAS / RMS
Reference observables for COCO 50127M001 (G): C1W-C2W       COMMENT             
   G  COCO 50127M001          -0.411     0.080              STATION / BIAS / RMS
Reference observables for CORD 41511M001 (G): C1W-C2W       COMMENT             
   G  CORD 41511M001          -4.867     0.080              STATION / BIAS / RMS
Reference observables for CRO1 43201M001 (G): C1W-C2W       COMMENT             
   G  CRO1 43201M001          10.362     0.080              STATION / BIAS / RMS
Reference observables for DARW 50134M001 (G): C1W-C2W       COMMENT             
   G  DARW 50134M001          10.540     0.079              STATION / BIAS / RMS
Reference observables for DAV1 66010M001 (G): C1W-C2W       COMMENT             
   G  DAV1 66010M001           7.685     0.118              STATION / BIAS / RMS
Reference observables for DUBO 40137M001 (G): C1W-C2W       COMMENT             
   G  DUBO 40137M001           3.857     0.068              STATION / BIAS / RMS
Reference observables for FAA1 92201M012 (G): C1W-C2W       COMMENT             
   G  FAA1 92201M012           5.451     0.101              STATION / BIAS / RMS
Reference observables for FAIR 40408M001 (G): C1W-C2W       COMMENT             
   G  FAIR 40408M001          -3.625     0.098              STATION / BIAS / RMS
Reference observables for FLIN 40135M001 (G): C1W-C2W       COMMENT             
   G  FLIN 40135M001          15.693     0.070              STATION / BIAS / RMS
Reference observables for FLRS 31907M001 (G): C1C-C2W       COMMENT             
   G  FLRS 31907M001           7.412     0.070              STATION / BIAS / RMS
Reference observables for FRDN 40146M001 (G): C1W-C2W       COMMENT             
   G  FRDN 40146M001           3.689     0.066              STATION / BIAS / RMS
Reference observables for FUNC 13911S001 (G): C1C-C2W       COMMENT             
   G  FUNC 13911S001           7.823     0.068              STATION / BIAS / RMS
Reference observables for GANP 11515M001 (G): C1C-C2W       COMMENT             
   G  GANP 11515M001         -14.209     0.060              STATION / BIAS / RMS
Reference observables for GLPS 42005M002 (G): C1W-C2W       COMMENT             
   G  GLPS 42005M002           0.668     0.096              STATION / BIAS / RMS
Reference observables for GLSV 12356M001 (G): C1C-C2W       COMMENT             
   G  GLSV 12356M001           7.886     0.061              STATION / BIAS / RMS
Reference observables for GODE 40451M123 (G): C1W-C2W       COMMENT             
   G  GODE 40451M123           2.830     0.066              STATION / BIAS / RMS
Reference observables for GODZ 40451M123 (G): C1W-C2W       COMMENT             
   G  GODZ 40451M123          -3.662     0.066              STATION / BIAS / RMS
Reference observables for GOPE 11502M002 (G): C1C-C2W       COMMENT             
   G  GOPE 11502M002         -22.096     0.060              STATION / BIAS / RMS
Reference observables for GRAS 10002M006 (G): C1C-C2W       COMMENT             
   G  GRAS 10002M006         -19.434     0.059              STATION / BIAS / RMS
Reference observables for GUAM 50501M002 (G): C1W-C2W       COMMENT             
   G  GUAM 50501M002           1.126     0.101              STATION / BIAS / RMS
Reference observables for GUAT 40901S001 (G): C1C-C2W       COMMENT             
   G  GUAT 40901S001          11.519     0.089              STATION / BIAS / RMS
Reference observables for HERT 13212M010 (G): C1C-C2W       COMMENT             
   G  HERT 13212M010           6.484     0.061              STATION / BIAS / RMS
Reference observables for HOB2 50116M004 (G): C1W-C2W       COMMENT             
   G  HOB2 50116M004           2.262     0.072              STATION / BIAS / RMS
Reference observables for HOFN 10204M002 (G): C1C-C2W       COMMENT             
   G  HOFN 10204M002          23.955     0.077              STATION / BIAS / RMS
Reference observables for HRAO 30302M004 (G): C1W-C2W       COMMENT             
   G  HRAO 30302M004           0.403     0.083              STATION / BIAS / RMS
Reference observables for HYDE 22307M001 (G): C1C-C2W       COMMENT             
   G  HYDE 22307M001           6.743     0.086              STATION / BIAS / RMS
Reference observables for IISC 22306M002 (G): C1W-C2W       COMMENT             
   G  IISC 22306M002           4.889     0.089              STATION / BIAS / RMS
Reference observables for IQAL 40194M001 (G): C1W-C2W       COMMENT             
   G  IQAL 40194M001           5.485     0.073              STATION / BIAS / RMS
Reference observables for ISBA 20308M001 (G): C1C-C2W       COMMENT             
   G  ISBA 20308M001         -10.853     0.071              STATION / BIAS / RMS
Reference observables for JOZ2 12204M002 (G): C1C-C2W       COMMENT             
   G  JOZ2 12204M002           9.215     0.061              STATION / BIAS / RMS
Reference observables for KERG 91201M002 (G): C1C-C2W       COMMENT             
   G  KERG 91201M002         -14.869     0.116              STATION / BIAS / RMS
Reference observables for KIR0 10422M001 (G): C1W-C2W       COMMENT             
   G  KIR0 10422M001          -1.385     0.076              STATION / BIAS / RMS
Reference observables for KIT3 12334M001 (G): C1W-C2W       COMMENT             
   G  KIT3 12334M001           6.384     0.076              STATION / BIAS / RMS
Reference observables for KOKB 40424M004 (G): C1W-C2W       COMMENT             
   G  KOKB 40424M004          -6.294     0.097              STATION / BIAS / RMS
Reference observables for KOKV 40424M004 (G): C1W-C2W       COMMENT             
   G  KOKV 40424M004          -1.725     0.097              STATION / BIAS / RMS
Reference observables for KOUR 97301M210 (G): C1W-C2W       COMMENT             
   G  KOUR 97301M210           2.893     0.090              STATION / BIAS / RMS
Reference observables for LAMA 12209M001 (G): C1C-C2W       COMMENT             
   G  LAMA 12209M001          12.656     0.086              STATION / BIAS / RMS
Reference observables for LHAZ 21613M002 (G): C1C-C2W       COMMENT             
   G  LHAZ 21613M002          13.811     0.077              STATION / BIAS / RMS
Reference observables for LPAL 81701M001 (G): C1C-C2W       COMMENT             
   G  LPAL 81701M001           9.778     0.071              STATION / BIAS / RMS
Reference observables for LPGS 41510M001 (G): C1W-C2W       COMMENT             
   G  LPGS 41510M001           5.370     0.081              STATION / BIAS / RMS
Reference observables for MAC1 50135M001 (G): C1W-C2W       COMMENT             
   G  MAC1 50135M001           9.516     0.098              STATION / BIAS / RMS
Reference observables for MAS1 31303M002 (G): C1W-C2W       COMMENT             
   G  MAS1 31303M002           5.106     0.071              STATION / BIAS / RMS
Reference observables for MAT1 12734M009 (G): C1C-C2W       COMMENT             
   G  MAT1 12734M009           6.104     0.059              STATION / BIAS / RMS
Reference observables for MATE 12734M008 (G): C1C-C2W       COMMENT             
   G  MATE 12734M008           7.614     0.059              STATION / BIAS / RMS
Reference observables for MAW1 66004M001 (G): C1W-C2W       COMMENT             
   G  MAW1 66004M001           2.447     0.119              STATION / BIAS / RMS
Reference observables for MCM4 66001M003 (G): C1W-C2W       COMMENT             
   G  MCM4 66001M003          -0.840     0.162              STATION / BIAS / RMS
Reference observables for MDVJ 12309M005 (G): C1W-C2W       COMMENT             
   G  MDVJ 12309M005          10.647     0.065              STATION / BIAS / RMS
Reference observables for MIZU 21702M002 (G): C1W-C2W       COMMENT             
   G  MIZU 21702M002           5.720     0.080              STATION / BIAS / RMS
Reference observables for MKEA 40477M001 (G): C1W-C2W       COMMENT             
   G  MKEA 40477M001          10.952     0.097              STATION / BIAS / RMS
Reference observables for MOBJ 12365M002 (G): C1W-C2W       COMMENT             
   G  MOBJ 12365M002          -8.777     0.065              STATION / BIAS / RMS
Reference observables for MOBS 50182M001 (G): C1W-C2W       COMMENT             
   G  MOBS 50182M001          -0.950     0.073              STATION / BIAS / RMS
Reference observables for MTKA 21741S002 (G): C1C-C2W       COMMENT             
   G  MTKA 21741S002         -20.663     0.081              STATION / BIAS / RMS
Reference observables for NANO 40138M001 (G): C1W-C2W       COMMENT             
   G  NANO 40138M001           6.204     0.079              STATION / BIAS / RMS
Reference observables for NKLG 32809M002 (G): C1C-C2W       COMMENT             
   G  NKLG 32809M002         -13.910     0.092              STATION / BIAS / RMS
Reference observables for NOT1 12717M004 (G): C1C-C2W       COMMENT             
   G  NOT1 12717M004           5.805     0.060              STATION / BIAS / RMS
Reference observables for NRIL 12364M001 (G): C1W-C2W       COMMENT             
   G  NRIL 12364M001          -4.511     0.109              STATION / BIAS / RMS
Reference observables for NTUS 22601M001 (G): C1C-C2W       COMMENT             
   G  NTUS 22601M001           4.589     0.077              STATION / BIAS / RMS
Reference observables for OHI3 66008M006 (G): C1C-C2W       COMMENT             
   G  OHI3 66008M006          18.401     0.108              STATION / BIAS / RMS
Reference observables for ONSA 10402M004 (G): C1W-C2W       COMMENT             
   G  ONSA 10402M004           2.499     0.061              STATION / BIAS / RMS
Reference observables for PARK 50108M001 (G): C1C-C2W       COMMENT             
   G  PARK 50108M001         -16.176     0.073              STATION / BIAS / RMS
Reference observables for PDEL 31906M004 (G): C1C-C2W       COMMENT             
   G  PDEL 31906M004           8.367     0.069              STATION / BIAS / RMS
Reference observables for PENC 11206M006 (G): C1C-C2W       COMMENT             
   G  PENC 11206M006           8.840     0.059              STATION / BIAS / RMS
Reference observables for PERT 50133M001 (G): C1C-C2W       COMMENT             
   G  PERT 50133M001         -19.535     0.083              STATION / BIAS / RMS
Reference observables for PFA2 11005M003 (G): C1C-C2W       COMMENT             
   G  PFA2 11005M003          13.221     0.060              STATION / BIAS / RMS
Reference observables for PIMO 22003M001 (G): C1W-C2W       COMMENT             
   G  PIMO 22003M001           4.294     0.093              STATION / BIAS / RMS
Reference observables for POTS 14106M003 (G): C1W-C2W       COMMENT             
   G  POTS 14106M003           7.985     0.061              STATION / BIAS / RMS
Reference observables for QAQ1 43007M001 (G): C1W-C2W       COMMENT             
   G  QAQ1 43007M001          -0.188     0.074              STATION / BIAS / RMS
Reference observables for RABT 35001M002 (G): C1W-C2W       COMMENT             
   G  RABT 35001M002          -1.844     0.065              STATION / BIAS / RMS
Reference observables for RBAY 30315M001 (G): C1C-C2W       COMMENT             
   G  RBAY 30315M001         -14.744     0.083              STATION / BIAS / RMS
Reference observables for REUN 97401M003 (G): C1C-C2W       COMMENT             
   G  REUN 97401M003          -5.122     0.089              STATION / BIAS / RMS
Reference observables for REYK 10202M001 (G): C1C-C2W       COMMENT             
   G  REYK 10202M001          20.407     0.078              STATION / BIAS / RMS
Reference observables for RIO2 41507M006 (G): C1W-C2W       COMMENT             
   G  RIO2 41507M006           5.799     0.101              STATION / BIAS / RMS
Reference observables for SANT 41705M003 (G): C1W-C2W       COMMENT             
   G  SANT 41705M003           8.351     0.082              STATION / BIAS / RMS
Reference observables for SCH2 40133M002 (G): C1W-C2W       COMMENT             
   G  SCH2 40133M002          10.020     0.066              STATION / BIAS / RMS
Reference observables for SCUB 40701M001 (G): C1W-C2W       COMMENT             
   G  SCUB 40701M001           3.929     0.080              STATION / BIAS / RMS
Reference observables for SOFI 11101M002 (G): C1C-C2W       COMMENT             
   G  SOFI 11101M002          13.587     0.059              STATION / BIAS / RMS
Reference observables for STJO 40101M001 (G): C1W-C2W       COMMENT             
   G  STJO 40101M001           4.240     0.067              STATION / BIAS / RMS
Reference observables for STR2 50119M001 (G): C1C-C2W       COMMENT             
   G  STR2 50119M001         -21.561     0.073              STATION / BIAS / RMS
Reference observables for SUTH 30314M002 (G): C1W-C2W       COMMENT             
   G  SUTH 30314M002          -0.307     0.083              STATION / BIAS / RMS
Reference observables for SYDN 50124M003 (G): C1W-C2W       COMMENT             
   G  SYDN 50124M003           0.967     0.073              STATION / BIAS / RMS
Reference observables for TABV 49901M001 (G): C1W-C2W       COMMENT             
   G  TABV 49901M001          10.263     0.086              STATION / BIAS / RMS
Reference observables for TASH 12327M001 (G): C1W-C2W       COMMENT             
   G  TASH 12327M001           7.151     0.076              STATION / BIAS / RMS
Reference observables for THTI 92201M009 (G): C1C-C2W       COMMENT             
   G  THTI 92201M009         -19.652     0.101              STATION / BIAS / RMS
Reference observables for THU2 43001M002 (G): C1C-C2W       COMMENT             
   G  THU2 43001M002          13.537     0.097              STATION / BIAS / RMS
Reference observables for TID1 50103M108 (G): C1W-C2W       COMMENT             
   G  TID1 50103M108           4.116     0.073              STATION / BIAS / RMS
Reference observables for TOW2 50140M001 (G): C1W-C2W       COMMENT             
   G  TOW2 50140M001           3.689     0.077              STATION / BIAS / RMS
Reference observables for ULAB 24201M001 (G): C1W-C2W       COMMENT             
   G  ULAB 24201M001          10.063     0.078              STATION / BIAS / RMS
Reference observables for UNB3 40146M002 (G): C1C-C2W       COMMENT             
   G  UNB3 40146M002         -12.037     0.066              STATION / BIAS / RMS
Reference observables for UNBJ 40146M002 (G): C1W-C2W       COMMENT             
   G  UNBJ 40146M002           5.498     0.066              STATION / BIAS / RMS
Reference observables for UNSA 41514M001 (G): C1W-C2W       COMMENT             
   G  UNSA 41514M001           3.154     0.076              STATION / BIAS / RMS
Reference observables for VALD 40156M001 (G): C1W-C2W       COMMENT             
   G  VALD 40156M001          -4.475     0.066              STATION / BIAS / RMS
Reference observables for WHIT 40136M001 (G): C1W-C2W       COMMENT             
   G  WHIT 40136M001          13.980     0.087              STATION / BIAS / RMS
Reference observables for WIND 31101M001 (G): C1W-C2W       COMMENT             
   G  WIND 31101M001           6.168     0.082              STATION / BIAS / RMS
Reference observables for XMIS 50183M001 (G): C1C-C2W       COMMENT             
   G  XMIS 50183M001         -18.135     0.078              STATION / BIAS / RMS
Reference observables for YARR 50107M006 (G): C1W-C2W       COMMENT             
   G  YARR 50107M006           1.369     0.082              STATION / BIAS / RMS
Reference observables for YSSK 12329M003 (G): C1W-C2W       COMMENT             
   G  YSSK 12329M003         -14.155     0.081              STATION / BIAS / RMS
Reference observables for ZIM2 14001M008 (G): C1C-C2W       COMMENT             
   G  ZIM2 14001M008         -13.174     0.060              STATION / BIAS / RMS
Reference observables for ZIMM 14001M004 (G): C1C-C2W       COMMENT             
   G  ZIMM 14001M004         -11.996     0.060              STATION / BIAS / RMS
DCB values in ns; zero-mean condition wrt satellite values  COMMENT             
DIFFERENTIAL CODE BIASES                                    END OF AUX DATA     
                                                            END OF HEADER       
     1                                                      START OF TEC MAP    
  2019     1     8     8     0     0                        EPOCH OF CURRENT MAP
    87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    8    8    8    8    8    8    8    9    9    9    9    9    9    9    9    9
    9    9    9    8    8    8    8    8    8    8    8    8    9    9    9    9
    9    9   10   10   10   10   10   11   11   11   11   11   11   11   11   11
   11   11   10   10   10   10   10    9    9    9    9    9    8    8    8    8
    8    8    8    8    8    8    8    8    8
    85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    8    8    9    9    9   10   10   10   11   11   11   11   11   11   11   10
   10   10    9    9    8    8    8    7    7    7    7    7    7    8    8    9
    9   10   11   11   12   12   13   14   14   14   14   15   15   15   15   14
   14   14   13   13   12   12   11   11   10   10    9    8    8    8    7    7
    7    7    7    7    7    7    7    7    8
    82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    9    9   10   11   12   12   13   13   14   14   15   15   15   14   14   13
   13   12   11   10    9    8    7    6    5    5    5    5    6    6    7    8
    9   11   12   13   14   15   16   17   18   18   19   19   19   19   19   18
   18   17   17   16   15   14   14   13   12   11   10    9    8    7    7    6
    6    6    6    6    6    7    7    8    9
    80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   11   12   13   14   15   15   16   17   17   18   18   18   18   18   18   17
   16   14   13   11    9    8    6    5    4    3    3    3    4    5    6    8
   10   12   14   16   17   19   20   21   21   22   22   22   22   22   21   21
   21   20   20   19   18   17   16   15   14   13   12   10    9    9    8    7
    7    7    7    7    8    9    9   10   11
    77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   15   16   17   18   18   19   20   20   21   21   21   21   21   21   21   20
   18   17   15   13   11    9    6    4    3    2    1    2    3    4    6    9
   11   14   16   18   20   21   22   23   23   23   23   23   22   22   22   22
   22   21   21   21   20   19   19   18   16   15   14   13   12   11   10   10
   10   10   10   11   12   13   13   14   15
    75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   21   22   22   22   23   23   23   23   23   23   23   23   23   23   22   22
   20   19   17   15   13   10    8    5    3    2    1    2    3    5    7   10
   13   16   18   20   21   22   23   22   22   21   21   20   20   20   20   20
   21   21   21   21   21   21   20   20   19   18   17   16   15   15   14   14
   14   15   16   17   18   19   20   20   21
    72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   28   28   28   28   27   26   26   25   24   24   24   23   23   23   22   22
   21   20   19   17   15   13   10    8    5    4    3    3    4    6    8   11
   14   17   19   20   21   21   20   19   18   17   16   16   16   16   16   17
   18   19   20   21   21   22   21   21   21   20   20   19   19   19   19   19
   20   21   22   23   25   26   27   28   28
    70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   35   35   34   33   32   30   29   27   26   25   24   23   22   22   21   21
   20   20   19   18   17   15   13   11    8    6    5    5    5    7    9   11
   14   16   17   18   17   16   15   14   12   11   10   10   10   11   12   14
   15   17   18   19   20   21   21   21   21   21   21   21   21   22   23   24
   25   27   29   31   32   33   34   35   35
    67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   41   41   40   38   36   34   32   30   28   26   25   23   22   21   20   19
   19   19   19   19   18   17   16   14   11    9    7    6    6    7    8   10
   11   12   13   13   12   10    8    7    5    4    4    5    6    7    9   11
   13   15   16   18   18   19   20   20   20   20   21   21   22   24   25   28
   30   32   34   37   38   40   41   41   41
    65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   46   45   44   43   41   38   35   33   31   29   27   25   23   21   19   17
   16   16   17   18   18   18   17   16   13   11    9    7    6    6    6    7
    7    7    7    6    5    3    2    1    0    0    1    2    4    6    8   10
   12   14   15   16   16   17   17   17   18   18   19   20   22   24   27   30
   33   36   39   41   43   44   45   46   46
    62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   49   49   48   47   45   42   39   37   34   32   30   28   25   22   19   16
   15   14   15   16   17   17   17   16   14   11    9    7    5    4    4    3
    2    2    1    0    0    0    0    0    0    0    1    3    6    8   10   12
   13   14   14   14   14   14   14   14   14   15   16   18   20   23   27   31
   35   38   41   44   46   47   48   49   49
    60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   52   52   52   51   49   46   44   41   38   36   34   32   29   25   21   17
   14   13   13   14   15   15   15   14   12   10    8    6    4    2    1    0
    0    0    0    0    0    0    0    0    1    3    6    9   11   14   15   16
   16   15   14   13   12   11   11   10   11   12   13   16   19   23   27   31
   36   40   43   46   48   49   50   51   52
    57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   55   56   57   56   54   52   48   45   43   41   39   37   34   30   25   20
   16   13   12   12   12   12   12   11    9    8    6    4    3    1    0    0
    0    0    0    0    0    0    2    5    8   12   15   18   20   21   22   21
   20   18   16   14   12   10    9    9    9   10   12   15   19   23   28   33
   38   42   46   48   50   52   53   54   55
    55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   60   62   63   63   61   58   54   50   47   44   43   41   38   35   30   24
   19   15   13   12   11   10    8    7    6    5    5    4    3    2    1    0
    0    0    0    1    4    8   12   16   20   24   26   29   30   30   29   27
   25   22   18   15   13   11   10   10   10   12   14   17   21   26   31   36
   41   45   49   51   53   55   56   58   60
    52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   67   69   71   71   70   66   61   56   51   48   46   44   42   39   35   30
   24   20   16   13   11    9    7    5    5    5    5    6    6    6    6    5
    5    6    8   11   15   20   25   30   34   37   39   39   39   38   36   33
   30   26   22   19   16   14   13   14   15   17   20   23   27   32   37   42
   46   50   53   56   58   60   62   64   67
    50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   74   77   80   80   79   75   69   62   56   51   48   46   45   43   40   36
   31   26   22   18   14   10    7    6    6    7    9   12   13   14   14   14
   15   17   20   24   30   35   40   45   48   49   50   49   47   45   42   39
   35   31   27   24   22   20   20   21   22   25   29   33   37   41   45   49
   53   56   59   61   63   65   68   71   74
    47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   81   85   87   88   87   84   78   70   62   56   51   48   47   46   44   42
   38   34   29   24   20   15   12   10   11   14   17   20   23   24   25   26
   28   30   34   39   44   50   54   57   59   59   58   56   53   50   47   43
   39   36   32   30   29   28   28   30   33   36   40   44   49   52   56   59
   61   63   65   67   69   71   74   78   81
    45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   86   89   92   93   93   91   86   79   70   61   55   51   50   49   49   47
   45   42   38   33   28   23   20   19   20   24   28   32   35   36   37   38
   40   43   47   52   57   62   65   67   67   65   63   60   57   54   51   48
   44   41   39   37   37   37   38   40   44   48   53   57   61   65   67   69
   70   71   71   73   74   77   80   83   86
    42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   89   91   92   94   95   95   92   87   78   69   61   55   53   53   53   53
   52   49   46   43   38   34   31   30   33   37   41   45   47   48   48   49
   51   54   58   62   66   70   72   72   71   68   65   62   60   57   54   51
   49   46   45   44   45   45   47   49   54   59   64   69   73   76   78   79
   79   78   78   78   79   82   84   87   89
    40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   89   89   89   90   92   94   95   93   86   77   68   62   59   58   59   59
   58   56   54   52   48   45   43   43   46   50   54   57   58   58   57   57
   58   61   64   68   71   74   74   74   71   69   66   63   61   59   57   55
   52   51   50   51   51   52   54   57   61   67   73   79   83   86   88   88
   87   86   85   85   86   87   88   89   89
    37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   90   87   84   83   85   89   94   96   93   86   77   70   66   65   65   64
   63   62   61   60   58   56   55   56   58   61   64   66   66   65   63   62
   63   64   67   70   73   74   74   72   70   67   65   63   62   61   59   57
   55   54   54   55   56   56   58   61   66   72   79   85   90   93   95   96
   96   95   94   94   94   94   94   93   90
    35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   95   88   81   77   77   82   90   96   97   92   85   78   74   72   71   70
   68   67   66   66   66   65   65   65   67   69   71   72   72   70   67   65
   65   65   68   70   72   72   71   70   68   66   64   63   62   62   60   58
   56   55   56   57   58   58   58   61   66   73   81   88   93   98  102  104
  106  106  107  107  107  107  105  101   95
    32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  108   97   86   76   73   76   84   93   97   96   91   85   81   80   78   76
   73   70   69   70   71   71   71   71   72   73   74   75   74   73   70   67
   65   65   67   69   70   70   69   67   65   64   63   62   62   61   59   57
   55   54   55   56   57   56   56   58   63   71   80   88   94  101  107  112
  116  120  123  125  126  125  122  116  108
    30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  130  117  101   87   77   75   81   90   96   97   94   90   87   86   84   81
   76   73   71   72   74   75   74   73   72   72   73   74   75   74   72   68
   66   65   66   68   68   67   66   64   63   62   61   60   60   58   56   53
   51   50   51   53   53   52   51   53   59   68   77   86   95  103  112  120
  129  136  143  148  151  151  147  140  130
    27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  162  148  130  110   93   84   84   90   95   97   95   93   91   91   89   85
   79   74   72   73   75   76   75   72   70   69   70   72   74   75   73   70
   67   66   67   67   67   66   64   62   61   60   58   57   56   54   51   48
   45   45   46   48   48   46   45   47   54   64   75   85   95  105  117  129
  142  153  164  174  179  181  178  172  162
    25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  200  188  169  145  121  104   96   96   98   98   96   95   94   94   93   88
   82   76   73   73   75   76   74   70   67   65   66   70   74   77   76   73
   69   68   67   67   66   64   61   59   58   56   55   53   51   49   45   41
   39   39   40   42   42   40   39   42   50   61   74   85   97  109  123  139
  154  170  185  199  208  212  211  207  200
    22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  238  231  214  189  160  134  118  110  106  103  100   97   96   96   95   91
   84   77   73   73   75   76   74   69   65   63   65   70   75   79   79   76
   72   70   69   68   65   62   58   56   54   53   51   48   45   42   39   35
   33   33   35   37   37   36   35   39   48   61   75   88  101  115  130  147
  165  184  202  219  231  238  240  240  238
    20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  268  268  257  234  203  172  148  133  123  115  108  103  100   99   97   93
   86   79   74   74   76   77   75   71   67   66   68   73   78   81   81   78
   74   71   69   67   64   59   55   52   50   48   46   43   40   37   33   30
   28   28   31   33   33   32   33   39   50   64   79   93  107  121  137  155
  174  193  213  232  246  255  260  264  268
    17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  285  292  289  273  244  212  184  162  146  133  122  113  107  104  100   96
   88   81   76   75   77   79   79   76   74   73   74   78   82   84   83   79
   75   71   69   66   61   56   51   48   46   44   42   38   35   32   29   27
   25   26   28   30   30   31   34   41   53   69   85  100  114  128  144  161
  179  197  216  235  250  260  267  275  285
    15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  284  299  305  297  275  246  218  195  175  158  142  128  118  111  105  100
   92   84   79   78   80   83   85   85   84   83   84   85   86   85   83   79
   74   71   68   64   58   53   48   45   43   41   38   35   32   30   28   26
   25   25   27   28   30   31   36   45   59   75   91  106  121  136  151  166
  180  196  212  229  242  252  259  270  284
    12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  268  288  302  303  291  270  246  225  204  185  165  148  133  121  113  105
   97   89   83   81   84   89   93   95   96   96   95   93   90   86   82   77
   73   69   66   61   56   50   45   42   41   39   37   34   31   29   28   26
   26   26   27   28   30   33   40   51   65   81   97  112  127  142  156  169
  181  192  204  217  228  235  241  251  268
    10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  240  262  281  291  289  279  264  247  229  210  189  169  150  134  123  113
  104   95   88   86   89   95  101  106  109  109  106  100   93   86   80   75
   71   68   64   59   54   48   44   41   40   38   36   34   32   30   29   28
   28   28   28   28   30   35   44   56   70   86  101  116  131  146  161  172
  180  187  195  204  211  214  217  224  240
     7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  210  230  251  267  274  274  269  260  247  230  210  188  166  148  134  122
  112  103   95   92   94  100  107  114  118  118  114  105   95   85   78   73
   70   67   63   59   54   49   44   41   40   38   37   35   33   31   30   30
   29   29   28   29   31   37   47   60   74   89  103  118  133  149  164  175
  181  185  189  194  197  197  195  198  210
     5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  186  202  221  238  251  260  263  262  255  241  223  202  181  161  145  133
  122  112  103   99  100  105  112  118  122  122  117  107   94   84   76   72
   69   66   62   59   54   50   46   42   40   38   37   36   34   32   31   31
   30   30   29   29   32   39   50   62   75   89  102  117  133  151  166  177
  183  186  189  191  192  188  182  180  186
     2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  177  185  199  215  230  243  253  257  254  244  229  211  191  172  155  143
  132  122  113  108  107  110  115  120  122  121  115  105   92   82   76   72
   69   66   63   59   56   52   48   43   40   38   37   36   35   33   32   31
   31   30   29   29   32   40   50   63   75   86   99  114  132  150  168  180
  187  190  193  196  196  191  182  176  177
     0.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  184  187  195  205  218  231  243  249  248  241  229  214  196  179  164  152
  142  133  124  118  116  116  117  118  118  116  110  100   90   81   76   72
   70   66   63   60   57   54   49   44   40   38   37   36   35   34   32   31
   31   30   29   28   32   39   50   61   72   83   95  110  129  149  167  181
  190  196  202  207  209  206  197  188  184
    -2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  209  207  209  213  220  230  240  244  243  235  225  212  198  184  171  160
  151  143  136  130  126  123  120  116  113  109  103   95   86   80   76   73
   70   65   62   60   58   55   50   44   39   37   37   37   36   34   33   32
   31   30   29   28   31   39   48   59   69   79   91  106  125  146  165  181
  192  202  211  221  228  228  222  214  209
    -5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  244  241  239  238  239  243  247  247  241  232  221  210  198  187  176  167
  160  154  148  143  138  132  125  117  109  103   96   90   84   79   76   73
   69   64   61   59   59   56   51   44   39   37   37   38   38   36   34   33
   32   31   29   29   31   38   47   56   65   76   88  103  122  142  161  177
  191  204  218  233  245  251  251  247  244
    -7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  280  280  277  273  270  268  266  260  248  234  221  209  199  189  180  172
  167  162  159  156  151  143  133  120  109   99   92   87   82   79   77   73
   68   62   59   59   59   57   52   45   40   39   39   41   41   39   37   36
   35   33   31   30   32   37   45   54   63   74   86  101  119  137  155  171
  185  201  219  239  257  269  276  279  280
   -10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  310  315  315  311  305  300  292  280  262  243  225  212  202  192  183  176
  172  170  169  168  164  157  144  128  113  101   92   86   82   80   77   73
   67   61   58   59   60   59   54   48   44   43   44   46   46   44   41   39
   37   35   33   31   33   37   44   53   63   73   86  100  116  132  147  161
  176  193  214  237  260  278  291  301  310
   -12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  325  337  342  341  336  330  320  303  281  257  235  219  207  196  187  180
  176  175  176  177  176  170  157  140  122  107   96   88   84   80   77   72
   66   61   59   60   62   62   58   53   50   49   50   51   51   49   46   43
   40   37   34   33   34   38   45   54   64   75   87  100  113  126  139  151
  163  180  202  227  252  274  293  310  325
   -15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  323  340  351  356  355  351  341  324  299  272  247  228  213  201  191  183
  178  177  180  184  186  182  170  153  133  116  101   92   85   81   77   72
   67   62   61   63   66   66   64   60   57   56   57   57   56   54   50   46
   43   39   36   34   36   40   47   55   66   77   89  100  111  121  130  140
  150  165  185  210  235  258  280  302  323
   -17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  305  326  342  351  356  356  350  335  312  284  257  236  219  205  194  184
  178  177  181  187  191  190  180  164  144  124  107   95   87   81   77   73
   69   66   66   68   71   71   70   67   65   64   63   62   60   57   52   48
   44   40   37   36   37   42   49   58   69   80   90  100  108  116  123  130
  138  150  168  189  213  235  258  281  305
   -20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  275  297  315  329  339  344  344  333  313  287  261  239  222  207  194  184
  177  175  179  185  191  192  185  171  151  130  111   97   87   81   77   74
   72   70   71   73   75   76   75   73   71   69   67   65   61   57   53   48
   43   39   37   37   39   44   51   61   71   81   90   98  104  110  116  122
  128  138  152  169  189  209  229  251  275
   -22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  240  259  277  292  306  316  321  316  301  280  256  236  219  205  192  181
  173  171  173  180  186  189  184  171  152  131  111   95   85   79   77   75
   75   74   75   77   78   79   78   76   73   71   67   63   59   55   51   46
   42   38   36   37   41   46   54   63   72   81   89   94  100  105  110  115
  121  128  138  152  168  185  201  220  240
   -25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  207  221  236  250  264  277  286  287  277  261  242  225  211  198  187  176
  168  164  166  171  177  179  175  163  146  125  106   91   81   77   75   76
   77   77   78   79   79   78   77   75   72   68   64   59   55   51   48   44
   40   37   36   38   42   48   55   63   72   79   85   89   93   99  104  110
  115  120  128  139  152  165  178  192  207
   -27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  180  189  198  209  221  234  245  249  245  234  221  208  197  188  178  169
  161  157  157  160  164  165  161  150  134  115   98   84   76   73   74   75
   77   77   77   77   76   74   73   70   66   62   57   53   49   46   44   41
   38   36   36   38   43   48   55   62   69   75   79   82   86   92   98  104
  109  114  121  130  140  151  161  171  180
   -30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  162  166  169  175  183  194  204  209  209  203  195  187  181  175  168  161
  154  149  147  148  150  149  144  133  118  102   87   77   71   70   71   73
   75   75   74   72   70   68   66   63   59   54   49   45   43   42   41   39
   37   35   36   39   43   48   54   60   65   70   72   75   78   84   91   98
  103  108  114  122  132  142  150  157  162
   -32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  152  152  150  151  155  161  169  174  175  173  169  166  164  161  157  152
  146  142  139  138  137  134  127  116  103   89   78   71   68   68   70   71
   71   70   68   66   64   61   59   55   51   46   41   39   38   39   39   38
   37   36   37   40   44   48   52   57   61   64   66   68   71   77   84   90
   96  100  106  114  125  135  144  150  152
   -35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  147  145  141  137  137  139  143  146  148  147  147  147  148  149  147  144
  139  135  132  129  126  121  113  102   90   80   72   68   67   68   69   69
   67   65   62   60   58   56   53   49   44   39   36   35   36   38   39   40
   39   38   39   41   44   47   51   54   57   59   60   62   65   70   76   82
   87   91   97  106  117  129  139  145  147
   -37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  143  141  136  131  128  127  127  128  128  129  131  133  136  139  139  137
  134  130  127  124  119  113  104   94   84   76   71   70   70   71   71   68
   65   61   58   57   55   53   51   46   41   37   34   34   36   39   41   42
   42   41   41   43   45   47   49   52   54   55   56   58   60   65   70   74
   78   82   88   97  109  122  133  141  143
   -40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  138  138  134  129  124  121  119  118  117  118  120  124  128  131  132  131
  129  126  124  120  116  109  100   91   83   78   76   76   77   77   74   70
   65   61   58   57   56   55   52   48   42   38   35   36   39   42   45   45
   45   44   44   44   46   47   49   51   53   54   56   57   59   62   65   68
   71   74   79   87   99  112  125  134  138
   -42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  130  131  130  127  122  118  115  112  111  112  114  118  122  126  127  126
  125  123  121  119  115  109  101   94   88   85   84   85   86   84   80   74
   68   63   61   60   61   60   57   52   46   41   39   40   43   46   48   49
   48   46   46   46   47   48   50   52   54   56   58   59   61   63   64   66
   67   69   72   79   90  102  114  124  130
   -45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  118  122  123  121  119  115  112  109  107  108  110  114  118  120  122  122
  121  121  120  118  115  111  105   99   95   94   95   96   96   93   87   80
   73   68   66   67   68   67   64   58   52   47   44   45   47   50   51   51
   50   48   47   47   48   50   52   55   58   60   62   64   65   66   66   67
   66   67   69   74   82   92  103  112  118
   -47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  106  110  112  113  112  110  107  104  103  103  105  108  112  115  116  117
  117  117  118  117  116  113  109  105  103  103  104  105  104  100   93   86
   79   74   73   74   75   74   70   64   57   52   49   49   50   52   53   53
   51   49   48   49   50   52   56   59   63   66   69   70   71   72   71   70
   69   69   70   72   78   85   92  100  106
   -50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   94   98  101  102  103  101   99   98   96   97   98  101  105  107  109  110
  112  113  115  116  115  114  112  109  108  109  110  110  109  104   98   90
   84   80   79   79   80   78   74   68   61   55   52   51   52   53   54   53
   51   50   49   50   52   55   59   64   68   72   75   77   78   78   77   76
   75   74   73   74   76   80   85   90   94
   -52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   86   88   91   92   93   92   91   89   88   88   90   93   96   98  101  103
  105  108  110  112  113  112  111  110  110  110  111  111  109  105   99   93
   87   84   83   82   82   79   75   68   62   56   53   52   52   53   53   52
   51   50   50   51   54   58   63   69   74   78   81   83   84   84   84   82
   81   80   78   78   78   79   81   83   86
   -55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   81   83   84   85   85   84   83   81   80   80   81   83   86   89   92   95
   98  101  104  107  108  108  108  107  107  108  108  108  106  102   97   92
   88   85   83   82   80   77   72   66   60   55   52   51   51   52   52   51
   51   50   51   53   57   61   67   73   78   82   86   88   89   89   88   87
   86   85   84   83   82   81   80   81   81
   -57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   82   82   83   82   82   80   78   76   74   73   74   75   78   81   84   88
   91   95   98  100  102  102  102  101  101  101  102  101  100   97   93   89
   86   83   81   79   77   73   68   62   56   52   50   49   49   50   51   51
   51   51   52   55   59   64   70   75   81   85   88   90   91   91   91   90
   90   89   89   87   86   85   83   83   82
   -60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   87   87   86   85   84   81   78   75   73   71   70   71   73   76   79   82
   86   89   92   94   95   95   95   94   94   94   94   93   92   91   88   85
   83   80   78   75   72   68   63   57   53   49   48   47   48   49   50   51
   51   52   54   57   61   66   71   77   82   86   88   90   91   91   91   91
   91   91   91   91   91   90   89   88   87
   -62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   96   95   95   93   91   88   84   80   76   73   72   71   72   74   76   79
   82   85   87   88   89   89   88   88   87   87   87   87   86   85   84   82
   80   77   75   72   68   64   59   55   51   48   47   47   48   49   50   51
   52   53   56   59   63   67   72   77   81   85   87   88   89   89   89   90
   91   92   93   94   95   95   96   96   96
   -65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  106  106  106  104  102   98   94   89   85   81   78   76   76   76   78   79
   81   83   84   85   85   84   84   83   82   82   82   82   82   82   81   80
   78   76   73   70   66   62   58   54   51   49   48   48   49   50   51   52
   53   55   57   60   64   68   72   76   80   83   85   86   86   87   87   88
   90   92   94   96   99  101  103  105  106
   -67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  115  117  117  116  113  110  106  101   96   92   88   85   83   82   82   82
   83   83   83   83   83   82   81   80   80   80   80   80   80   80   80   79
   78   76   74   70   67   63   59   56   53   51   50   50   51   51   52   54
   55   57   59   62   65   68   72   75   78   80   82   83   84   84   85   87
   89   92   95   99  103  106  110  113  115
   -70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  123  125  126  126  124  121  117  113  108  103   99   95   92   89   88   87
   86   85   84   84   83   82   81   80   80   80   80   80   81   81   81   80
   79   78   75   72   69   66   63   60   57   55   54   54   54   54   55   56
   57   58   60   63   65   68   71   74   76   78   80   81   82   83   84   86
   89   93   97  101  106  111  115  119  123
   -72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  127  130  132  132  131  129  126  122  117  112  108  104  100   97   94   92
   90   88   87   85   84   83   82   81   81   81   81   82   82   82   82   82
   81   79   77   75   72   69   66   64   61   59   58   57   57   57   57   58
   58   60   61   63   65   68   70   72   74   76   78   79   81   82   84   87
   90   94   98  103  109  114  119  123  127
   -75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  128  131  133  133  133  131  129  126  122  118  114  110  106  102   99   96
   94   91   89   88   86   85   84   83   83   83   83   83   83   83   83   83
   82   81   79   77   75   72   70   67   65   63   62   60   60   59   59   59
   60   61   62   64   65   67   69   71   73   75   77   78   80   83   85   88
   91   95  100  105  110  115  120  124  128
   -77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  124  127  129  130  130  129  127  125  122  119  115  112  108  105  102   99
   96   93   91   89   88   87   85   85   84   84   84   84   83   83   83   82
   81   80   79   77   75   73   71   69   67   66   64   63   62   61   61   61
   61   62   63   64   65   67   69   70   72   74   76   78   80   83   86   89
   92   96  100  104  109  113  117  121  124
   -80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  117  119  121  122  122  121  120  119  117  115  112  109  107  104  101   99
   96   94   92   90   88   87   86   85   84   84   83   83   82   82   81   81
   80   79   78   76   75   73   72   70   69   67   66   65   64   63   63   63
   63   64   64   65   66   67   69   70   72   74   76   78   80   83   85   88
   91   95   98  102  105  109  112  115  117
   -82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  108  109  110  111  111  111  110  110  108  107  105  104  102  100   98   96
   94   92   91   89   88   87   85   84   84   83   82   81   81   80   79   78
   78   77   76   75   74   73   71   70   69   68   67   67   66   66   65   65
   65   66   66   67   68   69   70   71   73   74   76   78   80   82   84   87
   89   92   94   97   99  102  104  106  108
   -85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   97   98   99   99   99   99   99   99   98   98   97   96   95   94   93   91
   90   89   88   87   86   85   84   83   82   81   81   80   79   78   78   77
   76   75   75   74   73   73   72   71   71   70   70   69   69   69   69   69
   69   69   69   70   71   71   72   73   74   75   77   78   80   81   83   84
   86   87   89   91   92   94   95   96   97
   -87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   87   88   88   88   88   88   88   88   88   88   88   87   87   87   86   86
   85   85   84   83   83   82   82   81   81   80   80   79   79   78   78   77
   77   76   76   75   75   75   74   74   74   74   73   73   73   73   73   73
   73   74   74   74   75   75   76   76   77   77   78   79   79   80   81   82
   82   83   84   85   85   86   86   87   87
     1                                                      END OF TEC MAP      
     2                                                      START OF TEC MAP    
  2019     1     8     9     0     0                        EPOCH OF CURRENT MAP
    87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    9    9    9   10   10   10   10   10   10   10   11   11   11   11   11   11
   11   11   11   11   11   11   11   11   11   11   11   11   11   12   12   12
   12   12   13   13   13   13   13   13   13   13   13   13   13   12   12   12
   12   11   11   11   11   10   10   10    9    9    9    9    9    9    8    8
    8    8    8    9    9    9    9    9    9
    85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    8    9    9   10   10   10   11   11   11   12   12   12   12   12   12   11
   11   11   10   10   10    9    9    9    9    9   10   10   10   11   12   12
   13   14   14   15   15   16   16   16   17   16   16   16   16   15   15   14
   13   13   12   11   10   10    9    8    8    7    7    7    6    6    6    6
    6    6    6    7    7    7    8    8    8
    82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    9    9   10   11   11   12   12   13   13   14   14   14   14   13   13   12
   12   11   10    9    8    7    6    6    6    6    6    7    8    9   10   11
   13   14   16   17   18   19   20   20   20   20   20   20   19   18   17   16
   15   14   13   12   11   10    9    8    7    6    6    5    5    5    5    5
    5    5    6    6    7    7    8    8    9
    80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   11   11   12   12   13   14   14   15   15   16   16   16   16   16   15   14
   12   11    9    8    6    4    3    2    1    1    2    3    4    6    8   10
   13   15   17   19   21   22   23   24   24   23   23   22   21   20   19   17
   16   15   13   12   11   10    9    8    7    7    6    6    6    6    6    7
    7    7    8    8    9    9   10   10   11
    77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   14   14   15   15   15   15   16   16   17   17   18   18   18   17   17   15
   14   12   10    7    5    3    1    0    0    0    0    0    1    3    6   10
   13   16   19   21   23   24   25   25   25   24   23   22   21   20   19   17
   16   15   14   13   12   11   10   10    9    9    9    9   10   10   11   11
   12   12   13   13   14   14   14   14   14
    75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   20   19   18   18   18   17   17   18   18   18   19   19   19   19   18   17
   15   13   11    8    5    3    0    0    0    0    0    0    0    2    6    9
   13   16   19   22   23   24   24   24   23   22   21   20   19   18   17   16
   15   15   14   14   13   13   13   13   13   13   13   14   15   16   17   18
   19   19   20   21   21   21   21   20   20
    72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   26   25   23   22   21   20   19   19   19   19   19   19   19   19   18   17
   16   14   12   10    7    5    2    0    0    0    0    0    0    3    6   10
   13   16   19   20   21   21   21   20   18   17   16   15   15   14   14   14
   14   14   15   15   15   15   16   16   16   17   18   19   21   22   24   25
   27   28   28   29   29   29   28   27   26
    70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   33   31   29   27   25   23   21   20   20   19   19   19   19   18   18   17
   16   15   14   12   10    8    5    3    1    0    0    0    2    4    7   10
   13   15   17   17   17   16   14   13   11   10    9    9    9   10   11   12
   13   14   15   16   17   17   18   18   19   20   22   23   25   28   30   32
   34   35   36   37   37   36   36   35   33
    67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   40   37   35   32   29   26   24   23   21   20   20   19   18   17   16   16
   15   15   14   14   13   11    9    7    6    4    3    3    4    6    8   10
   12   13   13   12   11    9    7    5    4    3    3    4    5    7    8   10
   12   14   15   16   17   18   18   19   20   21   23   25   28   31   34   36
   39   40   42   42   43   43   42   41   40
    65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   46   43   41   37   34   31   28   26   24   22   21   19   17   16   14   14
   13   13   14   14   14   14   13   11    9    8    6    5    5    6    7    8
    9    9    8    7    5    2    0    0    0    0    0    1    3    6    8   10
   12   14   15   16   17   17   17   18   19   20   22   25   29   32   35   39
   41   43   45   46   47   47   47   47   46
    62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   51   49   46   43   39   35   32   29   27   25   23   21   18   15   12   11
   11   11   12   13   14   14   14   13   11    9    8    6    5    5    5    5
    6    5    4    2    0    0    0    0    0    0    0    2    5    8   10   11
   13   14   15   15   16   15   15   15   16   17   20   24   28   32   36   39
   42   44   46   48   49   50   51   52   51
    60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   56   55   52   48   44   39   36   33   31   29   26   23   19   15   11    9
    8    8    9   11   12   12   12   12   11    9    7    6    4    3    3    3
    3    2    1    0    0    0    0    0    0    1    5    8   10   12   13   14
   14   15   15   14   14   13   12   12   13   15   18   22   27   31   36   39
   42   45   47   49   51   53   54   56   56
    57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   61   60   58   54   49   44   40   37   35   32   29   25   20   15   11    8
    7    7    7    8    8    9    9    9    8    7    6    4    3    2    2    1
    1    1    1    0    0    0    1    3    6   10   14   17   18   19   19   18
   17   16   15   14   12   11   10   10   11   14   18   23   28   33   37   41
   44   46   48   50   52   55   58   60   61
    55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   67   67   65   60   55   49   44   41   38   36   33   28   23   17   13    9
    7    7    6    6    5    5    5    5    6    5    5    4    3    3    3    3
    3    4    4    5    6    8   11   14   18   22   25   27   28   27   24   22
   19   18   16   14   13   11   10   10   12   16   21   26   32   37   42   45
   47   49   51   53   56   59   62   65   67
    52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   73   74   72   67   61   55   49   45   42   39   36   31   26   20   15   12
   11    9    8    6    4    3    3    3    5    6    6    6    6    6    7    7
    8   10   12   14   17   20   24   27   31   35   37   38   37   34   30   26
   23   20   18   17   15   13   13   14   17   22   28   35   40   45   49   51
   53   54   56   58   61   64   68   71   73
    50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   79   80   78   74   68   61   55   50   46   42   39   34   29   24   20   18
   16   15   13    9    6    4    3    5    7    9   11   12   12   13   14   15
   17   19   22   26   30   34   38   42   45   47   48   48   45   41   35   31
   27   24   22   21   20   19   19   22   26   33   40   46   52   56   58   59
   60   61   63   64   67   70   73   77   79
    47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   83   84   84   81   76   69   62   55   50   46   42   38   33   29   26   24
   24   23   20   16   12    9    8   10   13   16   19   21   22   23   24   25
   27   30   33   38   43   48   52   54   56   57   57   55   51   46   40   35
   32   30   28   27   26   26   28   32   38   46   53   60   65   68   69   69
   68   69   70   71   73   75   78   81   83
    45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   84   86   87   86   82   77   69   62   56   51   47   43   38   35   33   32
   33   32   30   26   21   18   18   20   24   28   31   33   34   35   36   36
   38   40   44   49   55   60   63   64   64   64   62   60   56   50   45   40
   38   37   36   35   35   35   38   44   51   60   68   74   78   79   79   78
   77   76   76   77   78   80   81   82   84
    42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   82   84   86   88   87   83   77   70   63   58   53   49   44   41   40   40
   42   42   41   37   33   30   30   33   37   41   44   46   47   47   47   47
   47   49   53   58   64   68   70   70   69   67   65   63   59   54   49   46
   44   44   44   43   43   44   48   55   63   73   80   86   89   90   88   86
   85   84   83   83   83   83   82   81   82
    40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   78   79   82   86   88   87   83   77   70   65   60   56   52   49   48   48
   50   51   50   48   45   43   43   46   50   55   57   58   58   56   55   54
   54   56   59   64   69   73   74   73   71   68   66   64   61   57   53   51
   51   51   51   50   49   51   56   64   73   82   90   95   98   98   97   95
   93   92   91   90   89   86   83   80   78
    37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   76   75   77   82   87   89   87   82   77   71   67   63   60   57   55   55
   56   58   58   57   56   54   55   58   62   66   68   68   66   63   61   60
   59   60   63   67   72   75   75   73   70   68   66   64   62   59   57   56
   56   56   55   54   53   55   61   70   79   89   96  101  104  105  105  105
  104  103  102  100   97   92   86   80   76
    35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   78   74   74   78   84   88   89   86   81   77   74   71   68   65   61   60
   60   62   64   64   64   63   64   67   71   74   75   74   71   67   64   62
   62   63   65   69   72   74   74   72   68   66   65   64   63   61   59   59
   59   59   57   55   54   56   63   72   82   91   99  104  109  112  115  117
  118  118  117  115  110  104   95   86   78
    32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   89   80   76   77   82   86   88   87   83   81   79   77   75   71   67   64
   63   64   66   68   69   69   69   71   74   77   78   77   73   69   65   64
   63   65   67   69   71   72   71   69   66   64   63   63   62   61   59   59
   60   59   56   53   51   55   62   72   83   92   99  106  113  120  126  132
  136  138  138  135  131  123  113  101   89
    30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  111   96   86   83   84   86   88   87   84   83   82   82   80   76   71   66
   64   64   67   70   71   71   71   72   74   77   78   77   73   69   65   64
   65   66   68   69   69   69   68   65   63   62   61   61   60   58   57   57
   57   55   52   48   47   51   60   72   82   91  100  108  118  129  140  150
  158  163  164  162  157  150  140  126  111
    27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  143  124  107   97   92   91   90   88   86   84   84   85   84   80   74   67
   64   64   67   70   72   71   69   69   71   74   77   76   73   69   66   65
   67   69   69   68   67   65   63   61   59   58   58   57   55   53   52   52
   51   50   46   42   42   48   59   71   82   92  101  111  124  139  154  169
  181  188  191  191  188  183  174  161  143
    25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  182  160  138  120  109  102   98   93   89   87   87   87   86   82   75   68
   64   64   67   71   72   70   67   66   68   71   75   75   73   69   67   67
   69   71   70   67   64   61   58   57   55   54   53   51   49   47   45   45
   45   43   39   37   38   46   58   72   84   94  104  117  132  150  169  186
  202  212  217  218  218  216  211  200  182
    22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  223  200  174  151  134  122  112  104   97   92   90   89   88   83   76   69
   64   64   68   72   73   70   67   65   66   70   74   76   74   71   69   69
   71   72   70   65   60   56   53   52   50   49   47   45   42   39   38   38
   38   36   34   33   36   46   60   75   87   99  110  124  141  160  180  200
  217  229  236  239  242  244  245  238  223
    20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  257  237  211  186  164  147  133  121  110  101   96   92   90   85   78   70
   66   66   70   74   75   73   69   66   67   71   75   77   76   74   71   71
   72   72   68   62   55   50   48   47   45   43   41   38   35   32   31   31
   32   32   30   31   37   48   64   79   93  105  118  133  149  168  188  207
  225  237  245  250  255  262  268  268  257
    17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  280  265  243  219  196  176  159  142  127  114  105   98   93   87   80   73
   68   69   73   78   80   78   75   72   72   74   78   80   79   76   73   72
   72   70   65   58   50   46   43   42   41   38   35   32   29   27   26   27
   28   29   29   32   40   52   69   85  100  113  127  142  157  174  191  208
  224  236  243  249  256  266  277  284  280
    15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  286  279  263  244  224  205  185  166  147  131  117  106   99   92   84   76
   72   72   76   82   85   85   83   81   80   81   82   82   81   78   75   72
   70   67   61   53   46   42   40   39   37   34   31   28   25   24   24   25
   27   28   30   35   44   58   74   91  107  121  136  150  164  177  190  204
  217  226  232  237  244  256  271  283  286
    12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  277  278  271  259  244  228  209  189  168  148  130  117  107   98   89   81
   76   76   80   86   91   93   93   92   90   88   87   85   82   79   75   71
   68   63   56   49   43   40   38   37   35   32   28   26   24   23   24   25
   27   29   32   38   49   63   79   96  112  128  143  157  169  179  188  197
  206  213  216  219  224  236  252  268  277
    10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  257  264  266  262  255  243  227  207  185  164  144  129  117  106   97   87
   81   79   83   89   95  100  102  102  100   95   90   86   83   79   75   70
   65   59   53   47   42   40   38   37   34   31   28   26   25   25   25   26
   28   31   35   42   53   67   83   99  116  133  149  163  173  181  186  192
  197  200  201  201  203  212  227  244  257
     7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  233  245  253  257  256  249  236  219  198  176  156  140  127  116  106   95
   87   84   86   91   98  105  109  110  107  100   92   86   81   78   74   68
   62   56   51   46   43   41   40   38   34   31   29   28   27   27   27   28
   29   32   37   45   56   70   85  100  117  135  152  166  177  183  186  189
  192  193  192  188  188  193  204  219  233
     5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  215  227  238  247  251  248  239  223  204  184  166  150  138  127  116  104
   94   89   89   93   99  106  111  113  109  102   92   84   79   76   72   67
   61   55   50   47   44   43   41   38   35   32   30   30   30   30   29   29
   30   32   38   47   58   70   84   99  116  134  152  167  179  185  189  192
  194  194  192  187  183  184  191  202  215
     2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  208  218  229  239  244  243  236  223  206  188  172  158  147  137  126  114
  102   95   92   94   99  105  110  111  108  100   90   82   77   74   70   66
   61   56   52   49   46   44   41   39   36   33   32   32   32   32   30   29
   29   32   38   47   58   70   83   96  113  130  149  166  179  188  194  199
  202  203  202  197  192  190  192  199  208
     0.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  216  223  230  237  241  240  232  220  205  190  176  164  155  146  136  124
  112  103   98   97   99  102  105  106  102   95   86   78   74   71   69   65
   61   57   54   50   47   44   41   38   36   34   34   34   34   33   31   29
   29   32   38   47   58   69   80   92  107  125  144  162  178  190  200  207
  214  218  219  217  213  210  209  212  216
    -2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  240  241  244  246  245  240  231  219  205  191  179  169  161  154  146  135
  124  114  106  102   99   99   99   98   95   88   81   75   71   69   67   64
   61   58   55   52   48   43   39   37   36   36   36   36   35   33   31   29
   29   31   38   47   57   67   77   88  101  118  136  156  174  190  203  214
  225  234  239  242  241  239  238  239  240
    -5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  272  270  267  263  257  248  235  221  207  194  182  173  166  161  155  146
  136  126  117  109  103   98   94   91   87   82   76   72   69   67   65   62
   61   59   57   53   47   42   38   36   36   37   38   37   36   34   31   29
   29   32   38   47   57   66   74   84   95  110  128  148  167  186  202  217
  232  245  256  264  269  271  272  273  272
    -7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  306  302  296  287  275  261  245  229  214  200  187  178  171  166  162  156
  148  139  130  120  110  101   93   86   81   77   73   70   68   65   62   61
   60   60   58   53   47   41   37   37   39   41   41   40   38   35   33   31
   31   33   39   48   57   65   73   80   90  103  120  139  159  178  197  214
  232  249  265  279  289  297  303  306  306
   -10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  332  329  322  310  295  278  259  241  223  208  194  182  174  170  167  164
  159  152  143  133  121  108   96   86   79   74   72   70   67   64   61   59
   60   60   59   54   47   41   39   40   43   45   45   44   41   38   36   34
   33   36   41   50   58   66   72   78   87   98  112  130  149  168  187  205
  224  244  264  282  298  311  322  329  332
   -12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  344  344  338  327  311  293  273  253  235  217  201  187  177  172  170  170
  168  163  156  146  134  120  105   91   81   76   73   71   68   64   60   59
   60   61   60   56   50   45   43   46   49   51   51   48   45   42   39   37
   36   38   44   51   60   67   72   78   85   94  107  122  139  157  174  191
  209  230  251  272  292  310  326  338  344
   -15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  338  342  340  331  318  301  282  262  243  225  207  191  179  172  171  173
  173  171  167  159  148  133  117  101   88   80   76   73   69   65   61   60
   62   64   63   60   54   51   50   53   56   57   56   52   48   45   42   40
   39   41   46   53   61   68   74   79   85   93  104  117  131  146  160  175
  191  210  231  252  274  295  313  329  338
   -17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  316  324  325  322  313  300  283  265  247  229  210  193  179  171  170  172
  175  175  173  168  159  146  130  112   96   85   79   76   72   67   64   63
   64   67   67   64   60   58   58   61   63   62   59   55   51   47   45   42
   41   42   47   54   62   70   75   81   87   94  103  113  125  137  148  159
  172  188  206  227  247  268  287  304  316
   -20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  282  292  297  299  296  287  274  260  244  227  209  191  176  168  166  169
  172  174  174  172  166  156  141  123  105   91   83   78   74   70   67   66
   68   70   71   69   67   65   65   67   67   65   61   56   52   48   45   43
   41   43   48   55   63   71   77   83   89   95  103  111  120  129  138  146
  155  168  182  199  217  236  254  270  282
   -22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  243  253  261  267  268  265  256  245  232  218  202  186  171  163  160  162
  165  168  169  170  168  161  148  130  110   95   85   79   76   73   71   70
   71   73   74   73   71   70   70   69   68   65   60   55   51   47   44   41
   41   43   48   55   63   71   77   84   90   96  102  109  116  123  130  136
  142  151  161  175  190  205  219  232  243
   -25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  206  213  222  230  235  235  231  224  215  204  191  177  165  156  153  154
  156  158  160  162  163  159  149  132  112   95   84   79   76   75   73   73
   73   75   75   75   73   71   70   68   65   61   56   51   48   44   41   39
   39   42   48   55   63   70   77   83   89   95  100  106  112  117  123  127
  132  138  145  155  167  178  189  198  206
   -27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  175  180  187  195  201  204  203  199  194  187  178  167  157  149  146  145
  146  146  147  150  152  151  143  128  109   93   82   76   75   75   74   74
   74   74   74   73   71   69   66   62   58   54   50   47   44   41   38   37
   38   42   48   56   63   69   75   81   86   91   96  100  105  110  116  120
  124  128  133  141  150  159  166  171  175
   -30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  153  155  159  165  172  176  176  175  172  169  164  157  149  143  139  137
  136  134  134  136  138  139  133  120  103   88   77   73   72   73   73   72
   72   71   70   69   66   63   59   54   50   47   44   42   40   38   36   35
   37   42   49   56   62   67   72   77   82   85   89   92   96  102  107  112
  116  119  124  131  139  146  151  153  153
   -32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  141  139  140  144  149  152  153  154  153  153  151  147  142  137  134  131
  127  124  121  122  124  125  121  110   96   82   73   69   69   70   70   69
   68   67   65   63   60   56   51   46   42   40   39   39   38   36   35   35
   38   44   51   57   61   65   68   72   75   78   80   82   86   91   98  103
  107  111  115  122  131  138  143  143  141
   -35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  134  131  129  131  133  135  136  137  138  140  140  139  136  133  129  126
  121  116  112  110  111  112  109  101   90   79   71   67   67   68   67   66
   65   63   61   58   55   50   44   39   37   36   37   38   38   37   36   37
   41   47   53   58   61   62   64   67   69   70   71   72   75   81   87   93
   97  101  106  114  123  132  137  137  134
   -37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  131  127  124  123  123  124  125  126  128  130  132  133  132  130  127  123
  118  112  106  103  103  103  101   96   87   78   71   68   67   66   66   64
   62   60   58   56   51   46   40   36   35   35   38   40   40   39   38   40
   44   49   55   59   60   61   61   63   64   64   64   64   67   71   78   83
   88   92   97  105  115  125  132  134  131
   -40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  127  124  120  118  117  117  117  118  120  123  126  128  128  127  125  121
  116  110  105  101  100  100   99   95   89   82   76   72   70   68   66   64
   62   61   59   56   51   46   40   37   36   38   41   44   44   43   42   43
   47   52   57   60   60   60   60   61   61   61   61   60   62   66   71   76
   80   83   88   96  106  117  125  129  127
   -42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  121  119  116  113  111  111  111  112  114  118  121  123  124  124  123  121
  117  112  107  103  101  101  101   99   95   89   84   79   75   72   69   67
   65   64   62   59   55   49   44   41   41   43   46   49   49   47   46   47
   50   55   59   61   61   61   61   61   62   62   61   61   62   65   69   72
   75   77   80   87   97  107  116  121  121
   -45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  113  112  109  107  105  104  105  106  109  112  115  117  119  120  120  119
  117  114  110  107  106  106  107  106  103   99   94   88   83   78   75   72
   71   70   68   65   60   55   49   46   46   49   52   53   53   52   50   51
   53   57   60   62   62   62   63   65   66   67   66   66   67   68   71   73
   73   74   75   80   88   98  106  111  113
   -47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  103  103  101   99   98   97   98   99  102  105  108  111  113  115  116  117
  117  115  113  112  112  112  114  114  112  108  103   97   91   85   81   78
   77   76   74   71   66   61   55   52   52   54   56   57   57   55   53   53
   55   58   60   62   64   65   67   70   72   74   74   74   75   76   77   77
   76   74   74   77   82   89   96  101  103
   -50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   94   94   93   91   90   89   90   92   94   97  101  104  106  109  111  113
  114  115  115  115  116  117  119  120  119  115  110  104   97   91   86   84
   82   81   79   76   71   65   60   57   56   57   58   59   58   57   55   55
   56   59   61   64   66   69   72   76   79   82   83   83   84   84   84   83
   81   78   76   77   79   84   89   93   94
   -52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   88   88   86   84   83   83   83   85   87   90   93   96   99  102  105  108
  111  112  113  115  116  118  120  121  121  118  113  107  101   95   90   87
   85   84   82   79   74   68   62   59   57   58   59   59   59   58   56   56
   57   60   62   65   68   72   77   81   85   89   91   91   92   92   91   89
   87   83   81   79   80   83   85   88   88
   -55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   87   85   84   81   80   79   79   79   81   84   86   89   92   96   99  102
  105  107  109  111  113  116  118  119  118  116  112  106  100   95   91   88
   86   85   82   78   73   68   62   59   57   57   58   58   58   58   57   58
   59   61   64   67   71   76   80   85   90   93   96   97   97   97   96   94
   92   89   86   85   84   85   86   87   87
   -57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   90   88   85   83   80   78   78   78   78   80   82   84   87   90   93   96
   99  101  103  105  107  109  111  112  112  110  106  102   97   93   89   87
   85   83   80   76   71   65   61   57   55   55   56   56   57   58   58   59
   60   63   66   69   74   78   83   88   92   95   97   98   99   99   98   97
   95   94   92   91   90   90   91   91   90
   -60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   97   95   92   89   85   83   81   80   79   80   81   82   84   87   89   91
   93   95   96   98   99  101  102  103  103  101   99   96   92   89   86   84
   82   80   77   73   68   63   58   55   53   53   54   55   56   57   59   60
   62   65   68   71   76   80   84   88   92   95   96   97   98   98   98   98
   97   97   97   97   97   98   98   98   97
   -62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  106  104  101   98   94   91   88   85   84   83   83   83   84   85   86   87
   88   89   90   91   91   92   93   94   94   93   92   90   88   86   84   82
   80   77   74   70   66   61   57   54   53   52   53   54   56   58   59   61
   64   66   69   73   77   80   84   87   90   92   93   94   95   95   96   97
   98   99  101  103  105  106  107  107  106
   -65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  116  115  112  108  104  100   97   93   91   89   87   87   86   86   86   86
   86   85   85   85   85   86   86   87   87   87   86   86   85   83   82   81
   79   76   73   69   65   61   58   55   53   53   53   55   56   58   60   62
   65   67   70   73   77   80   83   85   87   89   90   90   91   92   94   96
   99  102  105  108  111  114  116  117  116
   -67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  125  124  122  119  115  111  106  102   99   96   93   91   90   88   87   86
   85   84   83   82   82   82   82   83   83   84   84   84   84   83   82   81
   79   77   74   71   67   63   60   57   55   54   55   55   57   58   60   62
   65   67   70   73   75   78   80   83   84   85   87   88   89   91   93   96
  100  104  108  113  117  121  123  125  125
   -70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  131  131  129  126  123  119  114  110  106  103   99   96   94   91   89   87
   86   84   83   82   82   81   82   82   83   83   84   84   85   85   84   83
   81   79   76   73   69   66   63   60   58   57   56   57   57   59   60   62
   64   66   69   71   74   76   78   80   81   83   84   86   88   90   93   97
  101  106  111  116  121  125  129  131  131
   -72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  134  133  132  130  127  123  119  115  111  107  104  100   97   94   92   90
   88   86   85   84   83   83   83   84   84   85   86   86   87   87   86   85
   84   81   79   76   72   69   66   63   61   59   58   58   58   59   60   61
   63   65   67   69   71   73   75   77   79   81   83   85   88   91   94   98
  103  108  113  118  123  127  130  132  134
   -75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  131  132  131  129  127  124  120  117  113  109  106  103   99   97   94   92
   90   88   87   86   86   85   85   86   86   87   88   88   88   88   87   86
   85   83   80   77   74   71   68   65   63   61   60   59   58   59   59   60
   61   63   65   67   69   71   73   75   77   80   82   85   88   91   95   99
  103  108  113  117  122  125  128  130  131
   -77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  125  126  125  124  122  120  118  115  112  109  106  103  100   97   95   93
   91   90   89   88   87   87   87   87   87   88   88   88   88   88   87   86
   84   82   80   77   75   72   69   67   65   63   61   60   59   59   59   60
   60   62   63   65   67   69   71   73   76   78   81   84   87   91   94   98
  102  106  110  114  117  120  123  124  125
   -80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  116  117  116  116  115  113  111  109  107  105  103  100   98   96   94   93
   91   90   89   88   88   87   87   87   87   87   87   86   86   85   84   83
   82   80   78   76   74   72   69   67   65   64   62   61   60   60   60   60
   61   62   63   64   66   68   70   72   74   77   80   82   85   89   92   95
   98  101  105  107  110  112  114  116  116
   -82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  105  106  106  105  105  104  103  102  100   99   98   96   95   93   92   91
   90   89   88   87   86   86   85   85   85   84   84   83   82   82   81   80
   78   77   75   74   72   71   69   68   66   65   64   63   63   62   62   62
   63   63   64   65   67   68   70   72   74   76   78   80   83   85   88   90
   92   95   97   99  101  102  104  105  105
   -85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   94   95   95   95   95   94   94   93   93   92   91   90   89   89   88   87
   86   86   85   84   84   83   82   82   81   81   80   79   79   78   77   76
   75   74   74   73   72   71   70   69   68   67   67   66   66   66   66   66
   66   67   67   68   69   70   71   72   74   75   77   78   80   81   83   84
   86   87   89   90   91   92   93   94   94
   -87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   84   85   85   85   85   85   85   85   84   84   84   84   83   83   83   82
   82   81   81   81   80   80   79   79   78   78   78   77   77   76   76   75
   75   74   74   73   73   72   72   72   71   71   71   71   71   71   71   71
   71   71   72   72   72   73   74   74   75   75   76   77   78   78   79   80
   80   81   82   82   83   83   84   84   84
     2                                                      END OF TEC MAP 
     3                                                      START OF TEC MAP    
  2019     1     8     10     0     0                        EPOCH OF CURRENT MAP
    87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    9    9   10   10   10   11   11   11   11   12   12   12   12   12   13   13
   13   13   13   13   13   13   14   14   14   14   14   14   14   14   14   14
   15   15   15   15   14   14   14   14   14   14   13   13   13   12   12   11
   11   11   10   10    9    9    9    8    8    8    8    8    7    7    7    7
    7    8    8    8    8    8    9    9    9
    85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    7    8    9    9   10   10   11   12   12   12   13   13   13   13   13   13
   13   13   13   13   13   13   13   13   13   13   14   14   14   15   15   16
   17   17   17   18   18   18   18   17   17   16   16   15   14   13   12   11
   10   10    9    8    7    6    6    5    5    4    4    4    4    4    4    4
    4    4    4    5    5    6    6    7    7
    82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    7    7    8    9   10   11   12   12   13   14   14   14   14   14   14   13
   13   12   11   11   10   10    9    9    9   10   10   11   12   13   15   16
   17   19   20   21   21   21   21   21   20   20   18   17   16   14   13   11
   10    9    7    6    5    4    4    3    3    2    2    2    2    2    2    2
    3    3    3    4    4    5    5    6    7
    80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
    7    8    9    9   10   11   12   13   14   15   15   15   15   15   14   13
   12   11    9    8    6    5    4    4    4    4    5    7    9   11   13   15
   18   20   22   23   24   24   24   24   23   22   20   19   17   15   13   11
    9    8    7    6    5    4    4    4    3    4    4    4    4    4    5    5
    5    5    5    6    6    6    6    7    7
    77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   10   10   10   11   11   12   13   14   15   15   16   16   16   15   15   13
   11    9    7    5    3    1    0    0    0    0    1    3    5    8   11   15
   18   21   23   25   26   26   26   25   24   22   20   18   16   14   12   10
    9    8    7    6    6    6    6    7    7    8    8    9   10   10   11   11
   11   11   11   11   11   11   10   10   10
    75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   15   14   13   13   13   13   13   14   15   15   16   16   16   16   15   13
   11    9    6    4    1    0    0    0    0    0    0    0    3    7   10   14
   18   21   23   25   26   26   25   24   22   20   18   16   14   12   10    9
    8    8    8    8    9   10   11   12   13   14   16   17   18   19   20   20
   20   20   20   19   19   18   17   16   15
    72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   21   19   17   16   15   14   14   14   15   15   16   16   16   16   15   14
   12   10    7    4    2    0    0    0    0    0    0    0    3    7   10   14
   17   20   22   23   23   23   22   20   18   16   14   12   10    9    8    8
    8    9    9   11   12   14   15   17   19   21   23   25   26   28   29   30
   30   30   30   29   28   26   25   23   21
    70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   28   26   23   20   18   17   16   15   15   15   15   16   16   16   15   14
   13   11    9    7    4    2    1    0    0    0    1    3    5    8   11   14
   16   18   19   19   19   17   16   14   12   10    8    7    7    6    7    7
    8   10   11   13   15   17   19   21   24   26   28   31   33   35   37   38
   39   39   39   38   37   36   34   31   28
    67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   36   32   29   25   22   20   18   16   15   15   15   15   15   15   15   15
   14   12   11    9    8    6    5    4    4    4    5    6    7    9   11   13
   14   14   14   13   12   10    8    6    5    4    3    3    4    4    6    7
    9   11   13   15   17   19   21   23   26   28   31   34   36   39   41   43
   45   46   46   46   45   44   42   39   36
    65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   43   39   35   31   27   23   21   18   17   15   14   14   14   14   14   14
   14   13   12   11   10    9    9    8    8    8    8    9    9   10   10   10
   10   10    9    7    5    3    1    0    0    0    0    1    3    4    6    8
   10   12   14   15   17   19   20   22   24   27   30   33   37   40   43   45
   47   49   50   51   51   50   49   46   43
    62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   49   45   40   36   31   27   24   21   18   16   14   13   12   12   12   12
   12   12   12   11   11   11   11   11   11   11   10   10    9    9    8    8
    7    6    4    2    0    0    0    0    0    0    0    2    5    7    9   10
   12   13   14   15   16   17   18   19   21   24   27   31   35   38   42   45
   47   50   52   54   55   55   54   52   49
    60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   54   50   45   40   36   31   27   24   21   18   15   13   11   11   11   11
   11   10   10    9    9    9   10   10   11   10   10    9    8    7    6    5
    4    2    1    0    0    0    0    0    0    1    4    7    9   11   12   13
   14   14   14   14   14   14   15   16   18   21   25   29   33   37   41   44
   47   50   53   55   57   59   58   57   54
    57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   59   55   50   45   40   35   31   27   23   19   16   13   11    9    9    9
    9    8    7    7    6    6    7    8    8    9    8    7    6    4    4    3
    2    2    1    0    0    0    0    1    5    8   12   15   17   17   17   17
   16   15   14   14   13   13   14   15   18   21   26   30   35   38   42   45
   48   51   54   57   60   62   62   61   59
    55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   63   60   55   49   44   39   35   31   26   22   17   13   11    9    9    9
    8    7    6    4    3    3    4    5    6    6    6    5    5    4    4    4
    4    4    5    5    5    6    8   11   15   19   22   24   25   24   22   20
   18   17   15   14   14   14   15   18   21   26   31   36   40   43   46   48
   50   53   56   60   63   65   66   66   63
    52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   68   65   60   55   49   44   39   35   30   25   20   15   12   11   10   10
    9    8    6    3    2    1    2    3    5    6    6    6    6    6    7    7
    9   10   12   14   15   17   20   23   27   30   33   34   33   30   27   24
   21   19   18   17   17   19   21   25   30   35   41   46   50   52   54   55
   56   58   60   63   66   69   70   70   68
    50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   73   70   66   61   55   50   45   40   35   29   23   19   15   14   14   14
   13   12    9    6    3    3    3    5    7    9   10   11   12   12   13   14
   16   19   22   25   27   30   33   36   39   42   43   42   40   36   32   28
   25   23   22   22   24   27   31   36   42   48   54   59   63   64   64   63
   63   64   65   68   70   72   73   74   73
    47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   76   75   72   67   62   56   51   46   41   35   29   23   20   19   19   20
   20   18   15   12    9    8    9   11   14   17   19   20   21   21   22   23
   26   29   32   36   40   43   46   48   50   51   51   49   45   40   36   32
   29   28   28   29   33   37   43   50   57   63   70   74   77   77   75   73
   71   70   71   72   73   74   75   76   76
    45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   78   79   77   74   69   63   58   53   47   41   35   30   27   26   27   28
   28   27   25   21   19   18   19   22   25   28   31   32   33   33   33   33
   35   38   42   47   51   54   56   58   58   58   57   53   49   44   39   36
   34   34   35   38   42   48   56   63   71   78   84   88   90   89   86   82
   79   77   76   75   75   75   76   77   78
    42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   78   80   81   79   74   69   64   60   55   49   43   38   35   34   35   37
   38   37   35   33   31   30   32   35   39   42   44   45   45   44   43   43
   43   46   50   55   59   62   64   64   64   62   60   56   51   47   43   41
   40   40   42   46   52   59   67   76   83   90   96  100  101  100   96   91
   87   83   81   79   76   75   75   76   78
    40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   77   80   82   81   78   74   70   66   62   57   51   46   43   42   44   45
   47   47   45   44   43   43   45   49   53   56   58   58   56   54   52   50
   50   52   56   60   65   67   68   68   66   64   61   57   53   50   47   46
   46   47   49   53   59   67   76   85   92   99  104  108  110  109  106  100
   95   91   87   83   79   75   73   74   77
    37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   76   79   82   82   80   77   74   71   68   64   59   54   51   50   51   52
   54   54   54   54   54   55   57   61   65   68   68   67   65   62   58   56
   55   56   59   64   68   70   70   69   67   64   62   58   55   53   51   51
   51   51   53   57   64   72   81   90   97  104  109  115  118  119  116  112
  107  102   97   92   85   79   75   74   76
    35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   77   79   81   82   80   78   75   74   72   69   65   61   58   57   57   58
   58   59   60   60   62   63   66   70   74   75   75   73   69   66   62   59
   58   58   61   65   69   70   70   68   66   63   61   59   57   55   55   54
   54   54   56   59   66   75   84   92   99  106  113  120  126  130  130  127
  123  118  113  106   98   89   82   78   77
    32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   85   84   83   82   80   78   76   76   75   73   70   66   63   61   61   61
   61   62   63   64   66   68   71   75   78   79   78   74   70   67   64   62
   60   61   63   66   69   69   68   66   64   62   60   59   58   57   56   56
   55   54   56   60   66   75   85   93  101  108  117  127  137  144  147  147
  144  139  134  128  119  108   97   89   85
    30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  100   94   90   86   82   78   77   77   77   76   73   69   66   64   63   62
   62   62   64   66   68   70   73   76   79   80   77   73   69   66   65   64
   63   63   64   66   67   67   65   63   61   59   58   57   57   56   56   54
   53   52   54   58   66   75   85   94  102  111  122  136  150  161  168  170
  169  166  162  156  147  135  121  109  100
    27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  123  112  103   95   88   82   79   78   78   78   75   71   68   65   63   62
   61   62   63   66   68   70   72   75   78   78   75   71   67   66   66   66
   66   65   64   65   65   63   61   59   57   55   54   54   54   54   52   51
   49   49   51   57   66   76   87   96  106  116  130  147  165  180  190  194
  195  194  192  188  180  168  153  137  123
    25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  153  137  123  111  100   91   85   82   81   79   76   72   68   65   64   62
   61   62   64   66   68   69   71   74   76   76   73   69   66   66   68   69
   68   66   64   62   61   59   56   54   52   50   49   49   49   49   47   45
   44   45   50   58   68   79   90  101  111  124  139  159  180  198  210  216
  218  219  219  218  214  204  189  171  153
    22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  185  166  149  132  117  104   95   90   86   83   78   73   68   65   64   63
   62   63   65   67   68   69   70   73   75   75   72   68   66   67   70   71
   70   66   62   59   56   53   50   48   46   44   43   43   43   42   41   39
   39   42   49   59   72   84   96  107  119  133  150  170  192  211  225  232
  234  236  239  242  242  235  222  205  185
    20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  215  196  177  158  139  123  109  100   94   88   82   75   69   66   65   65
   65   66   68   70   71   71   71   73   76   76   74   70   68   70   72   73
   70   65   59   54   50   47   44   42   40   38   37   36   36   35   34   33
   35   40   50   63   77   90  103  115  128  143  160  180  200  218  231  238
  241  244  249  255  259  258  249  234  215
    17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  238  221  203  183  163  143  126  113  104   96   87   79   72   68   67   67
   68   70   72   75   75   74   74   76   78   79   77   73   71   72   73   73
   68   61   54   48   44   41   38   36   34   32   30   30   30   30   29   29
   33   41   53   68   84   98  111  124  138  152  168  186  203  219  229  235
  237  240  246  255  263  267  263  253  238
    15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  251  239  224  206  185  163  143  127  115  105   95   85   77   72   70   70
   72   74   78   80   80   79   78   79   81   82   80   77   74   74   73   70
   64   56   49   43   39   36   34   32   30   28   26   25   25   26   26   28
   33   43   58   74   90  104  118  132  146  160  174  188  202  213  221  224
  225  227  234  244  255  263  265  260  251
    12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  254  247  237  222  202  180  159  141  126  115  104   93   83   77   74   73
   75   79   83   86   86   84   82   83   85   85   84   80   77   74   72   67
   60   51   44   39   36   34   31   29   27   25   23   23   23   24   25   28
   35   47   62   79   95  110  124  139  154  167  179  189  198  205  210  211
  210  211  216  226  238  249  255  257  254
    10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  248  247  242  230  213  192  170  152  137  125  114  102   91   83   78   77
   78   82   87   90   90   88   86   86   87   88   86   82   77   73   69   62
   55   47   42   38   35   33   31   29   27   25   23   23   23   24   26   31
   38   51   66   83   98  113  128  144  159  172  182  189  194  198  200  200
  197  196  200  208  220  231  240  246  248
     7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  240  242  240  232  218  199  178  160  145  134  123  112  100   90   83   80
   81   84   89   92   93   91   89   88   88   88   86   82   77   71   65   58
   51   45   41   39   37   34   32   30   27   26   24   24   25   26   29   34
   42   54   69   84   99  114  130  147  163  175  184  189  193  195  196  195
  192  190  191  197  206  216  226  234  240
     5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  233  237  236  230  218  201  182  165  152  142  132  122  110   99   89   84
   83   85   90   93   94   92   89   88   87   87   84   80   74   69   62   55
   49   45   42   41   39   36   34   31   29   28   27   26   27   28   31   36
   45   56   70   83   98  113  130  147  164  177  186  191  194  197  199  198
  196  194  194  197  203  211  219  227  233
     2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  234  236  234  228  216  201  185  169  157  148  140  131  120  108   97   89
   86   87   89   92   92   90   88   86   85   83   81   76   71   66   60   54
   49   46   44   43   41   38   35   32   31   30   29   29   28   30   33   38
   47   57   69   81   95  110  127  146  163  176  187  194  199  204  208  210
  210  209  208  210  214  219  224  230  234
     0.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  244  242  237  228  217  203  187  173  162  154  148  140  130  118  105   96
   90   89   89   90   89   87   84   82   81   79   76   72   68   63   59   54
   50   47   46   45   42   39   36   33   32   32   31   30   30   30   33   40
   48   58   68   79   91  106  123  141  158  173  186  196  205  214  221  227
  230  231  232  234  236  239  241  243  244
    -2.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  262  255  245  234  221  207  192  178  167  159  154  148  139  128  116  105
   97   92   90   88   86   83   80   78   76   74   71   68   64   61   58   55
   52   49   47   45   42   39   36   34   33   33   33   32   30   30   34   40
   49   58   67   76   88  102  118  135  152  167  182  196  210  223  234  243
  250  255  260  263  266  267  267  266  262
    -5.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  284  273  259  244  229  214  200  185  173  165  159  155  148  138  127  116
  106   98   92   87   82   78   76   74   72   70   67   64   62   60   59   57
   53   50   46   43   41   38   37   36   35   35   35   33   31   30   34   41
   50   58   66   75   85   98  112  127  142  158  175  193  211  227  242  255
  266  275  283  290  295  296  295  292  284
    -7.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  306  292  275  257  240  225  209  194  180  170  164  160  155  148  138  127
  117  107   97   89   81   76   72   70   69   66   63   61   60   60   59   58
   55   50   45   41   39   38   38   38   39   38   37   35   32   31   34   42
   51   60   67   74   83   94  106  118  132  148  166  186  207  226  243  258
  272  285  298  309  316  319  319  315  306
   -10.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  320  306  288  269  251  235  219  202  186  174  167  163  160  156  149  139
  129  117  105   93   83   75   71   69   67   64   62   59   59   60   61   59
   56   50   44   40   39   40   42   43   43   43   41   38   34   33   36   43
   53   62   69   75   83   91  100  110  121  136  155  177  198  218  235  251
  267  283  300  314  324  330  331  328  320
   -12.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  323  311  295  277  259  243  226  208  191  176  168  164  163  161  157  149
  140  127  113   99   87   78   73   70   68   65   62   60   60   61   62   61
   57   51   45   41   42   44   48   49   49   48   45   41   37   35   38   46
   55   64   71   77   83   89   96  103  112  126  145  166  186  204  220  235
  252  270  288  305  317  325  329  328  323
   -15.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  312  304  292  277  261  246  229  211  192  176  166  163  162  162  161  156
  148  137  123  108   94   84   78   74   71   67   64   62   61   63   64   63
   59   53   48   45   47   51   55   56   56   53   49   45   41   39   41   48
   58   66   73   78   83   88   93   98  106  119  135  155  173  188  201  214
  229  247  266  283  296  305  311  314  312
   -17.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  289  286  279  268  255  241  226  208  190  174  163  159  159  160  161  159
  154  144  131  117  103   93   85   80   75   71   68   65   64   65   66   65
   62   57   53   52   54   59   62   63   61   57   53   48   44   42   44   50
   59   68   74   79   83   88   92   96  103  114  129  145  160  172  181  192
  204  220  237  253  265  275  282  287  289
   -20.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  258  259  257  250  241  230  217  201  184  168  158  153  153  155  157  157
  154  148  137  125  113  102   93   86   81   76   72   69   68   67   68   67
   65   62   59   59   62   66   68   67   64   59   54   50   45   43   45   51
   60   68   74   79   84   88   92   96  102  112  124  137  148  157  163  171
  180  193  207  220  231  239  246  252  258
   -22.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  223  228  229  227  221  213  203  190  175  161  151  146  145  147  149  151
  150  147  140  131  121  111  101   93   86   80   76   72   70   69   69   69
   68   66   65   65   67   70   70   68   64   58   53   49   45   44   46   51
   59   67   73   78   83   88   92   97  103  111  121  130  139  144  148  153
  160  170  180  190  197  203  209  216  223
   -25.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  190  197  200  201  198  194  186  176  165  154  145  139  137  138  139  141
  142  142  139  134  127  118  108   98   89   82   78   74   72   70   70   70
   70   69   68   68   69   70   69   65   60   55   50   46   43   43   45   50
   58   65   72   77   82   87   92   97  103  109  116  124  130  134  137  140
  144  151  158  164  169  173  177  183  190
   -27.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  163  169  174  176  176  174  170  163  155  146  139  133  130  129  129  130
  132  134  135  134  129  121  111   99   90   82   78   74   72   70   69   69
   69   69   69   68   67   66   63   58   53   49   45   42   40   40   43   49
   57   64   70   75   80   86   91   96  100  105  110  115  120  124  127  129
  132  137  141  145  148  150  153  157  163
   -30.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  144  148  152  155  156  156  154  151  146  140  134  129  125  122  120  120
  121  124  128  130  128  121  110   98   88   80   75   73   70   69   67   67
   67   67   66   64   61   58   54   50   46   43   40   38   37   38   42   48
   56   63   69   73   78   83   88   92   95   98  101  105  109  114  117  120
  123  125  129  132  134  135  137  140  144
   -32.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  131  133  136  139  141  142  142  141  138  135  131  126  122  117  113  111
  111  115  120  124  124  118  108   95   84   76   72   69   68   66   65   64
   63   62   60   57   53   50   46   42   40   38   36   35   35   37   42   49
   56   63   68   72   76   80   83   86   88   88   90   93   97  103  107  111
  114  116  118  121  124  126  127  129  131
   -35.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  123  124  125  127  129  131  133  133  132  131  128  125  121  115  110  106
  105  107  113  117  118  114  104   91   80   72   68   67   66   65   63   62
   60   58   54   50   46   42   39   37   36   35   35   34   34   37   43   50
   58   64   68   71   73   76   78   79   79   78   78   81   86   92   97  102
  104  106  109  113  117  120  122  123  123
   -37.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  118  117  117  118  121  123  125  127  127  127  126  124  121  115  109  104
  101  103  108  112  114  110  100   88   77   69   66   65   65   64   63   61
   58   54   50   45   41   38   36   36   37   37   36   36   36   39   46   53
   61   67   69   71   71   72   73   73   71   69   69   71   76   82   88   92
   95   97  100  105  110  115  118  118  118
   -40.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  113  111  110  111  113  116  119  121  123  124  124  123  121  116  110  105
  101  102  106  110  111  108   99   88   77   70   67   66   67   66   65   62
   58   53   48   43   39   37   38   39   40   41   40   39   40   43   49   57
   64   69   71   71   70   69   69   68   66   64   63   65   69   76   81   84
   86   88   91   97  104  110  113  114  113
   -42.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  107  104  103  104  107  110  113  115  117  119  120  121  120  117  112  107
  104  104  107  111  112  109  101   91   81   74   71   70   71   71   69   66
   61   56   50   45   42   41   43   45   47   47   46   45   45   48   54   61
   67   71   72   71   70   69   68   67   65   63   63   64   68   73   77   79
   80   81   84   90   97  104  108  108  107
   -45.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  100   98   96   97  100  103  106  109  111  113  116  118  119  118  115  111
  109  109  111  114  115  112  106   97   88   81   78   77   77   77   75   71
   66   61   55   50   48   48   49   52   54   54   52   50   50   52   58   64
   70   73   73   72   71   70   70   69   69   68   68   69   72   75   77   78
   77   77   80   85   92   98  101  102  100
   -47.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   94   91   90   90   93   96   99  102  104  107  110  113  115  116  115  114
  112  113  116  118  120  118  112  104   96   90   86   84   84   83   81   77
   72   67   61   57   55   55   56   58   59   59   57   55   54   56   61   67
   72   74   75   74   73   73   74   75   76   77   77   78   80   81   81   80
   78   77   79   83   88   93   96   96   94
   -50.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   89   86   84   85   87   90   93   95   97  100  104  107  111  113  114  114
  115  116  119  122  124  123  119  112  105   98   94   92   90   89   86   82
   77   72   67   63   61   60   62   63   63   62   60   58   58   60   64   69
   73   75   76   75   75   77   79   82   85   87   88   89   89   89   87   84
   82   80   81   83   87   91   92   91   89
   -52.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   87   84   82   82   83   86   88   90   92   95   98  101  105  108  111  112
  114  117  120  124  126  126  123  117  111  105  100   96   94   92   89   85
   80   75   70   67   64   64   64   64   64   63   61   60   60   62   66   71
   74   76   77   78   79   82   85   90   94   97   98   99   98   96   93   90
   87   85   85   87   90   92   92   90   87
   -55.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   88   85   82   82   83   84   86   87   89   91   93   96   99  103  106  108
  111  115  119  122  125  125  123  119  113  108  103   98   95   92   89   85
   81   76   72   68   66   64   64   64   63   62   61   61   62   64   68   72
   75   78   79   80   83   86   91   96  100  104  105  105  104  102   99   96
   93   92   92   93   95   95   94   92   88
   -57.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   93   89   87   85   85   86   86   87   88   89   90   92   94   97  100  102
  106  109  114  117  120  121  120  117  112  107  102   98   94   90   87   83
   79   75   71   67   65   63   62   62   61   61   61   61   63   66   70   74
   77   79   81   83   86   90   94   99  104  107  109  109  107  105  102  100
   98   98   98  100  101  101  100   97   93
   -60.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  101   97   94   92   91   90   90   89   89   89   89   89   90   92   94   96
   99  103  106  110  113  114  114  112  108  104   99   95   91   88   84   81
   77   73   69   66   63   62   60   60   59   60   60   62   65   68   72   76
   79   81   83   85   88   92   96  100  104  107  108  108  107  105  104  103
  102  103  105  107  108  108  107  104  101
   -62.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  110  106  103  100   98   96   95   93   92   90   89   88   88   88   89   90
   93   95   99  102  105  106  107  105  103   99   96   92   88   85   82   78
   75   72   68   65   63   61   59   59   59   60   61   63   66   70   74   77
   80   82   84   86   89   92   95   99  102  104  105  105  105  104  104  105
  106  108  111  114  115  116  115  113  110
   -65.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  119  116  112  109  106  104  101   99   96   93   91   89   87   86   86   86
   87   89   92   95   97   99  100   99   98   95   93   90   87   84   81   78
   75   72   69   66   63   61   60   59   60   61   62   65   68   71   74   77
   80   82   84   86   88   90   93   96   98  100  101  102  102  103  104  106
  109  112  116  119  122  123  123  121  119
   -67.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  126  124  121  117  114  110  107  104  100   96   93   90   87   85   84   84
   84   86   87   90   92   93   94   95   94   93   91   89   87   84   82   79
   76   74   71   68   66   64   62   62   61   62   64   66   68   71   74   76
   78   80   82   84   85   87   89   91   93   95   97   98  100  102  104  107
  111  116  120  124  127  129  129  128  126
   -70.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  131  129  126  123  119  116  112  108  104  100   96   92   89   87   85   84
   84   84   85   87   89   90   91   92   92   92   91   90   88   86   84   82
   79   77   74   71   69   67   65   64   64   64   65   66   68   69   71   73
   75   77   78   80   81   83   85   87   89   91   93   95   98  101  104  108
  113  118  122  126  130  132  133  133  131
   -72.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  133  131  128  125  122  118  114  110  106  102   98   94   91   88   86   85
   85   85   85   87   88   89   90   91   92   92   92   91   90   89   87   85
   82   80   77   75   72   70   68   66   65   64   64   65   66   67   68   69
   71   72   73   75   76   78   80   82   84   87   90   93   96  100  104  108
  113  118  123  127  130  132  133  133  133
   -75.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  130  129  127  124  121  118  114  110  106  103   99   96   93   90   89   87
   87   86   87   87   88   89   90   91   92   92   92   92   91   90   88   86
   84   82   79   76   74   71   69   67   65   64   63   63   63   63   64   65
   66   67   68   69   71   73   75   78   80   83   86   90   94   98  102  107
  111  116  120  124  127  129  130  130  130
   -77.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  123  123  121  119  117  114  111  108  105  102   99   96   94   92   90   89
   88   88   88   88   89   89   90   90   91   91   91   90   90   89   87   85
   83   81   79   76   73   71   69   66   65   63   62   61   60   60   60   61
   61   62   63   65   67   69   71   74   76   80   83   87   90   95   99  103
  107  111  114  118  120  122  123  124  123
   -80.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  114  113  113  111  110  108  106  104  101   99   97   95   93   92   90   89
   89   88   88   88   88   88   88   88   88   88   88   87   86   85   84   82
   80   78   76   74   72   69   67   65   63   62   61   59   59   58   58   58
   59   60   61   62   64   66   68   70   73   76   79   83   86   90   93   97
  100  103  106  108  111  112  113  114  114
   -82.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
  102  102  102  101  101  100   98   97   96   94   93   92   90   89   88   88
   87   86   86   86   85   85   85   84   84   84   83   82   81   80   79   78
   76   75   73   71   69   68   66   64   63   61   60   59   59   58   58   58
   59   59   60   61   63   64   66   68   71   73   76   78   81   84   86   89
   91   94   96   98   99  101  102  102  102
   -85.0-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   91   91   91   91   91   90   90   89   89   88   87   87   86   85   85   84
   83   83   82   82   81   81   80   80   79   79   78   77   76   75   74   73
   72   71   70   69   68   67   66   65   64   63   62   62   61   61   61   61
   61   62   62   63   64   65   67   68   69   71   73   74   76   78   80   81
   83   84   86   87   88   89   90   90   91
   -87.5-180.0 180.0   5.0 450.0                            LAT/LON1/LON2/DLON/H
   80   81   81   81   81   81   81   81   81   80   80   80   80   79   79   79
   78   78   78   77   77   76   76   76   75   75   74   74   73   72   72   71
   71   70   70   69   69   68   68   67   67   67   66   66   66   66   66   66
   66   67   67   67   68   68   69   70   70   71   72   72   73   74   75   76
   76   77   78   78   79   79   80   80   80
     3                                                      END OF TEC MAP      

"""
REF_DATE = PreciseDateTime.from_utc_string("08-JAN-2019 08:32:54.152948000000")


class IonosphereTest(unittest.TestCase):
    """Testing atmospheric/ionosphere.py functionalities"""

    def setUp(self) -> None:
        # creating test data
        self.tolerance = 1e-9
        self.timestamp = "  2019     1     8     0     0     0                        EPOCH OF CURRENT MAP"
        self.fc_hz = 5405000454
        self.vect_1 = np.array([-5132493.467464049, 3213981.721987821, -3139087.972193046])
        self.vect_2 = np.array([-80654.82146094833, 263291.61416462297, -156134.995438213])
        self.sat_pos = np.array(
            [
                [-5213148.288925, 3477273.33615244, -3295222.96763126],
                [-5209063.04576642, 3470355.46273472, -3308970.22887301],
                [-5204333.07709188, 3462384.33554006, -3324751.41275261],
                [-5197791.26572232, 3451426.28423579, -3346343.6922817],
                [-5193770.37483127, 3444728.34929901, -3359483.76228291],
                [-5197116.11248636, 3450299.65927584, -3348556.97423551],
                [-5202031.87656563, 3458520.94424205, -3332377.50336732],
                [-5204651.7435651, 3462920.08385765, -3323692.71608399],
                [-5209902.10503636, 3471773.77145256, -3306155.65055151],
                [-5215141.37420304, 3480659.77578179, -3288475.91262992],
            ]
        )
        self.target_coords = np.array(
            [
                [-4989394.044, 2746844.389, -2862070.09],
                [-4987723.092, 2737761.662, -2873635.587],
                [-4982121.114, 2732288.807, -2888334.621],
                [-4973496.308, 2726074.177, -2908844.803],
                [-4963032.78, 2729484.882, -2923421.927],
                [-4961686.017, 2740588.588, -2915329.591],
                [-4964674.991, 2750065.423, -2901345.385],
                [-4973836.181, 2744662.788, -2890816.802],
                [-4983429.822, 2746023.792, -2873110.56],
                [-4979009.54, 2766786.057, -2860862.575],
            ]
        )
        # expected results
        self.expected_timestamp = "2019-01-08 00:00"
        self.expected_angle = 0.6255869960711915
        self.expected_delay = np.array(
            [
                0.21439696172556352,
                0.2143296921425675,
                0.21325451411033114,
                0.21153831400005008,
                0.20911221486367682,
                0.2084489839549282,
                0.2087897299479023,
                0.2109815991792891,
                0.21307718390451413,
                0.21139986229548638,
            ]
        )

    def test_epoch_timestamp_formatter(self) -> None:
        """Testing ionosphere _epoch_timestamp_formatter function"""
        timestamp = iono._epoch_timestamp_formatter(self.timestamp)
        self.assertEqual(timestamp, self.expected_timestamp)

    def test_angle_between_vectors(self) -> None:
        """Testing ionosphere _angle_between_vectors function"""
        angle = iono._angle_between_vectors(self.vect_1, self.vect_2)
        np.testing.assert_allclose(angle, self.expected_angle, atol=self.tolerance, rtol=0)

    def test_ionospheric_delay_computation(self) -> None:
        """Testing ionosphere compute_delay function"""
        with TemporaryDirectory() as tmpdir:
            map_file = Path(tmpdir).joinpath("corg0080.19i")
            map_file.write_text(TEST_IONOSPHERE_MAP, encoding="UTF-8")
            delay = iono.compute_delay(
                acq_time=REF_DATE,
                analysis_center=iono.IonosphericAnalysisCenters.COR,
                sat_xyz_coords=self.sat_pos,
                targets_xyz_coords=self.target_coords,
                fc_hz=self.fc_hz,
                map_folder=tmpdir,
            )
            np.testing.assert_allclose(delay, self.expected_delay, atol=self.tolerance, rtol=0)

    def test_ionospheric_delay_computation_wrong_analysis_center_error(self) -> None:
        """Testing ionosphere compute_delay function, wrong analysis center error"""
        with self.assertRaises(iono.WrongAnalysisCenterNameError):
            iono.compute_delay(
                acq_time=REF_DATE,
                analysis_center="TEST",
                sat_xyz_coords=self.sat_pos,
                targets_xyz_coords=self.target_coords,
                fc_hz=self.fc_hz,
                map_folder="",
            )

    def test_ionospheric_delay_computation_map_not_found_error(self) -> None:
        """Testing ionosphere compute_delay function, invalid map path"""
        with self.assertRaises(iono.IonosphericMapFileNotFoundError):
            iono.compute_delay(
                acq_time=REF_DATE,
                analysis_center=iono.IonosphericAnalysisCenters.COR,
                sat_xyz_coords=self.sat_pos,
                targets_xyz_coords=self.target_coords,
                fc_hz=self.fc_hz,
                map_folder="",
            )


if __name__ == "__main__":
    unittest.main()
