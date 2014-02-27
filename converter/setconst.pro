;*****************************************************************
;*** Festlegung wichtiger physikalischer Konstannten *************
;*****************************************************************
COMMON kosmos,hbpar,Ompar,Lampar,wpar

   rgas   = 8.31425d+7
;           [cgs]
   mparsck= 3.085678d24
;           [ cm ]
   solmas = 1.989d33
;           [ g ]
   Lsol   = 1.0d+44
;           [ erg/s ]
   me     = 9.108d-28
;           [ g ]
   mec2_gev = 0.51092111
;           [ GeV ]
   qe     = 4.803d-10
;           [ Sqrt(erg*cm) ]
   qe_cgs = 4.803250e-10
;           [ Sqrt(erg*cm) ]
   mp     = 1.6726231e-24
;           [ g ]
   g      = 6.672d-8
;           [ cm/s^2 ]
   c      = 2.99792458d+10
;           [ cm/s ]
   k      = 8.61d-8
;           [ keV/K ]
   kcgs   = 1.380658d-16
;           [ erg/K ]  
   hq     = 1.05457266d-27
;           [ erg * s]
   hcgs   = 6.6260754d-27
;           [ erg * s]
   sigmat = 6.653d-25
;           [ cm^2 ]
   sigmat_mbarn = 665.3
;           [ mbarn ]
   sigmapp_mbarn= 32
;           [ cm^2 ]
   ergproeV=1.60217733d-12
;           [ erg/eV ]
   Jansky = 1.0d-23
;           [ erg / s / Hz / cm^2 ]
   watt2erg = 1e7
;           [ erg / watt ]

   Tback = 2.73
;           [ K ]
;
   pi     = 4.*ATAN(1.)
   fpg    = 4.*pi*g
;
   GYEAR  = 3.14712e+16
;           [ s ]

;
; Slope of N(E)=N0*E^-gam
   gam=3.5d0
;   gam=3.68d0
;   gam=2.54d0

; P-Madget Units
   l_unit = 3.085678d21        ;  1.0 kpc /h
   m_unit = 1.989d43           ;  10^10 solar masses
   v_unit = 1d5                ;  1 km/sec
   t_unit = l_unit / v_unit
   e_unit = m_unit * l_unit^2 / t_unit^2

; Solar abudances
   zsol=[3.3345e-3, $ ;dummy Anders & Grevesse,  1989 3.77e-3
         3.974e-3,  $ ; #C    3.03e-3
         1.164e-3,  $ ; #N    1.11e-3
         7.84e-3,   $ ; #O     9.59e-3
         9.125e-4,  $ ; #Mg   5.15e-4
         1.005e-3,  $ ; #Si   7.11e-4
         1.77e-3    $ ; #Fe    1.27e-3
         ]
; Vine units
l_vine = 1.07998809d22        
m_vine = 1.11384d44           
v_vine = 26.2d6    
t_vine = l_vine / v_vine
e_vine = m_vine * l_vine^2 / t_vine^2

