;; lies die Daten eines 3d-KH-Modells ein.

IF ( n_elements ( out ) EQ 0 ) THEN BEGIN
    ;; moeglich: 50, 150, 200, 250, 300, 350, 400, 450, 500, 900
    out0 = 500
ENDIF ELSE BEGIN
    out0 = out
ENDELSE

path='/afs/mpa/temp/mobergau/Daten/render/KH/'

liesrh = 1
liespg = 1
liesb  = 1
liesv  = 1

mx = 256
my = 256
mz = 256

s3d = [ mx, my, mz ]
v3d = [ 3, mx, my, mz ]

x1d = darray ( -1.,  1.,  mx )
y1d = darray ( -1.,  1.,  my )
z1d = darray ( -1.,  1.,  mz )

x3d = rebin ( x1d, s3d )
y3d = transpose ( rebin ( y1d, [ my, mx, mz] ), [1, 0, 2] )
z3d = transpose ( rebin ( z1d, [ mz, mx, my] ), [1, 2, 0] )

x = dblarr ( v3d )
x[0,*,*,*] = x3d[*,*,*]
x[1,*,*,*] = y3d[*,*,*]
x[2,*,*,*] = z3d[*,*,*]

;; lies die einzelnen Dateien ein

nummer = string ( out0,  FORMAT='(i8.8)' ) 

IF ( liesrh GT 0 ) THEN BEGIN
    ;; ... Dichte
    filename = strcompress ( path+'/d' + nummer + '.rh.dat', /remo ) 
    rh = dblarr ( s3d )
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, rh
    free_lun, lun
    print, ' rho '
ENDIF

IF ( liespg GT 0 ) THEN BEGIN
    ;; ... Druck
    filename = strcompress ( path+'/d' + nummer + '.pg.dat', /remo ) 
    pg = dblarr ( s3d )
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, pg
    free_lun, lun
    print, ' pgas '
ENDIF

IF ( liesb GT 0 ) THEN BEGIN
    ;; ... Magnetfeld
    ;;     ... bx
    filename = strcompress ( path+'/d' + nummer + '.bx.dat', /remo ) 
    b = dblarr ( v3d )
    bt = dblarr ( s3d )
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, bt
    b[0,*,*,*]=bt[*,*,*]
    free_lun, lun
    ;;     ... by
    filename = strcompress ( path+'/d' + nummer + '.by.dat', /remo ) 
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, bt
    b[1,*,*,*]=bt[*,*,*]
    free_lun, lun
    ;;     ... bz
    filename = strcompress ( path+'/d' + nummer + '.bz.dat', /remo ) 
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, bt
    b[2,*,*,*]=bt[*,*,*]
    free_lun, lun
    print,  ' b '
ENDIF
    
IF ( liesv GT 0 ) THEN BEGIN
    ;; ... Geschwindigkeit
    ;;     ... vx
    filename = strcompress ( path+'/d' + nummer + '.vx.dat', /remo ) 
    v = dblarr ( v3d )
    vt = dblarr ( s3d )
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, vt
    v[0,*,*,*]=vt[*,*,*]
    free_lun, lun
    ;;     ... vy
    filename = strcompress ( path+'/d' + nummer + '.vy.dat', /remo ) 
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, vt
    v[1,*,*,*]=vt[*,*,*]
    free_lun, lun
    ;;     ... vz
    filename = strcompress ( path+'/d' + nummer + '.vz.dat', /remo ) 
    openr, lun, filename, /get_lun, f77_u = 0, swap_if_little_endian = 1
    readu, lun, vt
    v[2,*,*,*]=vt[*,*,*]
    free_lun, lun
    print,  ' v '
ENDIF



; Define Gadget Header (kind of complicated)
   npart=lonarr(6)	
   npart(0)= long(mx)*long(my)*long(mz)
   massarr=dblarr(6)
   time=0.0D
   redshift=0.0D
   flag_sfr=0L
   flag_feedback=0L
   partTotal=lonarr(6)
   partTotal(0)=npart(0)
   flag_cooling=0L
   num_files=1L
   BoxSize=2.0D
   Omega0=0.0D
   OmegaLambda=0.0D
   HubbleParam=0.0D

   bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4 - 6*4 - 2*4 - 4*8
   la=bytarr(bytesleft)

   h = { head , npart:npart,$
                massarr:massarr,$
                time:time,$
                redshift:redshift,$
                flag_sfr:flag_sfr,$
                flag_feedback:flag_feedback,$
                partTotal:partTotal,$
		flag_cooling:flag_cooling,$
                num_files:num_files,$
                BoxSize:BoxSize,$
                Omega0:Omega0,$
                OmegaLambda:OmegaLambda,$
                HubbleParam:HubbleParam,$
                la:la}

   fnamenew='snap_' + nummer

; need to define smoothing length
   hsml=fltarr(npart(0))
   hsml(*)=2./mx   

   write_head,fnamenew,h
   add_block,fnamenew,float(x),'POS '
   add_block,fnamenew,float(v),'VEL '
   add_block,fnamenew,float(pg),'U   '
   add_block,fnamenew,float(rh),'RHO '
   add_block,fnamenew,float(hsml),'HSML'
   add_block,fnamenew,float(b),'BFLD'






Ende:
END

