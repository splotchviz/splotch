; example:
; 
; Reading the header information:
; readnew,'snap_000',head,'HEAD',type='HEAD'
; print,head.npart
;
; Reading the Poisitions:
; readnew,'snap_000',x,'POS ',type='FLOAT3'
; print,'X-Range:',min(x(0,*)),max(x(0,*))
; print,'Y-Range:',min(x(1,*)),max(x(1,*))
;
; Reading the Masses
; readnew,'snap_000',m,'MASS'
; ATTENTION! GADGET allows to save space by storing the masses
;            in the massarr if all particle of a certain species
;            have the same mass !!!!
;            Therefore its recomendet to have a look at the header first !        
; readnew,'snapo_000',head,'HEAD',type='HEAD'
; if head.massarr(0) EQ 0 then print,m(0) else print,head.massarr(0)


FUNCTION IS_DEF, x
aux = SIZE(x)
RETURN, aux(N_ELEMENTS(aux)-2) NE 0
END

PRO readhead,myfile,h,debug=debug
   IF IS_DEF(debug) THEN print,'Reading HEADER ...'
   npart=lonarr(6)	
   massarr=dblarr(6)
   time=0.0D
   redshift=0.0D
   flag_sfr=0L
   flag_feedback=0L
   partTotal=lonarr(6)
   flag_cooling=0L
   num_files=0L
   BoxSize=0.0D
   Omega0=0.0D
   OmegaLambda=0.0D
   HubbleParam=0.0D

   bytesleft=256-6*4 - 6*8 - 8 - 8 - 2*4 - 6*4 - 2*4 - 4*8
   la=bytarr(bytesleft)

   readu,myfile,npart,massarr,time,redshift,flag_sfr,flag_feedback,partTotal, $
                flag_cooling,num_files,BoxSize,Omega0,OmegaLambda,HubbleParam,la

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
END

PRO myopen,name,myfile,debug=debug
   bl=0L
   IF IS_DEF(debug) THEN print,'Testing File ',name
   openr,myfile,name
   readu,myfile,bl
   IF IS_DEF(debug) THEN print,'First Block = ',bl
   close,myfile

   IF bl EQ 256 THEN BEGIN
      openr,myfile,name,/f77 
      IF IS_DEF(debug) THEN print,'Open file in normal mode ...'
   END ELSE BEGIN
      openr,myfile,name,/SWAP_ENDIAN
      readu,myfile,bl
      IF IS_DEF(debug) THEN print,'First Block = ',bl
      close,myfile
      IF bl EQ 256 THEN BEGIN
         openr,myfile,name,/f77,/SWAP_ENDIAN
         IF IS_DEF(debug) THEN print,'Open file with SWAP_ENDIAN ...'
      END ELSE BEGIN
         print,'Wrong HEADER in ',name,' !',bl
      END
   END
END

PRO readold,name,h,x,v,id=id,m=m,u=u,hsml=hsml,rho=rho,debug=debug,$
            chem=chem,xne=xne,xnh=xnh,inifile=inifile,sfr=sfr,age=age,$
            stars=stars,b=b,readb=readb
@setconst
   status=-1
   IF NOT(IS_DEF(name)) THEN BEGIN
      print,'Usage: readnew,name,x,label,type=type,debug=debug'
      print,'       name  : filename'
      print,'       x     : variable containing the result'
      print,'       label : names the block identifier ("HEAD","POS ",...)'
      print,'       type  : names the data type ("FLOAT","FLOAT3","LONG","HEAD")'
      print,'       debug : give some debug information'
      return
   END

   get_lun,myfile
   myname=name
   numfiles=1

   ss=SIZE(FINDFILE(myname))
   IF ss(0) EQ 0 THEN BEGIN
      myname=name+'.0'
      ss=SIZE(FINDFILE(myname))
      IF ss(0) EQ 0 THEN BEGIN
         print,'Cant find file ',name,' or ',myname,' ...'
         stop
      END ELSE BEGIN
         npart=lonarr(6)

         myopen,myname,myfile,debug=debug
         readhead,myfile,h
         CLOSE,myfile
         npart=npart+h.npart
         WHILE ss(0) NE 0 DO BEGIN
            myname=name+'.'+strcompress(string(numfiles),/remove_all)
            ss=SIZE(FINDFILE(myname))
            IF ss(0) NE 0 THEN BEGIN
               numfiles=numfiles+1
               myopen,myname,myfile,debug=debug
               readhead,myfile,h
               CLOSE,myfile
               npart=npart+h.npart
            END
         END
         IF IS_DEF(debug) THEN print,'Found',numfiles,' files ...'
         IF IS_DEF(debug) THEN print,'TotNP',npart
         
         h.npart(*)=npart(*)
         ntot=npart(0)+npart(1)+npart(2)+npart(3)+npart(4)+npart(5)
         x=fltarr(3,ntot)
         v=fltarr(3,ntot)
         id=lonarr(ntot)

         istart=lonarr(6) 
         istart(0)=0L
         FOR j=1,5 DO BEGIN
            istart(j)=istart(j-1)+h.npart(j-1)
         END

         FOR i=0,numfiles-1 DO BEGIN
            myname=name+'.'+strcompress(string(i),/remove_all)
            myopen,myname,myfile,debug=debug
            readhead,myfile,htmp
            ntot=htmp.npart(0)+htmp.npart(1)+htmp.npart(2)+htmp.npart(3)+htmp.npart(4)+htmp.npart(5)

            xtmp=fltarr(3,ntot)
            vtmp=fltarr(3,ntot)
            idtmp=fltarr(ntot)

            IF IS_DEF(debug) THEN print,'Reading POSITIONS ...'
            READU,myfile,xtmp

            IF IS_DEF(debug) THEN print,'Reading VELOCITIES ...'
            READU,myfile,vtmp

            IF IS_DEF(debug) THEN print,'Reading IDs ...'
            READU,myfile,idtmp

            CLOSE,myfile
            itmp=0
            FOR j=0,5 DO BEGIN
               IF h.npart(j) GT 0 THEN BEGIN
                  x(*,istart(j):istart(j)+htmp.npart(j)-1)=xtmp(*,itmp:itmp+htmp.npart(j)-1)
                  v(*,istart(j):istart(j)+htmp.npart(j)-1)=vtmp(*,itmp:itmp+htmp.npart(j)-1)
                  id(istart(j):istart(j)+htmp.npart(j)-1)=idtmp(itmp:itmp+htmp.npart(j)-1)
                  istart(j)=istart(j)+htmp.npart(j)
                  itmp=itmp+htmp.npart(j)
               END
            END
         END
      END
   END ELSE BEGIN
      myopen,myname,myfile,debug=debug
      readhead,myfile,h
      ntot=h.npart(0)+h.npart(1)+h.npart(2)+h.npart(3)+h.npart(4)+h.npart(5)
jj=WHERE(h.massarr EQ 0)
nmass=0
nbar=h.npart(0)
IF jj(0) GT -1 THEN nmass=ROUND(TOTAL(h.npart(jj)))
      x=fltarr(3,ntot)
      v=fltarr(3,ntot)
      id=lonarr(ntot)
      IF nmass GT 0 THEN m=fltarr(nmass)
      IF nbar GT 0 THEN BEGIN
         u=fltarr(nbar)
         hsml=fltarr(nbar)
         rho=fltarr(nbar)
         IF IS_DEF(chem) THEN BEGIN
            dummy=fltarr(nbar)
            xne=fltarr(nbar)
            xnh=fltarr(nbar)
         END
         IF IS_DEF(stars) THEN BEGIN
            sfr=fltarr(h.npart(4))
            age=fltarr(h.npart(4))
         END
         IF IS_DEF(readb) THEN BEGIN
            b=fltarr(3,nbar)
         END
      END
      IF IS_DEF(debug) THEN print,'Reading POSITIONS ...'
      READU,myfile,x

      IF IS_DEF(debug) THEN print,'Reading VELOCITIES ...'
      READU,myfile,v

      IF IS_DEF(debug) THEN print,'Reading ids ...'
      READU,myfile,id

      IF nmass GT 0 THEN BEGIN
         IF IS_DEF(debug) THEN print,'Reading MASSES ...'
         READU,myfile,m
      END

      IF (h.npart(0) GT 0) THEN BEGIN
         IF IS_DEF(debug) THEN print,'Reading INTERNAL ENERGY ...'
         READU,myfile,u
      END

      IF (h.npart(0) GT 0) AND NOT(IS_DEF(inifile)) THEN BEGIN
         IF IS_DEF(debug) THEN print,'Reading DENSITY ...'
         READU,myfile,rho

         IF IS_DEF(chem) THEN BEGIN
            IF IS_DEF(debug) THEN print,'Reading NE ...'
            READU,myfile,xne
            IF IS_DEF(debug) THEN print,'Reading NH ...'
            READU,myfile,xnh
;            FOR idum=2,8 DO BEGIN
;               READU,myfile,dummy
;               print,idum,min(dummy),max(dummy)
;            END
         END

         IF IS_DEF(debug) THEN print,'Reading HSML ...'
         READU,myfile,hsml

         IF IS_DEF(stars) THEN BEGIN
            IF IS_DEF(debug) THEN print,'Reading SFR ...'
            READU,myfile,sfr
;            IF IS_DEF(debug) THEN print,'Reading AGE ...'
;            READU,myfile,age
        END

        IF IS_DEF(readb) THEN BEGIN
            IF IS_DEF(debug) THEN print,'Reading B ...'
            READU,myfile,b
        END
      END
      CLOSE,myfile
   END
   free_lun,myfile
 END
   
PRO gadget1_to_gadget2,name

   readold,name,h,x,v,/debug,u=u,m=m,id=id,hsml=hsml,rho=rho,xne=xne,xnh=xnh,sfr=sfr,age=age,/stars,/chem,b=b,/readb
   fnamenew='./'+name+'.new'

   print,'Writing ...'
   write_head,fnamenew,h

   add_block,fnamenew,x,'POS '
   add_block,fnamenew,v,'VEL '
   add_block,fnamenew,id,'ID  '
   IF IS_DEF(m) THEN add_block,fnamenew,m,'MASS'
   add_block,fnamenew,u,'U   '
   add_block,fnamenew,rho,'RHO '
   add_block,fnamenew,xne,'NE  '
   add_block,fnamenew,xnh,'NH  '
   add_block,fnamenew,hsml,'HSML'
   add_block,fnamenew,sfr,'SFR '
   add_block,fnamenew,age,'AGE '
   add_block,fnamenew,b,'BFLD'

;   stop

END
