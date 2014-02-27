pro add_block,name,x,blockid,debug=debug,new=new,SWAP_ENDIAN=SWAP_ENDIAN,$
              double=double
   IF IS_DEF(double) THEN BEGIN
      bln=N_ELEMENTS(x)*8L+8L 
   END ELSE BEGIN
      bln=N_ELEMENTS(x)*4L+8L
   END   
   b4n=BYTE(blockid+"    ")
   b4n=b4n(0:3)

   IF IS_DEF(debug) THEN print,bln,':',STRING(b4n)
   pos=0LL
   b4=BYTARR(4)
   bl=0L

   get_lun,myfile

   IF NOT(IS_DEF(new)) THEN BEGIN
      openr,myfile,name
      readu,myfile,bl
      close,myfile

      IF bl EQ 8 THEN BEGIN
         openu,myfile,name,/f77 
         IF IS_DEF(debug) THEN print,'Open file in normal mode ..,'
      END ELSE BEGIN
         openu,myfile,name,/f77,/SWAP_ENDIAN
         IF IS_DEF(debug) THEN print,'Open file with SWAP_ENDIAN ..,'
      END
;       openu,myfile,name,/f77,SWAP_ENDIAN=SWAP_ENDIAN 
   END ELSE BEGIN
       openw,myfile,name,/f77,SWAP_ENDIAN=SWAP_ENDIAN
   END

   IF NOT(IS_DEF(new)) THEN BEGIN
      WHILE(NOT(EOF(myfile))) DO BEGIN
         readu,myfile,b4,bl
         thislabel=STRING(b4)
         IF IS_DEF(debug) THEN print,thislabel," <=> ",STRING(b4n),bl
         point_lun,-myfile,pos
         point_lun,myfile,long64(pos)+long64(bl)
      ENDWHILE
   END

   writeu,myfile,b4n,bln
   writeu,myfile,x
   close,myfile

   free_lun,myfile
END   
