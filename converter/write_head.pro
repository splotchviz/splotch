pro write_head,name,h,SWAP_ENDIAN=SWAP_ENDIAN
   bl=264L

   get_lun,myfile
   b4=BYTE("HEAD    ")
   b4=b4(0:3)

   openw,myfile,name,/f77,SWAP_ENDIAN=SWAP_ENDIAN
   writeu,myfile,b4,bl
   writeu,myfile,h.npart,h.massarr,h.time,h.redshift,h.flag_sfr,h.flag_feedback,$
                 h.partTotal,h.flag_cooling,h.num_files,$
                 h.BoxSize,h.Omega0,h.OmegaLambda,h.HubbleParam,h.la
   close,myfile

   free_lun,myfile
END   
