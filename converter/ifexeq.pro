FUNCTION IFEXEQ, var0, $ 
                 nx, ny, nz, $ 
                 noerrormessage=noerrormessage, $ 
                 default=default, $ 
                 nx=nx0, ny=ny0, nz=nz0, $ 
                 integer=integer0, long=long0, shortinteger=shortinteger0, $ 
                 cuttodefault=cuttodefault,  append_def=append_def0
   
    compile_opt hidden

 
    IF ( n_elements ( default ) EQ 0 ) THEN BEGIN 
        default = 0. 
        default_given = 0 
    ENDIF ELSE BEGIN 
        default_given = 1 
    ENDELSE 
 
    IF ( N_ELEMENTS (cuttodefault) EQ 0 ) THEN cuttodefault =  0 
    IF ( N_ELEMENTS (append_def0) EQ 0 ) THEN append_def =  0 ELSE append_def = 1

 
    IF n_elements ( nx ) eq 0 then nx = n_elements ( default ) 
    IF n_elements ( ny ) eq 0 then ny = 1 
    IF n_elements ( nz ) eq 0 then nz = 1
 
    IF N_ELEMENTS( nx0 ) NE 0 THEN nx = nx0 
    IF N_ELEMENTS( ny0 ) NE 0 THEN ny = ny0 
    IF N_ELEMENTS( nz0 ) NE 0 THEN nz = nz0
 
    IF N_ELEMENTS( long0 ) NE 0 THEN BEGIN 
        long = long0 
    ENDIF ELSE BEGIN 
        long = 0 
    ENDELSE 
    IF N_ELEMENTS( integer0 ) NE 0 THEN BEGIN 
        integer = integer0 
    ENDIF ELSE BEGIN 
        integer = 0 
    ENDELSE 
    long = long OR integer 
    IF N_ELEMENTS( shortinteger0 ) NE 0 THEN BEGIN 
        shortinteger = shortinteger0 
    ENDIF ELSE BEGIN 
        shortinteger = 0 
    ENDELSE 
 
    if ( n_elements ( noerrormessage ) eq 0 ) then begin 
        noerrormessage = 1 
    endif else begin 
        noerrormessage = noerrormessage ne 0 
    endelse
 
    n_var0 = N_ELEMENTS(var0) 
    n_def  = N_ELEMENTS(default) 
 
    IF n_var0 NE 0 THEN BEGIN 
        IF ( ( default_given EQ 0 ) OR ( n_var0 EQ n_def ) ) THEN BEGIN 
            var1 = var0 
        ENDIF ELSE $ 
          IF ( ( append_def GT 0 ) AND ( n_def GT n_var0 ) ) THEN BEGIN 
            rest = default [ n_var0 : n_def - 1 ] 
            var1 = [ var0, rest ] 
        ENDIF ELSE $ 
          IF default_given EQ 1 THEN BEGIN 
            IF ( cuttodefault EQ 0 ) THEN BEGIN 
                var1 = var0 [ 0 : n_var0 - 1 ] 
                IF ( n_var0 GT 1 ) THEN BEGIN 
                    dimens = size ( var0, /dim ) 
                    var1   = rebin ( var1, dimens ) 
                ENDIF 
            ENDIF ELSE BEGIN 
                var1 = var0 [ 0 : n_def - 1  ] 
                IF ( n_var0 GT 1 ) THEN BEGIN 
                    dimens = size ( default, /dim ) 
                    var1   = rebin ( var1, dimens ) 
                ENDIF 
            ENDELSE 
        ENDIF 
    ENDIF ELSE BEGIN 
        if ( noerrormessage eq 0 ) then PRINT, '    IFEXEQ :: error: variable not defined ' 
        dimen = size ( [default], /dim ) 
        var1 = reform ( dblarr ( dimen ) ) 
        var1[0: * ] = default 
        if ( n_elements ( var1 ) eq 1 ) then BEGIN 
            var1 = var1[0] 
        ENDIF ELSE BEGIN 
;        dimens = size ( var1, /dim ) 
;        var1   = rebin ( var1, dimens ) 
        ENDELSE
 
    ENDELSE 
 
    IF ( shortinteger GT 0 ) THEN BEGIN 
        var1 = long ( var1 ) 
    ENDIF 
    IF ( long GT 0 ) THEN BEGIN 
        var1 = long ( var1 ) 
    ENDIF 


 
    Ende: 
    RETURN, var1
 
END
