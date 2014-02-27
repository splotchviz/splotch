FUNCTION darray, xmin0, xmax0, n0, float=float0, position=posi, loga=log0
 
    ;; create a double (float) array of size n ranging from xmin to xmax
 
    n = IFEXEQ ( n0, def=100, /int ) 
 
    xmin = IFEXEQ ( xmin0, def=0.d0 ) 
    xmax = IFEXEQ ( xmax0, def=1.d0 ) 
 
    flt = IFEXEQ ( float0, def=0, /int ) 
    ;; values at central, left, right position
    pos = IFEXEQ ( posi, def=0, /int ) 
 
    loga = IFEXEQ ( log0, def=0, /int ) 
     
    IF ( loga ) THEN BEGIN 
        xmin = alog10 ( xmin ) 
        xmax = alog10 ( xmax ) 
    ENDIF

    IF ( pos EQ 0 ) THEN BEGIN 
        ;; central
        offset = 0.5d0 
        m      = n
    ENDIF ELSE $ 
      IF ( pos EQ 1 ) THEN BEGIN 
        ;; left
        offset = 0.d0 
        m      = n
    ENDIF ELSE $ 
      IF ( pos EQ 2 ) THEN BEGIN 
        ;; right
        offset = 1.d0 
        m      = n
    ENDIF ELSE $ 
      IF ( pos EQ 3 ) THEN BEGIN 
        ;; right
        offset = 0.d0 
        m      = n - 1
    ENDIF
      

    array = findgen ( n ) 
 
    dx = ( xmax - xmin ) / double ( m )
  
    array = xmin + ( array + offset ) * dx

    IF ( loga ) THEN BEGIN 
        array = 10.d0 ^ array
    ENDIF

    IF ( flt ) THEN BEGIN 
 
        array = float ( array )

    ENDIF

return, array

Ende:
END
