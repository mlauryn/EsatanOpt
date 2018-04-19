$MODEL RADIATOR, GLOBALFILE = radiator.gbl
#
# Model of an electronics unit with a variable-size radiator.
#
# Copyright (c) 2006 ITP Engines UK Ltd.
#
# ******************************************************************************
$LOCALS
# ******************************************************************************
#
   $REAL
   k_Al    = 180.0;    # Thermal conductivity of aluminium, W/m/K
   cp_Al   = 920.0;    # Specific thermal heat capacity of aluminium, J/kg/K
   rho_Al  = 2800.0;   # Density of aluminium, kg/m3
   t_box   = 0.003;    # Thickness of box sides, m
   l_box   = 0.150;    # Length of box, m
   w_box   = 0.200;    # Width of box, m
   h_box   = 0.150;    # Height of box, m
   temp    = 20.0;     # Initial temperature, deg C
#
# ******************************************************************************
$NODES
# ******************************************************************************
#
D10 = 'Internal Electronics', T = temp, C = 500.0, QI = 12.0;
D20 = 'Unit top', T = temp, C = w_box * l_box * t_box * rho_Al * cp_Al;
D30 = 'Unit side 1', T = temp, C = h_box * l_box * t_box * rho_Al * cp_Al;
D40 = 'Unit side 2', T = temp, C = h_box * w_box * t_box * rho_Al * cp_Al;
D50 = 'Unit side 3', T = temp, C = h_box * l_box * t_box * rho_Al * cp_Al;
D60 = 'Unit side 4', T = temp, C = h_box * w_box * t_box * rho_Al * cp_Al;
D70 = 'Radiator', T = temp, C = w_box * RadLen * t_box * rho_Al * cp_Al,
                  EPS = 0.9, A = w_box * RadLen;
B999= 'Deep Space', T = -270.0;
#
#
# ******************************************************************************
$CONDUCTORS
# ******************************************************************************
#
GR(70,999) = w_box * RadLen * 0.9; 
#
GL(10, 30) = 1.50;
GL(10, 50) = 1.50;
GL(30, 40) = h_box * t_box / ((w_box + l_box) / 2.0) * k_Al;
GL(40, 50) = h_box * t_box / ((w_box + l_box) / 2.0) * k_Al;
GL(50, 60) = h_box * t_box / ((w_box + l_box) / 2.0) * k_Al;
GL(60, 30) = h_box * t_box / ((w_box + l_box) / 2.0) * k_Al;
GL(20, 30) = l_box * t_box / ((w_box + h_box) / 2.0) * k_Al;
GL(20, 40) = w_box * t_box / ((l_box + h_box) / 2.0) * k_Al;
GL(20, 50) = l_box * t_box / ((w_box + h_box) / 2.0) * k_Al;
GL(20, 60) = w_box * t_box / ((l_box + h_box) / 2.0) * k_Al;
GL(70, 30) = l_box * t_box / ((w_box + h_box) / 2.0) * k_Al;
GL(70, 40) = w_box * t_box / ((l_box + h_box) / 2.0) * k_Al;
GL(70, 50) = l_box * t_box / ((w_box + h_box) / 2.0) * k_Al;
GL(70, 60) = w_box * t_box / ((l_box + h_box) / 2.0) * k_Al;
#
# ******************************************************************************
$CONSTANTS
# ******************************************************************************
#
   $REAL
   RadLen  = 0.4;          # Radiator length
   Period  = 6047.80;      # Orbital period
#
   $INTEGER
   IRESULT = 0;            # Flag to control recording of results
#
   $CONTROL
   WIDTH = 90;           # Width of output file 
#
# ******************************************************************************
$ARRAYS
# ******************************************************************************
#
   $REAL
   ORBTIM(12) =  # orbit times
             0.0,  377.98, 2645.83, 3023.80, 3401.78, 3779.76,
         4157.73, 4535.71, 4913.68, 5291.66, 5669.63, 6047.80;
#
   ALBFLUX(12) = # Absorbed albedo flux density
          4.47,   0.0,   0.0,  3.76, 36.31, 68.04, 
         89.42, 97.18, 90.15, 69.36, 38.07,  4.71;
#
   PLFLUX(12) = # Absorbed planet flux density
         94.61, 94.61, 94.61, 94.61, 94.61, 94.61, 
         94.61, 94.61, 94.61, 94.61, 94.61, 94.61;
#
# ******************************************************************************
$INITIAL
# ******************************************************************************
      CALL SETNDR(' ', 'T_MIN', 1.0D10, CURRENT)
      CALL SETNDR(' ', 'T_MAX', -1.0D10, CURRENT)
#
# ******************************************************************************
$VARIABLES1
# ******************************************************************************
#
# Set radiator fluxes
#
       QA70 = INTCYC (TIMEM, ORBTIM, ALBFLUX, 1, Period, 0.0D0) * w_box * RadLen
       QE70 = INTCYC (TIMEM, ORBTIM, PLFLUX, 1, Period, 0.0D0) * w_box * RadLen
#       
# ******************************************************************************
$VARIABLES2
# ******************************************************************************
#
      IF (IRESULT .EQ. 1) THEN
         CALL STORMM('T', 'T_MIN', 'TIM_MIN', 'T_MAX', 'TIM_MAX')
      END IF
#
# ******************************************************************************
$EXECUTION
# ******************************************************************************
#
      HEADER = 'Radiator Sizing Model'
#
      NLOOP = 100
      RELXCA = 0.01
      TIMEND = Period
      OUTINT = TIMEND / 10.0
      DTIMEI = 20.0
#
      IRESULT = 0
      CALL SOLCYC('SLFWBK', 0.1D0, 0.1D0, Period, 99, ' ', 'NONE')
#
      IRESULT = 1
      CALL SLFWBK
#
# ******************************************************************************
$OUTPUTS
# ******************************************************************************
#
      IF (TIMEN .EQ. TIMEND) THEN
         CALL PRNDTB(' ', 'L, T_MIN, TIM_MIN, T_MAX, TIM_MAX', CURRENT)   
      END IF
#
$ENDMODEL RADIATOR

