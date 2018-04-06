      PROGRAM RADIATOR                                                          
C   
      INCLUDE 'RADIATOR.h'                                                      
C   
      CHARACTER MNAME * 24                                                      
C   
      MNAME = 'RADIATOR                '                                        
C   
      FLG(1) = 8                                                                
      FLG(2) = 14                                                               
      FLG(3) = 0                                                                
      FLG(4) = 1                                                                
      FLG(5) = 0                                                                
      FLG(6) = 2                                                                
      FLG(7) = 1                                                                
      FLG(8) = 0                                                                
      FLG(9) = 0                                                                
      FLG(10) = 0                                                               
      FLG(11) = 2                                                               
      FLG(12) = 0                                                               
      FLG(13) = 49                                                              
      FLG(14) = 25                                                              
      FLG(15) = 16                                                              
      FLG(16) = 3                                                               
      FLG(17) = 92                                                              
      FLG(18) = 33                                                              
      FLG(19) = 3                                                               
      FLG(20) = 36                                                              
      FLG(21) = 0                                                               
      FLG(22) = 0                                                               
      FLG(23) = 0                                                               
      FLG(24) = 130                                                             
      FLG(25) = 2                                                               
      FLG(26) = 1                                                               
      FLG(27) = 133                                                             
      FLG(28) = 1000127                                                         
      FLG(29) = 1                                                               
      FLG(30) = 0                                                               
      FLG(31) = 0                                                               
      FLG(32) = 0                                                               
      FLG(33) = 0                                                               
      FLG(34) = 0                                                               
      FLG(35) = 22                                                              
      FLG(36) = 1                                                               
      FLG(37) = 0                                                               
      FLG(38) = 0                                                               
      FLG(39) = 28                                                              
      FLG(40) = 0                                                               
      FLG(41) = 0                                                               
      CALL SVMNAM(MNAME)                                                        
C   
      SPRNDM = 4                                                                
      SPRNDN = 3                                                                
      USRNDC = 0                                                                
      USRNDI = 0                                                                
      USRNDR = 4                                                                
      SPRNDE(1,1) = 'T_MIN             '                                        
      SPRNDE(2,1) = 'TIM_MIN           '                                        
      SPRNDE(3,1) = 'T_MAX             '                                        
      SPRNDE(4,1) = 'TIM_MAX           '                                        
C   
      CALL MAINA(MNAME)                                                         
C   
      CALL SUCCES                                                               
C   
      STOP                                                                      
C   
      END                                                                       
      SUBROUTINE IN0001                                                 
      INCLUDE 'RADIATOR.h'
      LOGICAL HTFLAG                                                    
      HTFLAG = .TRUE.                                                   
      OPBLOK = 'INITIAL   '                                             
      IG(2) = 1                                                         
      CALL SETNDR(' ','T_MIN',1.0D10,1)                                 
      CALL SETNDR(' ','T_MAX',-1.0D10,1)                                
      CALL FINITS(' ' , -    1)                                         
      OPBLOK = ' '                                                      
      RETURN                                                            
      END                                                               
      SUBROUTINE V10001                                                 
      INCLUDE 'RADIATOR.h'
      LOGICAL HTFLAG                                                    
      HTFLAG = (SOLTYP .EQ. 'THERMAL' .OR. SOLTYP .EQ. 'HUMID')         
      OPBLOK = 'VARIABLES1'                                             
      CALL PRMUPD                                                       
      CALL CHKTRM                                                       
      CALL ACDDYU                                                       
      IG(2) = 1                                                         
      QA(MD(3,1)+7)=INTCYC(RG(16),MD(14,1)+2,MD(14,1)+1,1,RU(MD(8,1)+1),
     $0.0D0)*0.2000000000000000D+00*RU(MD(8,1)+2)                       
      QE(MD(3,1)+7)=INTCYC(RG(16),MD(14,1)+2,MD(14,1)+3,1,RU(MD(8,1)+1),
     $0.0D0)*0.2000000000000000D+00*RU(MD(8,1)+2)                       
      CALL GM0001(HTFLAG)                                               
      OPBLOK = ' '                                                      
      RETURN                                                            
      END                                                               
      SUBROUTINE GM0001_00001(HTFLAG)                                   
      INCLUDE 'RADIATOR.h'                                              
      LOGICAL HTFLAG                                                    
      C(MD(3,1)+7)=0.2000000000000000D+00*RU(MD(8,1)+2)*0.30000000000000
     $00D-02*0.2800000000000000D+04*0.9200000000000000D+03              
      A(MD(3,1)+7)=0.2000000000000000D+00*RU(MD(8,1)+2)                 
      GR(MD(6,1)+1)=0.2000000000000000D+00*RU(MD(8,1)+2)*0.9            
      RETURN                                                            
      END                                                               
      SUBROUTINE GM0001(HTFLAG)                                         
      LOGICAL HTFLAG                                                    
      CALL GM0001_00001(HTFLAG)                                         
      RETURN                                                            
      END                                                               
      SUBROUTINE V20001                                                 
      INCLUDE 'RADIATOR.h'
      OPBLOK = 'VARIABLES2'                                             
      CALL SSNCNT(FLG(24),FLG(25),MAX0(FLG(1),1),PCS,T)                 
      IG(25) = IG(25) + 1                                               
      CALL PRMUPD                                                       
      IG(2) = 1                                                         
      IF(IU(MD(9,1)+1).EQ.1)THEN                                        
      CALL STORMM('T','T_MIN','TIM_MIN','T_MAX','TIM_MAX')              
      ENDIF                                                             
      OPBLOK = ' '                                                      
      CALL PARWRT('VARIABLES2')                                         
      RETURN                                                            
      END                                                               
      SUBROUTINE EXECTN                                                 
      INCLUDE 'RADIATOR.h'
      HEADER='Radiator Sizing Model'                                    
      IG(4)=100                                                         
      RG(13)=0.01                                                       
      RG(18)=RU(MD(8,1)+1)                                              
      RG(12)=RG(18)/10.0                                                
      RG(3)=20.0                                                        
      IU(MD(9,1)+1)=0                                                   
      CALL SOLCYC('SLFWBK',0.1D0,0.1D0,RU(MD(8,1)+1),99,' ','NONE')     
      IU(MD(9,1)+1)=1                                                   
      CALL SLFWBK                                                       
      RETURN                                                            
      END                                                               
      SUBROUTINE OUTPUT                                                 
      INCLUDE 'RADIATOR.h'
      IF (OUTIME .NE. 'ALL') RETURN                                     
      OPBLOK = 'OUTPUTS'                                                
      IF(RG(17).EQ.RG(18))THEN                                          
      CALL PRNDTB(' ','L, T_MIN, TIM_MIN, T_MAX, TIM_MAX',1)            
      ENDIF                                                             
      OPBLOK = ' '                                                      
      CALL PARWRT('OUTPUTS')                                            
      RETURN                                                            
      END                                                               
