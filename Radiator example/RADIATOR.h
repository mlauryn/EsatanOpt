      USE gar_GlobalArrays_m, DYCORE => gar_dynamicCoreReal,
     &   GR => gar_radiativeConds, GRFLG => gar_radiativeCondActs,
     &   INCORE => gar_dynamicCoreInt, LIST70 => gar_list70,
     &   NPCS => gar_pcsThermal, PCS => gar_pcsThermalOld
      IMPLICIT DOUBLE PRECISION (A-H, O-Z)
      COMMON /AINF/   AI     (          33 )
      COMMON /ALINAM/ ALNAM  (           1 )
      COMMON /ALIVAL/ ALVAL  (           1 )
      COMMON /APS/    ALP    (           8 )
      COMMON /AREA/   A      (           8 )
      COMMON /ARPT/   AP     (           3 )
      COMMON /ARRC/   CA     (           1 )
      COMMON /ARRI/   IA     (           1 )
      COMMON /ARRR/   RA     (          36 )
      COMMON /CAP/    C      (           8 )
      COMMON /CLST69/ LIST69 (          28 )
      COMMON /CLST71/ LIST71 (           1 )
      COMMON /CMFLH/  MFLH   (           1 )
      COMMON /CMFLX/  MFLX   (           1 )
      COMMON /CMODND/ MODNOD (           1 )
      COMMON /CMTYPE/ MTYPE  (           1 )
      COMMON /CPHSON/ PHSON  (           2 ,           1 )
      COMMON /CPHSX/  PHSX   (           2 ,           1 )
      COMMON /CQEVL/  QEVL   (           1 )
      COMMON /CSP/    PCSP   (           8 )
      COMMON /EMM/    EPS    (           8 )
      COMMON /FAREA/  FA     (           1 )
      COMMON /FCMP/   CMP    (           1 )
      COMMON /FCSEQ/  FPCS   (          22 )
      COMMON /FCSP/   FPCSP  (           1 )
      COMMON /FDIA/   FD     (           1 )
      COMMON /FDPSDV/ DPSDV  (           1 )
      COMMON /FENTH/  FE     (           1 )
      COMMON /FFRIC/  FF     (           1 )
      COMMON /FFST/   FST    (           1 )
      COMMON /FGM/    GM     (           1 )
      COMMON /FGP/    GP     (           1 )
      COMMON /FHSOR/  FH     (           1 )
      COMMON /FLABEL/ L      (           1 )
      COMMON /FLAF/   FLA    (           1 )
      COMMON /FLAGS/  FLG    (          41 )
      COMMON /FLEN/   FL     (           1 )
      COMMON /FLRG/   FRG    (           1 )
      COMMON /FLST/   FLFLG  (           1 )
      COMMON /FLSTT/  FNST   (           1 )
      COMMON /FMFLO/  M      (           1 )
      COMMON /FMSOR/  FM     (           1 )
      COMMON /FNPAIR/ FNP    (           1 )
      COMMON /FNUMB/  FNNUM  (           1 )
      COMMON /FPHI/   PHI    (           1 )
      COMMON /FPRESS/ P      (           1 )
      COMMON /FPSRC/  PSRC   (           1 )
      COMMON /FQSOR/  FQ     (           1 )
      COMMON /FQUAL/  QUAL   (           1 )
      COMMON /FRSOR/  FR     (           1 )
      COMMON /FSPHUM/ SPHUM  (           1 )
      COMMON /FSPRNC/ FCUD   (           1 )
      COMMON /FSPRNI/ FIUD   (           1 )
      COMMON /FSPRNR/ FRUD   (           4 )
      COMMON /FTEMP/  TF     (           1 )
      COMMON /FTYPE/  FT     (           1 )
      COMMON /FVCF/   VCF    (           1 )
      COMMON /FVDT/   VDT    (           1 )
      COMMON /FVOL/   VOL    (           1 )
      COMMON /FWSOR/  FW     (           1 )
      COMMON /FWVSNK/ WVSNK  (           1 )
      COMMON /FXN/    FX     (           1 )
      COMMON /FXNT/   FXT    (           8 )
      COMMON /FYN/    FY     (           1 )
      COMMON /FYNT/   FYT    (           8 )
      COMMON /FZN/    FZ     (           1 )
      COMMON /FZNT/   FZT    (           8 )
      COMMON /GFLO/   GF     (           1 )
      COMMON /GFST/   GFFLG  (           1 )
      COMMON /GLIN/   GL     (          14 )
      COMMON /GLOC/   OPBLOK, FORMAT, LINTYP, MODULE,  TFORM,
     $         THEAD, PARNAM, HEADER, TFORMF, SOLTYP, RESULT,
     $          MFFB, TIMELA, OUTIME, SOLVER, QTRSOL
      COMMON /GLOI/   IG     (          25 )
      COMMON /GLOR/   RG     (          49 )
      COMMON /GLST/   GLFLG  (          14 )
      COMMON /GVEW/   GV     (           1 )
      COMMON /GVST/   GVFLG  (           1 )
      COMMON /LABEL/  NLAB   (           8 )
      COMMON /LOCC/   CL     (           1 )
      COMMON /LOCI/   IL     (           2 )
      COMMON /LOCR/   RL     (           1 )
      COMMON /MDIR/   MD     (          18 ,           2 )
      COMMON /NAMA/   AN     (           3 )
      COMMON /NAMC/   CCN    (          92 )
      COMMON /NAMM/   SN     (           1 )
      COMMON /NAMU/   VCN    (           3 )
      COMMON /NCSP/   NPCSP  (           6 ,           8 )
      COMMON /NUMBR/  NNUM   (           8 )
      COMMON /QAIS/   QAI    (           8 )
      COMMON /QALB/   QA     (           8 )
      COMMON /QEAR/   QE     (           8 )
      COMMON /QEIS/   QEI    (           8 )
      COMMON /QINT/   QI     (           8 )
      COMMON /QRES/   QR     (           8 )
      COMMON /QSIS/   QSI    (           8 )
      COMMON /QSOL/   QS     (           8 )
      COMMON /SPRNC/  CUD    (           8 )
      COMMON /SPRNI/  IUD    (           8 )
      COMMON /SPRNOD/ SPRNDE (           4 ,           3 )
      COMMON /SPRNR/  RUD    (          32 )
      COMMON /SPRSIZ/ SPRNDM, SPRNDN, USRNDC, USRNDI, USRNDR
      COMMON /STAT2/  NST2   (           8 )
      COMMON /STATM/  MST    (           1 )
      COMMON /STATUS/ NST    (           8 )
      COMMON /TEMP/   T      (           8 )
      COMMON /UCNC/   CU     (           1 )
      COMMON /UCNI/   IU     (           1 )
      COMMON /UCNR/   RU     (           2 )
      COMMON /UCPT/   PU     (           1 )
      COMMON /VSEQ/   VLNK   (           1 )
      COMMON /VSLIP/  VSL    (           1 )
      CHARACTER ACGET *132, ALNAM *18, AN *18
      CHARACTER CA *133, CCN *18, CL *1, CU *133, CUD *24
      CHARACTER FCUD *24, FLFLG *1, FNST *1, FORMAT *8, FRG *4
      CHARACTER FST *4, FT *24, FTYPEC *24
      CHARACTER GETFRG *4, GETFST *4, GETFT *24, GETL *24
      CHARACTER GFFLG *1, GLFLG *1, GVFLG *1
      CHARACTER HEADER *132
      CHARACTER L *24, LINTYP *6
      CHARACTER MFFB *3, MODULE *6, MST *1
      CHARACTER NLAB *24, NST *1,  NST2 *1
      CHARACTER OPBLOK *10, OUTIME *18
      CHARACTER PARNAM *24
      CHARACTER QTRSOL *3
      CHARACTER RESULT *80
      CHARACTER SN *24, SOLTYP *7, SOLVER *6, SPRNDE *18, STATRP *3
      CHARACTER SUBMDN  *256, SUBMOD *256
      CHARACTER TFORM *132, TFORMF *132, THEAD *132, TIMELA *3
      CHARACTER VCN *18, VCF *1
      CHARACTER ZDAYDT *17, ZDAYTM *8
      INTEGER ADIM, ADIMVL, AI, AIGET, ALVAL, AP, ASIZE, AUNDF
      INTEGER FIUD, FLG, FNNUM, FNP, FPCS, FPCSP, FTYPEI
      INTEGER IA, IG, IL, INTNOD, IQQMAX, IU, IUD
      INTEGER LIST69, LIST71
      INTEGER MATDTI, MATSMI, MD, MODNOD, MTYPE
      INTEGER NNUM, NODNUM, NPCSP
      INTEGER PCSP, PU
      INTEGER SPRNDM, SPRNDN, STRLNA
      INTEGER USRNDC, USRNDI,USRNDR
      INTEGER VLNK
      INTEGER WORST
      DOUBLE PRECISION A, ACLOSS, ALP
      DOUBLE PRECISION C, CMP
      DOUBLE PRECISION DPSDV
      DOUBLE PRECISION EPS
      DOUBLE PRECISION FA, FD, FE, FF, FH, FL, FLA, FM, FQ, FR, FRUD
      DOUBLE PRECISION FW, FX, FXT, FY, FYT, FZ, FZT
      DOUBLE PRECISION FLUXF, FLUXGF, FLUXGL, FLUXGR FLUXGT, FLUXL
      DOUBLE PRECISION FLUXMF, FLUXML, FLUXMR, FLUXR, FLUXT
      DOUBLE PRECISION GF, GL, GM, GP, GRPAVE, GRPMAX, GRPMIN
      DOUBLE PRECISION GRPSUM,GV
      DOUBLE PRECISION HCAP, HTCOEF
      DOUBLE PRECISION INTCY1, INTCY2, INTCY3, INTCYC, INTEGL
      DOUBLE PRECISION INTERP, INTGL1, INTGL2, INTRP1, INTRP2, INTRP3
      DOUBLE PRECISION INTRPA
      DOUBLE PRECISION LSTRP1, LSTRP2
      DOUBLE PRECISION M, MATDTR, MATSMR, MFLH, MFLX
      DOUBLE PRECISION NDMFL, NODFN1, NODFN2, NUVRE
      DOUBLE PRECISION P, PHI, PHSX, POLYNM, PSRC
      DOUBLE PRECISION QA, QAI, QE, QEI, QEVL, QI, QR, QS, QSI, QUAL
      DOUBLE PRECISION RA, RG, RL, RU, RUD
      DOUBLE PRECISION SPHUM, SRLMAX, STVRE
      DOUBLE PRECISION T, TF
      DOUBLE PRECISION VDT, VOL, VSL
      DOUBLE PRECISION WVSNK
      LOGICAL AFTER, AT
      LOGICAL BEFORE, BTWEEN
      LOGICAL PHSON
      LOGICAL SUNDCK
      EXTERNAL GETA
      EXTERNAL GETALP
      EXTERNAL GETC
      EXTERNAL GETCMP
      EXTERNAL GETCCR
      EXTERNAL GETEPS
      EXTERNAL GETFD
      EXTERNAL GETFE
      EXTERNAL GETFF
      EXTERNAL GETFH
      EXTERNAL GETFL
      EXTERNAL GETFLA
      EXTERNAL GETFM
      EXTERNAL GETFQ
      EXTERNAL GETFR
      EXTERNAL GETFRG
      EXTERNAL GETFST
      EXTERNAL GETFT
      EXTERNAL GETFW
      EXTERNAL GETFX
      EXTERNAL GETFY
      EXTERNAL GETFZ
      EXTERNAL GETGL
      EXTERNAL GETGL2
      EXTERNAL GETGF
      EXTERNAL GETGF2
      EXTERNAL GETGP
      EXTERNAL GETGR
      EXTERNAL GETGR2
      EXTERNAL GETGV
      EXTERNAL GETGV2
      EXTERNAL GETL
      EXTERNAL GETM
      EXTERNAL GETP
      EXTERNAL GETPHI
      EXTERNAL GETQA
      EXTERNAL GETQAI
      EXTERNAL GETQE
      EXTERNAL GETQEI
      EXTERNAL GETQI
      EXTERNAL GETQR
      EXTERNAL GETQS
      EXTERNAL GETQSI
      EXTERNAL GETT
      EXTERNAL GETVDT
      EXTERNAL GETVOL
      EXTERNAL GETVQ
      EXTERNAL SETA
      EXTERNAL SETALP
      EXTERNAL SETC
      EXTERNAL SETCMP
      EXTERNAL SETEPS
      EXTERNAL SETFD
      EXTERNAL SETFE
      EXTERNAL SETFF
      EXTERNAL SETFLA
      EXTERNAL SETFH
      EXTERNAL SETFL
      EXTERNAL SETFM
      EXTERNAL SETFQ
      EXTERNAL SETFR
      EXTERNAL SETFRG
      EXTERNAL SETFST
      EXTERNAL SETFT
      EXTERNAL SETFW
      EXTERNAL SETFX
      EXTERNAL SETFY
      EXTERNAL SETFZ
      EXTERNAL SETGL
      EXTERNAL SETGL2
      EXTERNAL SETGF
      EXTERNAL SETGF2
      EXTERNAL SETGP
      EXTERNAL SETGR
      EXTERNAL SETGR2
      EXTERNAL SETGV
      EXTERNAL SETGV2
      EXTERNAL SETL
      EXTERNAL SETM
      EXTERNAL SETP
      EXTERNAL SETPHI
      EXTERNAL SETQA
      EXTERNAL SETQAI
      EXTERNAL SETQE
      EXTERNAL SETQEI
      EXTERNAL SETQI
      EXTERNAL SETQR
      EXTERNAL SETQS
      EXTERNAL SETQSI
      EXTERNAL SETT
      EXTERNAL SETVDT
      EXTERNAL SETVOL
      EXTERNAL SETVQ
      PARAMETER (IQQMAX = 999999)
