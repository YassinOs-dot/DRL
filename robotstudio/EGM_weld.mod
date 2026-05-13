MODULE EGM_weld
  VAR egmident egmID;
  VAR egmstate egmSt;
  PERS tooldata torch:=[TRUE,[[0,0,200],[1,0,0,0]],[0.1,...]];
  
  PROC main()
    EGMGetId egmID;
    ! Start EGM in sensor correction mode
    EGMActPose egmID\Tool:=torch\WObj:=wobj0,
               e1:=corr_frame\SensorPose\Frame, egmSt;
    WHILE TRUE DO
      ! Wait for new correction from external client
      WaitEGM 4; 
    ENDWHILE
  ENDPROC
ENDMODULE
