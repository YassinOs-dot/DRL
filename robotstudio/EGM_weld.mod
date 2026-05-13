MODULE EGM_weld
  ! ABB EGM module for continuous sensor correction mode.
  ! The external PC sends a small correction frame every 4 ms.
  ! The robot executes EGMActPose with the tool 'torch'.
  
  CONST egmident egmID:=[6510, "EGM"];  ! port 6510, identifier must match
  VAR egmstate egmSt;
  PERS tooldata torch:=[TRUE, [[0,0,200], [1,0,0,0]], [0.1, [0,0,0], [1,0,0,0], 0,0,0]];
  TASK PERS wobjdata wobj0:=[FALSE,TRUE,"",[[0,0,0],[1,0,0,0]],[[0,0,0],[1,0,0,0]]];
  
  PROC main()
    ! Activate EGM in sensor correction mode
    EGMActPose egmID\Tool:=torch\WObj:=wobj0,
               e1:=corr_frame\SensorPose\Frame, egmSt;
    WHILE TRUE DO
      WaitEGM 4;
    ENDWHILE
  ERROR
    IF ERRNO = ERR_EGM_STOPPED THEN
      RETRY;
    ELSE
      ! Unexpected error: stop robot
      Stop;
    ENDIF
  ENDPROC
ENDMODULE
