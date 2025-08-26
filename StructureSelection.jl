include("DDAfunctions.jl");
include("MinError.jl");

ROS=[[0  0 2];                                                        # Roessler system
     [0  0 3];
     [1  0 1];
     [1  0 2];
     [2  0 0];
     [2  0 3];
     [2  1 3]
     ];
NrSyst=10;                                                            # 10 Roessler systems
(MOD_nr,DIM,ODEorder,P) = make_MOD_nr(ROS,NrSyst);

example = "_1";

a = 0.15; 
b = 0.2;
c = 10;

MOD_par=repeat([ -1 -1 1 a b -c  1 ],NrSyst,1);
MOD_par[:,5] .+= randn(NrSyst,1)./100;                                # slightly different b parameters
MOD_par=reshape(MOD_par',size(ROS,1)*NrSyst)';

LL=10000;                                                             # integration length

TRANS=20000;                                                          # transient
dt=0.05;                                                              # integration step size
X0=rand(DIM*NrSyst,1);                                                # initial conditions

DATA_DIR="DATA"; dir_exist(DATA_DIR);                                 # DATA folder
FN_data=@sprintf("%s%sdata_%s.ascii",DATA_DIR,SL,example);            # data file
CH_list=1:DIM*NrSyst;                                                 # x,y,z
DELTA=2;                                                              # every second data point
if !isfile(FN_data)
   integrate_ODE_general_BIG(MOD_nr,MOD_par,                          # encoding of the systems
                             dt,                                      # step size of num. integration
                             LL,                                      # length 
                             DIM*NrSyst,ODEorder,X0,                  # parameters
                             FN_data,
                             CH_list,DELTA,
                             TRANS);
end

DDA_DIR=@sprintf("DDA_%s",example); dir_exist(DDA_DIR);               # DDA folder

dm=4;
TM=50; DELAYS=collect(dm+1:TM);

#WL=2000;WS=500;
WL=[]; WS=[];

DDAorder=2;

CH_list=[];
MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"yes","ALL",[],"ASCII");
                                                                      # compute DDA outputs for all timeseries


CH_list = [collect(1:3:DIM*NrSyst)];                                  # only x; common best model
(TAU_select,mm_select) = MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"","ALL",[],"ASCII");

CH_list = [collect(1:3:DIM*NrSyst)];                                  # only y; common best model
(TAU_select,mm_select) = MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"","ALL",[],"ASCII");

CH_list=map(i -> [i],1:DIM*NrSyst)                                    # each of the 30 timeseries
(TAU_select,mm_select) = MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"","ALL",[],"ASCII");

CH_list = [[[4; 7; 10]]; [[19]]; [[25;22]]];                          # certain timeseries number combinations
(TAU_select,mm_select) = MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"","ALL",[],"ASCII");


######


DDA_DIR=@sprintf("DDA_%s_part",example); dir_exist(DDA_DIR);           # DDA folder


                                                                       # compute DDA outputs for all timeseries
CH_list = [[[4; 7; 10]]; [[19]]; [[25;22]]];                           # certain timeseries number combinations
(TAU_select,mm_select) = MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,"yes","ALL",[1 2000],"ASCII");
