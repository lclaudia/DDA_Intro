include("DDAfunctions.jl");

NrSyst=4;    
ROS=[[0  0 2];  
     [0  0 3];
     [1  0 1];
     [1  0 2];
     [2  0 0];
     [2  0 3];
     [2  1 3]
    ];
(MOD_nr,DIM,ODEorder,P) = make_MOD_nr(ROS,NrSyst);


a123= 0.21; 
a456= 0.20;

b1  = 0.2150;
b2  = 0.2020; 
b4  = 0.4050; 
b5  = 0.3991; 

c   = 5.7;

MOD_par=[
         -1 -1 1 a123  b1 -c  1 
         -1 -1 1 a123  b2 -c  1
         -1 -1 1 a456  b4 -c  1
         -1 -1 1 a456  b5 -c  1
        ];
MOD_par=reshape(MOD_par',size(ROS,1)*NrSyst)';

TRANS=20000;          
dt=0.05;             
X0=rand(DIM*NrSyst,1);
CH_list = 1:DIM:DIM*NrSyst;
DELTA=2;
FN_DATA = "ROS_4.ascii";
L=20000;
if !isfile(FN_DATA)
   integrate_ODE_general_BIG(MOD_nr,MOD_par,dt,L,DIM*NrSyst,ODEorder,X0,FN_DATA,CH_list,DELTA,TRANS);
end

Y=readdlm(FN_DATA);
WL=2000;WS=1000;TAU=[32 9];dm=4;order=3;nr_delays=2; TM=maximum(TAU);
WN = Int(1+floor((size(Y,1)-(WL+TM+2*dm-1))/WS));

########  SingleTimeseries DDA  ########


Y=readdlm(FN_DATA); Y=Y[:,1];

ST = fill(NaN,WN,4);
for wn=0:WN-1
    anf=wn*WS; ende=anf+WL+TM+2*dm-1;
    
    data=Y[anf+1:ende+1]; ddata=deriv_all(data,dm); data=data[dm+1:end-dm];
    
    STD=std(data); 
    DATA = (data .- mean(data)) ./ STD; dDATA = ddata / STD;
    dDATA=dDATA[TM+1:end];
    M=hcat(DATA[(TM+1:end) .- TAU[1]] , 
           DATA[(TM+1:end) .- TAU[2]] ,
           DATA[(TM+1:end) .- TAU[1]].^3 );
           
    ST[wn+1,1:3] = (M \ dDATA); 
    ST[wn+1,4]   = sqrt(mean((dDATA .- M*ST[wn+1,1:3]).^2));
end


###  for all time series

Y=readdlm(FN_DATA);

ST = fill(NaN,WN,4,size(Y,2));
for n_Y=1:size(Y,2)
    for wn=0:WN-1
        anf=wn*WS; ende=anf+WL+TM+2*dm-1;
        
        data=Y[anf+1:ende+1,n_Y]; ddata=deriv_all(data,dm); data=data[dm+1:end-dm];
        
        STD=std(data); 
        DATA = (data .- mean(data)) ./ STD; dDATA = ddata / STD;
        
        M=hcat(DATA[(TM+1:end) .- TAU[1]] , 
               DATA[(TM+1:end) .- TAU[2]] ,
               DATA[(TM+1:end) .- TAU[1]].^3 );
        dDATA=dDATA[TM+1:end];
               
        ST[wn+1,1:3,n_Y] = (M \ dDATA); 
        ST[wn+1,4,n_Y]   = sqrt(mean((dDATA .- M*ST[wn+1,1:3,n_Y]).^2));
    end
end




########  CrossTimeseries DDA  ########

NrCH=size(Y,2); CH=collect(1:NrCH);
LIST=collect(combinations(CH,2));
LL1=vcat(LIST...)';
LIST=reduce(hcat,LIST)';   

CT = fill(NaN,WN,4,size(LIST,1));
for n_LIST=1:size(LIST,1)
    ch1=LIST[n_LIST,1]; ch2=LIST[n_LIST,2];
    for wn=0:WN-1
        anf=wn*WS; ende=anf+WL+TM+2*dm-1;
        
        data1=Y[anf+1:ende+1,ch1]; ddata1=deriv_all(data1,dm); data1=data1[dm+1:end-dm];
        STD=std(data1); DATA1 = (data1 .- mean(data1)) ./ STD; dDATA1 = ddata1 / STD;
        dDATA1=dDATA1[TM+1:end];
        M1=hcat(DATA1[(TM+1:end) .- TAU[1]] , 
                DATA1[(TM+1:end) .- TAU[2]] ,
                DATA1[(TM+1:end) .- TAU[1]].^3 );

        data2=Y[anf+1:ende+1,ch2]; ddata2=deriv_all(data2,dm); data2=data2[dm+1:end-dm]; 
        STD=std(data2); DATA2 = (data2 .- mean(data2)) ./ STD; dDATA2 = ddata2 / STD;
        dDATA2=dDATA2[TM+1:end];
        M2=hcat(DATA2[(TM+1:end) .- TAU[1]] , 
                DATA2[(TM+1:end) .- TAU[2]] ,
                DATA2[(TM+1:end) .- TAU[1]].^3 );

        M=vcat(M1,M2); dDATA=vcat(dDATA1,dDATA2);             

        CT[wn+1,1:3,n_LIST] = (M \ dDATA); 
        CT[wn+1,4,n_LIST]   = sqrt(mean((dDATA .- M*CT[wn+1,1:3,n_LIST]).^2));
    end
end


########  DynamicalErgodicity DDA  ########

st=dropdims(mean(ST[:,end,:],dims=1),dims=1);
ct=dropdims(mean(CT[:,end,:],dims=1),dims=1);

E = fill(NaN,size(Y,2),size(Y,2));
for n_LIST=1:size(LIST,1)
    ch1=LIST[n_LIST,1]; ch2=LIST[n_LIST,2];
    E[ch1,ch2] = abs(mean([st[ch1],st[ch2]]) / ct[n_LIST] - 1);
    E[ch2,ch1] = E[ch1,ch2];
end

heatmap(E)


#################


nr_delays=2; 
DDAmodel=[[0 0 1];  
          [0 0 2]; 
          [1 1 1]];
(MODEL, L_AF, DDAorder)=make_MODEL(DDAmodel);                    

FN_DDA="ROS_4.DDA";

if Sys.iswindows()
   if !isfile("run_DDA_AsciiEdf.exe")
      run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
   end
   CMD=".\\run_DDA_AsciiEdf.exe";
else
   CMD="./run_DDA_AsciiEdf";
end
CMD = "$CMD -ASCII";                                 
CMD = "$CMD -MODEL $(join(MODEL," "))"                    
CMD = "$CMD -TAU $(join(TAU," "))"                        
CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays"    
CMD = "$CMD -DATA_FN $FN_DATA -OUT_FN $FN_DDA"        
CMD = "$CMD -WL $WL -WS $WS";                            
CMD = "$CMD -SELECT 1 1 0 0";  
CMD = "$CMD -CH_list $(join(LIST'[:]," "))";              
CMD = "$CMD -WL_CT 2 -WS_CT 2";                                       
if Sys.iswindows()
   run(Cmd(string.(split(CMD, " "))));
else
   run(`sh -c $CMD`);
end

ST2=readdlm("ROS_4.DDA_ST"); ST2=ST2[:,3:end];
CT2=readdlm("ROS_4.DDA_CT"); CT2=CT2[:,3:end];

mean(reshape(ST,WN,size(ST,2)*size(ST,3)) .- ST2)
mean(reshape(CT,WN,size(CT,2)*size(CT,3)) .- CT2)

