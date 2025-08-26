#=
import Pkg; Pkg.add("Combinatorics")
import Pkg; Pkg.add("LinearAlgebra")
import Pkg; Pkg.add("Printf")
import Pkg; Pkg.add("DelimitedFiles")
import Pkg; Pkg.add("Plots")
import Pkg; Pkg.add("StatsBase")

import Pkg; Pkg.add("LaTeXStrings")
import Pkg; Pkg.add("Graphs")
import Pkg; Pkg.add("GraphRecipes")
import Pkg; Pkg.add("Colors")
import Pkg; Pkg.add("JLD2")
=#

using Plots
using LaTeXStrings
using Colors
using Printf
using StatsBase
using DelimitedFiles
using Combinatorics
using LinearAlgebra
using Graphs
using GraphRecipes
using JLD2


if Sys.iswindows()
   SL="\\";
else
   SL="/";
end

function index(DIM, ORDER)
    B = ones(DIM^ORDER, ORDER)
    
    if DIM > 1
        for i = 2:(DIM^ORDER)
            if B[i-1, ORDER] < DIM
                B[i, ORDER] = B[i-1, ORDER] + 1
            end
            
            for i_DIM = 1:ORDER-1
                if round((i/DIM^i_DIM - floor(i/DIM^i_DIM))*DIM^i_DIM) == 1
                    if B[i-DIM^i_DIM, ORDER-i_DIM] < DIM
                        for j = 0:DIM^i_DIM-1
                            B[i+j, ORDER-i_DIM] = B[i+j-DIM^i_DIM, ORDER-i_DIM] + 1
                        end
                    end
                end
            end
        end
        
        i_BB = 1
        BB = Vector{Int}[]
        for i = 1:size(B,1)
            jn = 1
            for j = 2:ORDER
                if B[i, j] >= B[i, j-1]
                    jn += 1
                end
            end
            if jn == ORDER
                push!(BB, B[i, :])
                i_BB += 1
            end
        end
    else
        println("DIM=1!!!")
    end
    
    return hcat(BB...)
end

function monomial_list(nr_delays, order)
    # monomials
    P_ODE = index(nr_delays+1, order)'
    P_ODE = P_ODE .- ones(Int64,size(P_ODE))
    
    P_ODE = P_ODE[2:size(P_ODE,1),:];
    
    return P_ODE
end

function make_MOD_nr(SYST,NrSyst)

   DIM=length(unique(SYST[:,1]));
   order=size(SYST,2)-1;

   P=[[0 0]; monomial_list(DIM * NrSyst,order)];

   MOD_nr=fill(0,size(SYST,1)*NrSyst,2);
   for n=1:NrSyst
       for i=1:size(SYST,1)
           II=SYST[i,2:end]';
           II[II .> 0] .+= DIM*(n-1);

           Nr=i+size(SYST,1)*(n-1);
           MOD_nr[Nr,2]=findall( sum(abs.(repeat(II,size(P,1),1)-P),dims=2)' .== 0 )[1][2] - 1;
           MOD_nr[Nr,1]=SYST[i,1]+DIM*(n-1);
       end
       #P[MOD_nr[1:size(SYST,1),2].+1,1:2]
   end
   MOD_nr=reshape(MOD_nr',size(SYST,1)*NrSyst*2)';

   return MOD_nr,DIM,order,P

end


function integrate_ODE_general_BIG(MOD_nr,MOD_par,dt,L,DIM,ODEorder,X0,FNout,CH_list,DELTA,TRANS=nothing)
  if TRANS===nothing
     TRANS=0;
  end

  if Sys.iswindows()
     if !isfile("i_ODE_general_BIG.exe")
        cp("i_ODE_general_BIG","i_ODE_general_BIG.exe");
     end

     CMD=".\\i_ODE_general_BIG.exe";
  else
     CMD="./i_ODE_general_BIG";
  end

  MOD_NR = join(MOD_nr, " ");
  CMD = "$CMD -MODEL $MOD_NR";
  MOD_PAR = join(MOD_par, " ");
  CMD = "$CMD -PAR $MOD_PAR";
  ANF=join(X0," ");
  CMD = "$CMD -ANF $ANF";
  CMD = "$CMD -dt $dt";
  CMD = "$CMD -L $L";
  CMD = "$CMD -DIM $DIM";
  CMD = "$CMD -order $ODEorder";
  if TRANS>0
     CMD = "$CMD -TRANS $TRANS";
  end
  if length(FNout)>0
     CMD = "$CMD -FILE $FNout";
  end
  CMD = "$CMD -DELTA $DELTA";
  CMD = "$CMD -CH_list $(join(CH_list," "))";

  if length(FNout)>0
     if Sys.iswindows()
        run(Cmd(string.(split(CMD, " "))));
     else
        run(`sh -c $CMD`);
     end
  else
     if Sys.iswindows()
       X = read(Cmd(string.(split(CMD, " "))),String);
     else
       X = read(`sh -c $CMD`,String);
     end
     X = split(strip(X), '\n');
     X = hcat([parse.(Float64, split(row)) for row in X]...)';

     return X
  end
end

function MakeDataNoNoise(PF,FromTo,CH_list,DELTA)
  epsilon=0.15;  

  II=make_MOD_nr_Coupling(FromTo,DIM,P);  
  MOD_par_add=repeat([epsilon -epsilon],1,size(FromTo,1));

  L1=WS*(WN-1)+WL+TM+2*dm-1; 
  TRANS=20000; dt=0.05;

  FN=@sprintf("%s%sCD_DDA_data_NoNoise__WL%d_WS%d_WN%d%s.ascii",
             DATA_DIR,SL,WL,WS,WN,PF);

  if !isfile(FN)
     X0 = rand(DIM*NrSyst,1);    
     X=integrate_ODE_general_BIG([MOD_nr II],[MOD_par MOD_par_add],
                                 dt,                 
                                 L1*2,              
                                 DIM*NrSyst,ODEorder,X0,
                                 FN,CH_list,DELTA,
                                 TRANS);               
     #X = X[1:2:end,1:3:end];     
     #writedlm(FN, map(number_to_string, X),' '); 
  end
end                    

function MakeDataNoise(PF,NOISE,SNRadd_list)
  for SNRadd=SNRadd_list
      if NOISE=="NoNoise"
         NOISEadd=@sprintf("%02ddB",SNRadd);  
         noise="NoNoise";        
      else
         NOISEadd=@sprintf("%02ddB_add%02d",NOISE,SNRadd);
         noise=@sprintf("%02ddB",NOISE);
      end
      FNadd=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.ascii",
                DATA_DIR,SL,NOISEadd,WL,WS,WN,PF);
      if !isfile(FNadd)
         FN=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.ascii",
                      DATA_DIR,SL,noise,WL,WS,WN,PF);
         X=readdlm(FN);
         for k=1:size(X,2)               
             X[:,k]=add_noise(X[:,k],SNRadd);
         end
         writedlm(FNadd, map(number_to_string, X),' '); 
         X = nothing; GC.gc();
      end
  end
end

function make_MOD_nr_Coupling(FromTo,DIM,P)

   order=size(P,2);
   II=fill(0,size(FromTo,1),4);
   for j=1:size(II,1)
       n1=FromTo[j,1]; k1=FromTo[j,2]+1; range1=3:3+order-1;
       n2=FromTo[j,1+range1[end]]; k2=FromTo[j,2+range1[end]]+1; range2=range1 .+ range1[end];

       JJ=FromTo[j,range1]'; JJ[JJ .> 0] .+= DIM * (n1-1);
       II[j,4] = findall( sum(abs.(repeat(JJ,size(P,1),1)-P),dims=2)' .== 0 )[1][2] - 1;

       JJ=FromTo[j,range2]'; JJ[JJ .> 0] .+= DIM * (n2-1);
       II[j,2] = findall( sum(abs.(repeat(JJ,size(P,1),1)-P),dims=2)' .== 0 )[1][2] - 1;

       II[j,1] = DIM*n2-(DIM-k2)-1;
       II[j,3] = DIM*n2-(DIM-k1)-1;
   end
   II=reshape(II',length(II))';

   return II

end


function integrate_ODE_general_BIG(MOD_nr,MOD_par,dt,L,DIM,order,X0,FNout,CH_list,DELTA,TRANS=nothing)
  if TRANS===nothing
     TRANS=0;
  end

  if Sys.iswindows()
     if !isfile("i_ODE_general_BIG.exe")
        run(`cp i_ODE_general_BIG i_ODE_general_BIG.exe`);
     end

     CMD=".\\i_ODE_general_BIG.exe";
  else
     CMD="./i_ODE_general_BIG";
  end

  MOD_NR = join(MOD_nr, " ");
  CMD = "$CMD -MODEL $MOD_NR";
  MOD_PAR = join(MOD_par, " ");
  CMD = "$CMD -PAR $MOD_PAR";
  ANF=join(X0," ");
  CMD = "$CMD -ANF $ANF";
  CMD = "$CMD -dt $(string(dt))";
  CMD = "$CMD -L $(string(L))";
  CMD = "$CMD -DIM $(string(DIM))";
  CMD = "$CMD -order $(string(order))";
  if TRANS>0
     CMD = "$CMD -TRANS $(string(TRANS))";
  end
  CMD = "$CMD -FILE $FNout";
  CMD = "$CMD -CH_list $(join(CH_list," ")) -DELTA $(string(DELTA))";

  if Sys.iswindows()
     run(Cmd(string.(split(CMD, " "))));
  else
     run(`sh -c $CMD`);
  end
end


function dir_exist(DIR)
    if !isdir(DIR)
        mkdir(DIR)
    end
end

function add_noise(s,SNR)
   N = length(s);

   n = randn(N);
   n .= (n.-mean(n))./std(n);
   # c is given  from SNR = 10*log10( var(s)/var(c*n) )
   c = sqrt( var(s)*10^-(SNR/10) );
   
   s_out = (s+c.*n);

   return s_out
end

function number_to_string(n::Number)
   return @sprintf("%.15f", n);
end

function RunDDA(PF,NOISE,SNRadd_list)
  if NOISE=="NoNoise"
     noise="NoNoise";        
  else
     noise=@sprintf("%02ddB",NOISE);
  end

  for n_SNRadd=0:length(SNRadd_list)
      if n_SNRadd==0
         FN_DDA=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.DDA",
                         DDA_DIR,SL,noise,WL,WS,WN,PF);   
         FN_data=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.ascii",
                         DATA_DIR,SL,noise,WL,WS,WN,PF);
      else   
         SNRadd=SNRadd_list[n_SNRadd];
         if NOISE=="NoNoise"
            NOISEadd=@sprintf("%02ddB",SNRadd);  
         else
            NOISEadd=@sprintf("%02ddB_add%02d",NOISE,SNRadd);
         end
         FN_DDA=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.DDA",
                         DDA_DIR,SL,NOISEadd,WL,WS,WN,PF);
         FN_data=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.ascii",
                         DATA_DIR,SL,NOISEadd,WL,WS,WN,PF);
      end

      if !isfile(join([FN_DDA,"_ST"]))
         if Sys.iswindows()
            if !isfile("run_DDA_AsciiEdf.exe")
               run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
            end

            CMD=".\\run_DDA_AsciiEdf.exe";
         else
            CMD="./run_DDA_AsciiEdf";
         end
         CMD = "$CMD -ASCII";   
         CMD = "$CMD -MODEL $(join(MODEL," "))";  
         CMD = "$CMD -TAU $(join(TAU," "))";     
         CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays";  
         CMD = "$CMD -DATA_FN $FN_data -OUT_FN $FN_DDA";   
         CMD = "$CMD -WL $WL -WS $WS";             
         CMD = "$CMD -SELECT 1 1 1 0";             
         CMD = "$CMD -WL_CT 2 -WS_CT 2";            
         CMD = "$CMD -CH_list $(join(LL1," "))";  
         
         if Sys.iswindows()
            run(Cmd(string.(split(CMD, " "))));
         else
            run(`sh -c $CMD`);
         end
 
         rm(@sprintf("%s.info",FN_DDA));     
      end
  end
end

function make_MODEL_new(MOD,SSYM,mm)
   MODEL=findall(x -> x == 1, MOD[mm,:]);
   L_AF=length(MODEL)+1;
   SYM=SSYM[mm,:];
   model = join([lpad(x, 2, '0') for x in MODEL], "_");

   return MODEL,SYM,model,L_AF
end

function make_MODEL(SYST)

   order=size(SYST,2);
   nr_delays=2;

   P_ODE=monomial_list(nr_delays,order);

   MODEL=fill(0,size(SYST,1))';
   for i=1:size(SYST,1)
       II=SYST[i,:]';

       MODEL[i] = findall( sum(abs.(repeat(II,size(P_ODE,1),1)-P_ODE),dims=2)' .== 0 )[1][2];
   end
   #P_ODE[MODEL',:]
   L_AF=length(MODEL)+1;

   return MODEL, L_AF, order

end

function makeCE(PF,NOISE,SNRadd_list)
  c=fill(NaN,NrSyst,NrSyst,1+length(SNRadd_list));
  e=fill(NaN,NrSyst,NrSyst,1+length(SNRadd_list));

  if NOISE=="NoNoise"
     noise="NoNoise";        
  else
     noise=@sprintf("%02ddB",NOISE);
  end

  for n_SNRadd=0:length(SNRadd_list)
      if n_SNRadd==0
         FN_DDA=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.DDA",
                         DDA_DIR,SL,noise,WL,WS,WN,PF);   
      else   
         SNRadd=SNRadd_list[n_SNRadd];
         if NOISE=="NoNoise"
            NOISEadd=@sprintf("%02ddB",SNRadd);  
         else
            NOISEadd=@sprintf("%02ddB_add%02d",NOISE,SNRadd);
         end
         FN_DDA=@sprintf("%s%sCD_DDA_data_%s__WL%d_WS%d_WN%d%s.DDA",
                         DDA_DIR,SL,NOISEadd,WL,WS,WN,PF);
      end

      ST=readdlm(join([FN_DDA,"_ST"]));            
      T=ST[:,1:2]; ST=ST[:,3:end];                
                                                 
      global WN=Int(size(T,1));                        
      ST=ST[:,L_AF:L_AF:end];                  
      ST=reshape(ST,WN,7);                    
      
      CT=readdlm(join([FN_DDA,"_CT"]));       
      CT=CT[:,3:end];                      
                                             
      CT=CT[:,L_AF:L_AF:end];               
      CT=reshape(CT,WN,size(LIST,1));     
      
      E=fill(NaN,WN,NrSyst,NrSyst);   
      for l=1:size(LIST,1)
          ch1=LIST[l,1];ch2=LIST[l,2];
          E[:,ch1,ch2] = abs.( dropdims(mean(ST[:,[ch1,ch2]],dims=2),dims=2) ./ CT[:,l] .- 1 );
          E[:,ch2,ch1] = E[:,ch1,ch2];
      end    
      E=dropdims(mean(E,dims=1),dims=1);           

      CD=readdlm(join([FN_DDA,"_CD_DDA_ST"]));
      CD=CD[:,3:end];                        
                                      
      CD=reshape(CD,WN,2,size(LIST,1));    

      C=fill(NaN,WN,NrSyst,NrSyst);  
      for l=1:size(LIST,1)
          ch1=LIST[l,1];ch2=LIST[l,2];
          C[:,ch1,ch2] = CD[:,2,l];
          C[:,ch2,ch1] = CD[:,1,l];
      end    
      C[isnan.(C)].=0;
      C=dropdims(mean(C,dims=1),dims=1);  

      e[:,:,n_SNRadd+1]=E;
      c[:,:,n_SNRadd+1]=C;
  end

  return c,e
end

function deriv_all(data, dm, order=nothing, dt=nothing)
   if order===nothing 
      order=2 
   end
   if dt===nothing 
      dt=1
   end

   t=collect(1+dm:length(data)-dm)
   L=length(t)

   #### second order:
   
   if order==2
      ddata = zeros(L)
      for n1=1:dm
          ddata += (data[t.+n1].-data[t.-n1])/n1
      end
      ddata /= (dm/dt);  
   end
   
   #### third order:
   
   if order==3
      ddata = zeros(L)
      d=0;
      for n1=1:dm
          for n2=n1+1:dm
              d+=1
              ddata -= (((data[t.-n2].-data[t.+n2])*n1^3- 
                      (data[t.-n1].-data[t.+n1])*n2^3)/(n1^3*n2-n1*n2^3))
          end
      end
      ddata /= (d/dt);  
   end
   
   return ddata
end



function deriv_all(data, dm, order=nothing, dt=nothing)
   if order===nothing
      order=2
   end
   if dt===nothing
      dt=1
   end

   t=collect(1+dm:length(data)-dm)
   L=length(t)

   #### second order:

   if order==2
      ddata = zeros(L)
      for n1=1:dm
          ddata += (data[t.+n1].-data[t.-n1])/n1
      end
      ddata /= (dm/dt);
   end

   #### third order:

   if order==3
      ddata = zeros(L)
      d=0;
      for n1=1:dm
          for n2=n1+1:dm
              d+=1
              ddata -= (((data[t.-n2].-data[t.+n2])*n1^3-
                      (data[t.-n1].-data[t.+n1])*n2^3)/(n1^3*n2-n1*n2^3))
          end
      end
      ddata /= (d/dt);
   end

   return ddata
end

