function MinError(FN_data,CH_list,DDA_DIR,DDAorder,dm,WL,WS,DELAYS,yn,ALLE,StartEnd,AsciiEdf,DDAmodel=nothing)
   STOP=0;
   if DDAmodel === nothing
      DDAmodel = [];
   else
      #= e.g.:
      DDAmodel =[[0 0 1];  
                 [0 0 2]; 
                 [0 1 1]];
      =#
      (MODEL, L_AF, DDAorder1)=make_MODEL(DDAmodel);  
      if DDAorder1 !== DDAorder
         STOP = 1;
      end
      SYM=[length(unique([value for value in DDAmodel[:] if value > 0])) -1];
      DDAmodel2 = replace(replace(DDAmodel, 1 => -1, 2 => 1), -1 => 2);
      DDAmodel2 = sortslices(sort(DDAmodel2,dims=2),dims=1);
      if DDAmodel == DDAmodel2
         SYM[2] = 1;
      else
         SYM[2] = 0;
      end; 
      SSYM=SYM;
      P_DDA=monomial_list(nr_delays,DDAorder);
      MOD=fill(0,1,size(P_DDA,1))
      MOD[MODEL] .= 1; 
      model = join([lpad(x, 2, '0') for x in MODEL], "_");
      N_MOD=size(DDAmodel,2);
   end

   # e.g. CH_list=[[6:10];[2];[[5; 6:7]]] or CH_list = [1:10] or CH_list = [[1];[2];[3]];
   NrCH=length(CH_list);
   if NrCH>0
      CHs = vcat(CH_list[1:NrCH][:]...);
      if ALLE !== "ALL"
         CHs_su = sort(union(CHs));
         ChIDX=map(i -> findfirst(CHs[i] .== CHs_su),1:length(CHs));
         # CHs_su[ChIDX] is the same as CHs
      end
   else
      ALLE="";
   end

   if length(DDAmodel)==0
      N_MOD=3; 
      (MOD,P_DDA,SSYM) = make_MOD_new_new(N_MOD,nr_delays,DDAorder);
   end

   if STOP == 0

      make_TAU_ALL(SSYM,DELAYS);   
       
      MIN=fill(1000.0,NrCH); mm_select=fill(0,NrCH); TAU_select=fill(-1,NrCH,2);
      for mm = 1:size(MOD,1)
   
          if length(DDAmodel)==0
             (MODEL,SYM,model,L_AF) = make_MODEL_new(MOD,SSYM,mm);
          end
          TAU_name = @sprintf("TAU_ALL__%d_%d",SYM[1],SYM[2]);
          TAU = readdlm(TAU_name); N_TAU=size(TAU,1);
   
          FN_DDA = @sprintf("%s%s%s",DDA_DIR,SL,model);
      
          if !isfile(@sprintf("%s_ST",FN_DDA)) || yn=="yes"
             println(mm);
   
             if Sys.iswindows()
                if !isfile("run_DDA_AsciiEdf.exe")
                   run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
                end
           
                CMD=".\\run_DDA_AsciiEdf.exe";
             else
                CMD="./run_DDA_AsciiEdf";
             end

             if AsciiEdf == "ASCII"
                CMD = "$CMD -ASCII";
             else
                CMD = "$CMD -EDF";
             end
             CMD = "$CMD -MODEL $(join(MODEL, " "))";
             CMD = "$CMD -TAU_file $TAU_name";             
             CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays";
             CMD = "$CMD -DATA_FN $FN_data -OUT_FN $FN_DDA";
             CMD = "$CMD -SELECT 1 0 0 0";
             if length(StartEnd)>0
             CMD = "$CMD -StartEnd $(join(StartEnd, " "))";
             end
             if NrCH>0 
                if ALLE !== "ALL"
                   CMD = "$CMD -CH_list $(join(CHs_su, " "))";
                end
             end
             if length(WL)>0
                CMD = "$CMD -WL $WL";
             end
             if length(WS)>0
                CMD = "$CMD -WS $WS";
             end
         
             if Sys.iswindows()
                run(Cmd(string.(split(CMD, " "))));
             else
                run(`sh -c $CMD`);
             end
          end
   
          if NrCH>0
             AF=readdlm(join([FN_DDA,"_ST"])); AF=AF[:,3:end]; WN=size(AF,1);
   
             if ALLE == "ALL"
                CHs_su = collect(1:Int(round(size(AF,2)/L_AF/N_TAU)));
                ChIDX=map(i -> findfirst(CHs[i] .== CHs_su),1:length(CHs));
             end

             AF=reshape(AF,WN,L_AF,length(CHs_su),N_TAU)[:,end,:,:];
             AF=AF[:,ChIDX,:];
             L_CH = [0;cumsum(length.(CH_list))]; 
             for k=1:NrCH
                 AF[1,k,:] = mean(AF[:,L_CH[k]+1:L_CH[k+1],:],dims=2);
             end
             AF=dropdims(mean(AF[:,1:NrCH,:],dims=1),dims=1);
      
             M=minimum(AF,dims=2);
             for k=1:NrCH
                 if M[k] < MIN[k]
                    MIN[k] = M[k];
                    mm_select[k] = mm;
                    TAU_select[k,1:size(TAU,2)]=TAU[findfirst(M[k] .== AF[k,:]),:];
                 end
             end
          end
      end

      if NrCH>0
         for k=1:NrCH
             CH_list[k] = collect(CH_list[k]);
             @printf("\nCH = ");
             for i=1:length(CH_list[k])
                 @printf("%d ",CH_list[k][i]);
             end
             @printf(":\n  Best model:\n");
             p=P_DDA[findall(MOD[mm_select[k],:] .== 1),:];
             for i=1:N_MOD
                 @printf("    [ ");
                 for j=1:DDAorder
                     @printf("%d ",p[i,j]);
                 end
                 @printf("]\n");
             end
             @printf("  Best delays: [ ");
             for i=1:nr_delays
                 if TAU_select[k,i]>0
                    @printf("%d ",TAU_select[k,i]);
                 end
             end
             @printf("]\n\n");
         end
      end
   else
      @printf("\nDDA order inconsistant\n\n");
      TAU_select=nothing; 
      mm_select=nothing;
   end
   
   return TAU_select, mm_select
end

