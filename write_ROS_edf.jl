include("DDAfunctions.jl");

using EDF
using EDF: TimestampedAnnotationList, PatientID, RecordingID, SignalHeader,
           Signal, AnnotationsSignal
using Dates
using FilePathsBase

# modified from https://github.com/beacon-biosignals/EDF.jl/blob/main/src/write.jl
# EDF+C is for edf+ files  -  removed
function EDF.write_header(io::IO, file::EDF.File)
    length(file.signals) <= 9999 ||
        error("EDF does not allow files with more than 9999 signals")
    expected_bytes_written = EDF.BYTES_PER_FILE_HEADER +
                             EDF.BYTES_PER_SIGNAL_HEADER * length(file.signals)
    bytes_written = 0
    bytes_written += EDF.edf_write(io, file.header.version, 8)
    bytes_written += EDF.edf_write(io, file.header.patient, 80)
    bytes_written += EDF.edf_write(io, file.header.recording, 80)
    bytes_written += EDF.edf_write(io, file.header.start, 16)
    bytes_written += EDF.edf_write(io, expected_bytes_written, 8)
    ####
    bytes_written += EDF.edf_write(io, file.header.is_contiguous ? "     " : "EDF+D", 44)
    # bytes_written += EDF.edf_write(io, file.header.is_contiguous ? "EDF+C" : "EDF+D", 44)
    ####
    bytes_written += EDF.edf_write(io, file.header.record_count, 8)
    bytes_written += EDF.edf_write(io, file.header.seconds_per_record, 8)
    bytes_written += EDF.edf_write(io, length(file.signals), 4)
    signal_headers = EDF.SignalHeader.(file.signals)
    for (field_name, byte_limit) in EDF.SIGNAL_HEADER_FIELDS
        for signal_header in signal_headers
            field = getfield(signal_header, field_name)
            bytes_written += EDF.edf_write(io, field, byte_limit)
        end
    end
    bytes_written += EDF.edf_write(io, ' ', 32 * length(file.signals))
    @assert bytes_written == expected_bytes_written
    return bytes_written
end



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
CH_list = 1:DIM*NrSyst;
DELTA=2;
FN_DATA = "ROS_4_edf_test.ascii";
L=20000;
if !isfile(FN_DATA)
   integrate_ODE_general_BIG(MOD_nr,MOD_par,dt,L,DIM*NrSyst,ODEorder,X0,FN_DATA,CH_list,DELTA,TRANS);
end

Y=readdlm(FN_DATA);
Y = Y .- minimum(filter(!isnan,Y[:]));
Y = Int.(floor.(Y ./ maximum(filter(!isnan,Y[:])) .* 65535 .- 32768));


edf_A = L;
edf_B = 1000;
edf_C = Int(edf_A/edf_B);     # make sure edf_B is a multiple of L
                              # if not, pad with 0

edf_D = Int(1/dt);


header = EDF.FileHeader(
       "0",
       "Roessler",                            # new patient name
       "Date:01.01.10 Time:00.00.00",
       DateTime("2010-01-01T00:00:00"),
       true,
       edf_C,                                 # number of segments for each channel
       edf_D                                  # seconds_per_record, here 1/dt
       );


LABEL=fill("",size(Y,2));
xyz=replace(mod.(1:size(Y,2),3),0 => "z", 1 => "x", 2 => "y");
sys=Int.(floor.((collect(1:12).-1) ./ 3)) .+ 1;
for ch=1:size(Y,2)
    LABEL[ch] = @sprintf("ROS_%s_Nr%d",xyz[ch],sys[ch]);
end

signals = Array{Union{EDF.AnnotationsSignal, EDF.Signal{Int16}},1}(undef,size(Y,2));
for ch in 1:size(Y,2)
    signal_header = EDF.SignalHeader(
        LABEL[ch],
        "",                     
        "aU", 
        0,
        1,
        0,
        1,
        "",                     
        edf_B
        );
    signals[ch] = EDF.Signal{Int16}(signal_header,Y[:,ch]);
end

EDF_file = replace(FN_DATA,".ascii" => ".edf");

open(EDF_file, "w") do io
    edf_file = EDF.File(io, header, signals);
    EDF.write(io, edf_file);
end
#the do-block form of open can be used to automatically close the file even in the case of exceptions



#=   test

WL=2000;WS=1000;TAU=[32 9];dm=4;order=3;nr_delays=2; TM=maximum(TAU);

nr_delays=2; 
DDAmodel=[[0 0 1];  
          [0 0 2]; 
          [1 1 1]];
(MODEL, L_AF, DDAorder)=make_MODEL(DDAmodel);                    

CHs=1:3:DIM*NrSyst;



FN_DDA=replace(EDF_file,".edf" => ".edf.DDA");
FN_DATA=replace(EDF_file,".edf" => ".edf");
if Sys.iswindows()
   if !isfile("run_DDA_AsciiEdf.exe")
      run(`cp run_DDA_AsciiEdf run_DDA_AsciiEdf.exe`);
   end
   CMD=".\\run_DDA_AsciiEdf.exe";
else
   CMD="./run_DDA_AsciiEdf";
end
CMD = "$CMD -EDF";                                 
CMD = "$CMD -MODEL $(join(MODEL," "))"                    
CMD = "$CMD -TAU $(join(TAU," "))"                        
CMD = "$CMD -dm $dm -order $DDAorder -nr_tau $nr_delays"    
CMD = "$CMD -DATA_FN $FN_DATA -OUT_FN $FN_DDA"        
CMD = "$CMD -WL $WL -WS $WS";                            
CMD = "$CMD -SELECT 1 0 0 0";  
CMD = "$CMD -CH_list $(join(CHs," "))";              
if Sys.iswindows()
   run(Cmd(string.(split(CMD, " "))));
else
   run(`sh -c $CMD`);
end

ST1=readdlm(replace(EDF_file,".edf" => ".edf.DDA_ST"));


####

FN_DATA = "ROS_4_edf_test.ascii";

FN_DDA=replace(FN_DATA,".ascii" => ".DDA");
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
CMD = "$CMD -SELECT 1 0 0 0";  
CMD = "$CMD -CH_list $(join(CHs," "))";              
if Sys.iswindows()
   run(Cmd(string.(split(CMD, " "))));
else
   run(`sh -c $CMD`);
end

ST2=readdlm(replace(FN_DATA,".ascii" => ".DDA_ST"));

#####

FN_DATA = "ROS_4_edf_test_Int16.ascii"
fid=open(FN_DATA,"w");
for k1=1:size(Y,1)
    for k2=1:size(Y,2)
       @printf(fid,"%6d ",Y[k1,k2]);
    end;
    @printf(fid,"\n");
end
close(fid)

FN_DDA=replace(FN_DATA,".ascii" => ".DDA");
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
CMD = "$CMD -SELECT 1 0 0 0";  
CMD = "$CMD -CH_list $(join(CHs," "))";              
if Sys.iswindows()
   run(Cmd(string.(split(CMD, " "))));
else
   run(`sh -c $CMD`);
end

ST3=readdlm(replace(FN_DATA,".ascii" => ".DDA_ST"));


=#
