%GFP test, Koenig Melie-Garcia 2010

%subjects = {'gm'} %when you want to plot the grand mean GFP across participants
subjects = {'4';'5';'6';'7';'8';'9';'10';'12';'13';'14';'15';'16';'17';'18';'19';'20'} %when you want to do the stats

conds = {'11';'12';'21';'22';'31';'32';'41';'42';'51';'52';'61';'62';'71';'72';'81';'82'};

%for colour factor
labels = {'blue';'blue';'green';'green';'red';'red';'yellow';'yellow'};

%Component to analyze (in samples)
comp=[94:112] %we need time of MMN - in paper it is between 162 and 232 ms, i.e. the early part of MMN
base=[25:52] %-100 to 0 baseline

sampler=256; %how many samples in a trial? needed to create an empty gfp matrix

%now we want to write a file, in which we copy relevant information
%triplet and colour
filepath = ['E:\Thierry_Data\VEPs\'];
filepath = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\data\'];
fid2=fopen(sprintf('%sRstats_new_GFPs.txt',filepath),'w');

%print header
fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %s\t %s\n','participant','Lightness','Colour','Deviancy','Condition no','baseGFP','compGFP');


gfp=zeros(size(subjects,1),size(conds,1),sampler);

for nsub=1:size(subjects,1)
    for ncond=1:size(conds,1)      
        
EEG = pop_loadset([ subjects{nsub} '_C' conds{ncond} '.set'], filepath);

ta=EEG.times; %256 samples

nElectrodes = size(EEG.data,1);
nObservations = size(EEG.data,3);
nDataPoints = size(EEG.data,2);

gfp(nsub,ncond,:) = std(mean(EEG.data(1:64,:,:),3),0,1); % GFP of the mean across observations; parameter 2 is normalisation but it doesn't seem to matter whether it is zero or 1

%strip the colour of the condition from the deviancy
cstr=char(conds(ncond));
colour=str2double(cstr(1)); deviancy=str2double(cstr(2));

   %for stats
       dummy = gfp(nsub,ncond,comp); 
       basedummy = gfp(nsub,ncond,base);
       dummy= squeeze(mean(dummy,3));
       basedummy = squeeze(mean(basedummy,3));

       if rem(deviancy, 2) ~= 0 %odd numbers
    sat="standard";
elseif rem(deviancy, 2) == 0 %even numbers
    sat="deviant";
       end

  if rem(colour, 2) ~= 0 %odd numbers
   if rem(deviancy, 2) == 0
      lightness="light";
   else
       lightness="dark";
   end
elseif rem(colour, 2) == 0 %even numbers
       if rem(deviancy, 2) == 0
      lightness="dark";
   else
    lightness="light";
       end    
            end

              fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %6f\t %6f\n',subjects{nsub},lightness,labels{colour},sat,cstr,basedummy,dummy);

    end    
end

fclose(fid2); %close file with data for stats

%plot 1
gfp_c1=squeeze(mean(gfp(:,1:4,:),1)); %blue
col_c1={[0,0,0.6];[0.4,0.4,1];[0.4,0.4,1];[0,0,0.6]};
linestyle={'--';'-';'--';'-'}; %this will stay the same for all the plots
figure
for i = 1:4
    plot(ta,gfp_c1(i,:),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([0 6]);
    hold on;
end
    yline([0],'--');
    xline([0],'--');
    title('blue');
    legend({'dark std','light dev','light std','dark dev'});
%plot2
gfp_c2=squeeze(mean(gfp(:,5:8,:),1));
col_c2={[0,0.4,0];[0.3,0.8,0.3];[0.3,0.8,0.3];[0,0.4,0]};

figure
for i = 1:4
    plot(ta,gfp_c2(i,:),'color',col_c2{i},'LineStyle',linestyle{i},'LineWidth',2);
 ylim([0 6]);
    hold on;
end
    yline([0],'--');
    xline([0],'--');
    title('green');
    legend({'dark std','light dev','light std','dark dev'});
%plot3
gfp_c3=squeeze(mean(gfp(:,9:12,:),1));
col_c3={[0.5,0,0];[1,0.4,0.4];[1,0.4,0.4];[0.5,0,0]};
figure
for i = 1:4
    plot(ta,gfp_c3(i,:),'color',col_c3{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([0 6]);
    hold on;
end
    yline([0],'--');
    xline([0],'--');
    title('red/pink');
    legend({'red std',' pink dev','pink std','red dev'});
%plot4
gfp_c4=squeeze(mean(gfp(:,13:16,:),1));
col_c4={[0.4,0.4,0];[0.9,0.9,0.4];[0.9,0.9,0.4];[0.4,0.4,0]};
figure
for i = 1:4
    plot(ta,gfp_c4(i,:),'color',col_c4{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([0 6]);
    hold on;
end
    yline([0],'--');
    xline([0],'--');
    title('yellow');
    legend({'brown std','yellow dev','yellow std','brown dev'});


