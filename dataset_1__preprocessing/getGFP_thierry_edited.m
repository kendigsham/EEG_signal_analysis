%GFP test, Koenig Melie-Garcia 2010

%subjects = {'gm'} %when you want to plot the grand mean GFP across participants
subjects = {'4';'5';'6';'7';'8';'9';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20'} %when you want to do the stats

conds = {'11';'111';'1111';'11111';'9211';'9311';'9411';'22';'21';'221';'2221';'22221';'9221';'9321';'9421';'12'}; %all blues 
conds2 = {'31';'331';'3331';'33331';'9231';'9331';'9431';'42';'41';'441';'4441';'44441';'9241';'9341';'9441';'32'}; %all greens 
conds3 = {'51';'551';'5551';'55551';'9251';'9351';'9451';'62';'61';'661';'6661';'66661';'9261';'9361';'9461';'52'}; %all red/pinks
conds4 = {'71';'771';'7771';'77771';'9271';'9371';'9471';'82';'81';'881';'8881';'88881';'9281';'9381';'9481';'72'}; %all yellow/browns 

%for colour factor
labels = {'blue';'blue';'green';'green';'red';'red';'yellow';'yellow'};

%Component to analyze (in samples)
comp=[94:112] %we need time of MMN - in paper it is between 162 and 232 ms, i.e. the early part of MMN
base=[25:52] %-100 to 0 baseline

sampler=256; %how many samples in a trial? needed to create an empty gfp matrix

%now we want to write a file, in which we copy relevant information
%triplet and colour
filepath = ['E:\Thierry_Data\VEPs\'];
fid2=fopen(sprintf('%sRstats_edited_GFPs.txt',filepath),'w');

%print header
fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %s\t %s\n','participant','Lightness','Colour','Deviancy','Condition no','baseGFP','compGFP');


gfp=zeros(size(subjects,1),size(conds,1),sampler);

for nsub=1:size(subjects,1)
    for ncond=1:size(conds,1)

        FilePath = ['E:\Thierry_Data\dataedited\'];
        
EEG = pop_loadset([ subjects{nsub} '_C' conds{ncond} '.set'], FilePath);

ta=EEG.times; %256 samples

nElectrodes = size(EEG.data,1);
nObservations = size(EEG.data,3);
nDataPoints = size(EEG.data,2);

gfp(nsub,ncond,:) = std(mean(EEG.data,3),1,1); % GFP of the mean across observations

% %strip the colour of the condition from the deviancy
% cstr=char(conds(ncond));
% colour=str2double(cstr(1)); deviancy=str2double(cstr(2));
% 
%    %for stats
%        dummy = gfp(nsub,ncond,comp); 
%        basedummy = gfp(nsub,ncond,base);
%        dummy= squeeze(mean(dummy,3));
%        basedummy = squeeze(mean(basedummy,3));
% 
%        if rem(deviancy, 2) ~= 0 %odd numbers
%     sat="standard";
% elseif rem(deviancy, 2) == 0 %even numbers
%     sat="deviant";
%        end
% 
%   if rem(colour, 2) ~= 0 %odd numbers
%    if rem(deviancy, 2) == 0
%       lightness="light";
%    else
%        lightness="dark";
%    end
% elseif rem(colour, 2) == 0 %even numbers
%        if rem(deviancy, 2) == 0
%       lightness="dark";
%    else
%     lightness="light";
%        end    
%             end
% 
%               fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %6f\t %6f\n',subjects{nsub},lightness,labels{colour},sat,cstr,basedummy,dummy);

    end    
end

fclose(fid2); %close file with data for stats

%plot 1 - blue
gfpc1=squeeze(mean(gfp,1)); %average over participants
gfp_c1=[(gfpc1(1,:)+gfpc1(9,:))/2; (gfpc1(2,:)+gfpc1(10,:))/2; (gfpc1(3,:)+gfpc1(11,:))/2; (gfpc1(4,:)+gfpc1(12,:))/2; (gfpc1(5,:)+gfpc1(13,:))/2; (gfpc1(6,:)+gfpc1(14,:))/2; (gfpc1(7,:)+gfpc1(15,:))/2; (gfpc1(8,:)+gfpc1(16,:))/2]; 
col_c1={[0,0,0.2];[0,0,0.5];[0.1,0.1,0.7];[0.1,0.1,1];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.85,0.85,0.85];[0,0,0]};
linestyle={'-';'-';'-';'-';'-';'-';'-';':'}; %this will stay the same for all the plots
figure
for i = 1:size(gfp_c1,1)
    plot(ta,gfp_c1(i,:),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([0 6]);
    hold on;
end
    yline([0],'--');
    xline([0],'--');
    title('blue');
    legend({'standard 2','standard 3','standard 4','standard 5','standard1_2','standard1_3','standard1_4','deviant'});
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


