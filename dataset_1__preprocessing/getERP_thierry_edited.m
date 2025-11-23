%ERP plots

subjects = {'gm'}
conds = {'11';'111';'1111';'11111';'9211';'9311';'9411';'22';'21';'221';'2221';'22221';'9221';'9321';'9421';'12'}; %all blues 
conds2 = {'31';'331';'3331';'33331';'9231';'9331';'9431';'42';'41';'441';'4441';'44441';'9241';'9341';'9441';'32'}; %all greens 
conds3 = {'51';'551';'5551';'55551';'9251';'9351';'9451';'62';'61';'661';'6661';'66661';'9261';'9361';'9461';'52'}; %all red/pinks
conds4 = {'71';'771';'7771';'77771';'9271';'9371';'9471';'82';'81';'881';'8881';'88881';'9281';'9381';'9481';'72'}; %all yellow/browns 

electrodes=[27 28 29 64 25 26]; %same as in thierry plots, leaving out p010 and po9
%electrodes=[25 26 30 63 62 27 29 64 28];%all PO,O,Iz

%y axis limits for plotting
ylim_low=-3;
ylim_high=10;
%samples to plot
ta1=26;
ta2=206; 
%lowpass flitering to use, in Hz
filtering=8;

%Component to analyze (in samples)
comp=[94:112] %we need time of MMN - in paper it is between 162 and 232 ms, i.e. the early part of MMN
base=[25:52] %-100 to 0 baseline

%now we want to write a file, in which we copy relevant information
%triplet and colour
filepath = ['e:\Thierry_Data\dataedited\'];
fid2=fopen(sprintf('%sRstats_VEPs.txt',filepath),'w');

%print header
fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %s\n','participant','Lightness','Colour','Deviancy','Condition no','amplitude');

labels = {'blue','blue','green','green','red','red','yellow','yellow'};

res_mat=[];

for nsub=1:size(subjects,1)
    for ncond=1:size(conds,1)

        
EEG = pop_loadset([ subjects{nsub} '_C' conds{ncond} '.set'], filepath);

%for plotting
ERP(ncond,:) = squeeze(mean(mean(EEG.data(electrodes,:,:)),3)); 

EEG = pop_loadset([ subjects{nsub} '_C' conds2{ncond} '.set'], filepath);

ERP2(ncond,:) = squeeze(mean(mean(EEG.data(electrodes,:,:)),3)); 

EEG = pop_loadset([ subjects{nsub} '_C' conds3{ncond} '.set'], filepath);

ERP3(ncond,:) = squeeze(mean(mean(EEG.data(electrodes,:,:)),3)); 

EEG = pop_loadset([ subjects{nsub} '_C' conds4{ncond} '.set'], filepath);

ERP4(ncond,:) = squeeze(mean(mean(EEG.data(electrodes,:,:)),3)); 

% %for stats
%        dummy = EEG.data(electrodes,comp,:); 
%        basedummy = EEG.data(electrodes,base,:);
%        dummy=squeeze(mean(mean(dummy,2),1)) - squeeze(mean(mean(basedummy,2),1));
% 
% %strip the colour of the condition from the deviancy
% cstr=char(conds(ncond));
% colour=str2double(cstr(1)); deviancy=str2double(cstr(2));
% 
%        if rem(deviancy, 2) ~= 0 %odd numbers
%     sat="standard";
% elseif rem(deviancy, 2) == 0 %even numbers
%     sat="deviant";
%        end
% 
%   if rem(colour, 2) ~= 0 %odd numbers
%    lightness="dark";
% elseif rem(colour, 2) == 0 %even numbers
%     lightness="light";
%             end
% 
%             for np=1:length(dummy)
%             fprintf(fid2,'%i\t %s\t %s\t %s\t %s\t %6f\n',np,lightness,labels{colour},sat,cstr,dummy(np));
%             end
              
    end
end

%get further parameters from EEG file
ta=EEG.times; 
srate=EEG.srate;

%plot 1
%average across dark and light blue
gfp_c1=[(ERP(1,:)+ERP(9,:))/2; (ERP(2,:)+ERP(10,:))/2; (ERP(3,:)+ERP(11,:))/2; (ERP(4,:)+ERP(12,:))/2; (ERP(5,:)+ERP(13,:))/2; (ERP(6,:)+ERP(14,:))/2; (ERP(7,:)+ERP(15,:))/2; (ERP(8,:)+ERP(16,:))/2]; 
col_c1={[0,0,0.2];[0,0,0.5];[0.1,0.1,0.7];[0.1,0.1,1];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.85,0.85,0.85];[0,0,0]};
linestyle={'-';'-';'-';'-';'-';'-';'-';':'}; %this will stay the same for all the plots
figure
for i = 1:size(gfp_c1,1)
gfp_c1filt(i,:)=lowpass(gfp_c1(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c1filt(i,ta1:ta2),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);    
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('blue');
    legend({'standard 2','standard 3','standard 4','standard 5','standard1_2','standard1_3','standard1_4','deviant'});

%plot 2
%average across dark and light green
figure
gfp_c2=[(ERP2(1,:)+ERP2(9,:))/2; (ERP2(2,:)+ERP2(10,:))/2; (ERP2(3,:)+ERP2(11,:))/2; (ERP2(4,:)+ERP2(12,:))/2; (ERP2(5,:)+ERP2(13,:))/2; (ERP2(6,:)+ERP2(14,:))/2; (ERP2(7,:)+ERP2(15,:))/2; (ERP2(8,:)+ERP2(16,:))/2]; 
col_c1={[0,0.2,0];[0,0.5,0];[0.1,0.8,0.1];[0.1,1,0.1];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.85,0.85,0.85];[0,0,0]};
linestyle={'-';'-';'-';'-';'-';'-';'-';':'}; %this will stay the same for all the plots
figure
for i = 1:size(gfp_c1,1)
gfp_c2filt(i,:)=lowpass(gfp_c2(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c2filt(i,ta1:ta2),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);    
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('green');
    legend({'standard 2','standard 3','standard 4','standard 5','standard1_2','standard1_3','standard1_4','deviant'});
    
%plot 3
%average across red/pink
gfp_c3=[(ERP3(1,:)+ERP3(9,:))/2; (ERP3(2,:)+ERP3(10,:))/2; (ERP3(3,:)+ERP3(11,:))/2; (ERP3(4,:)+ERP3(12,:))/2; (ERP3(5,:)+ERP3(13,:))/2; (ERP3(6,:)+ERP3(14,:))/2; (ERP3(7,:)+ERP3(15,:))/2; (ERP3(8,:)+ERP3(16,:))/2]; 
col_c1={[0.2,0,0];[0.6,0,0];[0.8,0.1,0.1];[1,0.1,0.1];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.85,0.85,0.85];[0,0,0]};
linestyle={'-';'-';'-';'-';'-';'-';'-';':'}; %this will stay the same for all the plots
figure
for i = 1:size(gfp_c1,1)
gfp_c3filt(i,:)=lowpass(gfp_c3(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c3filt(i,ta1:ta2),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);    
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('red pink');
    legend({'standard 2','standard 3','standard 4','standard 5','standard1_2','standard1_3','standard1_4','deviant'});

%plot 4
%average across yellow/brown
gfp_c4=[(ERP4(1,:)+ERP4(9,:))/2; (ERP4(2,:)+ERP4(10,:))/2; (ERP4(3,:)+ERP4(11,:))/2; (ERP4(4,:)+ERP4(12,:))/2; (ERP4(5,:)+ERP4(13,:))/2; (ERP4(6,:)+ERP4(14,:))/2; (ERP4(7,:)+ERP4(15,:))/2; (ERP4(8,:)+ERP4(16,:))/2]; 
col_c1={[0.2,0.2,0];[0.6,0.6,0];[0.8,0.8,0.1];[1,1,0.1];[0.3,0.3,0.3];[0.6,0.6,0.6];[0.85,0.85,0.85];[0,0,0]};
linestyle={'-';'-';'-';'-';'-';'-';'-';':'}; %this will stay the same for all the plots
figure
for i = 1:size(gfp_c1,1)
gfp_c4filt(i,:)=lowpass(gfp_c4(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c4filt(i,ta1:ta2),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);    
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('yellow brown');
    legend({'standard 2','standard 3','standard 4','standard 5','standard1_2','standard1_3','standard1_4','deviant'});
    
    
    %plot 2- difference waves for blue
    figure
diff(1,:)=gfp_c1(5,:)-gfp_c1(4,:); %1st standard (after deviant non-target) minus 5th standard - shows adaptation 
diff(2,:)= gfp_c1(8,:) - (gfp_c1(1,:)+gfp_c1(2,:)+gfp_c1(3,:)+gfp_c1(4,:))/4; %deviant minus average of 2nd-5th standards, shows MMN
diff(3,:)= gfp_c1(8,:) - gfp_c1(5,:); %deviant minus 1st non target standard

for i = 1:3   
    plot(ta(ta1:ta2),diff(i,ta1:ta2),'LineWidth',2);
 %   ylim([-4 2])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('difference waves');
    legend({'adaptation','MMN','lightness diff'});
        
        %now make difference waves; deviant minus standard but for warm vs
        %cold
    figure
diff(1,:)=(gfp_c1(5,:)+gfp_c2(5,:))/2 - (gfp_c1(4,:) + gfp_c2(4,:))/2; %1st standard (after deviant non-target) minus 5th standard - shows adaptation 
diff(2,:)= (gfp_c1(8,:)+gfp_c2(8,:))/2 - (gfp_c1(1,:)+gfp_c1(2,:)+gfp_c1(3,:)+gfp_c1(4,:)+gfp_c2(1,:)+gfp_c2(2,:)+gfp_c2(3,:)+gfp_c2(4,:))/8; %deviant minus average of 2nd-5th standards, shows MMN
diff(3,:)= (gfp_c1(8,:)+gfp_c2(8,:))/2 - (gfp_c1(5,:)+gfp_c2(5,:))/2; %deviant minus 1st non target standard - shows effect of feature change
%now do the same for warm cols
diff(4,:)=(gfp_c3(5,:)+gfp_c4(5,:))/2 - (gfp_c3(4,:) + gfp_c4(4,:))/2; %1st standard (after deviant non-target) minus 5th standard - shows adaptation 
diff(5,:)= (gfp_c3(8,:)+gfp_c4(8,:))/2 - (gfp_c3(1,:)+gfp_c3(2,:)+gfp_c3(3,:)+gfp_c3(4,:)+gfp_c4(1,:)+gfp_c4(2,:)+gfp_c4(3,:)+gfp_c4(4,:))/8; %deviant minus average of 2nd-5th standards, shows MMN
diff(6,:)= (gfp_c3(8,:)+gfp_c4(8,:))/2 - (gfp_c3(5,:)+gfp_c4(5,:))/2; %deviant minus 1st non target standard

col_c1={[0.2,0.2,0.2];[0.5,0.5,0.5];[0.8,0.8,0.8];[0.4,0,0];[0.65,0.1,0.1];[1,0.8,0.8]};

for i = 1:size(diff,1)    
difffilt(i,:)=lowpass(diff(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),difffilt(i,ta1:ta2),'color',col_c1{i},'LineWidth',2);
 %   ylim([-4 2])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('difference waves');
    legend({'adaptation','MMN','lightness diff','adaptation','MMN','lightness diff'});

