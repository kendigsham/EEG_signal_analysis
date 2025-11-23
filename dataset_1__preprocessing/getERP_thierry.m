%ERP plots

subjects = {'gm'};
conds = {'11';'12';'21';'22';'31';'32';'41';'42';'51';'52';'61';'62';'71';'72';'81';'82'};

electrodes=[25 27 28 29 62 64]; %same as in thierry plots, leaving out p010 and po9
%electrodes=[25 26 30 63 62 27 29 64 28];%all PO,O,Iz

%y axis limits for plotting
ylim_low=-3;
ylim_high=6;
%samples to plot
ta1=26;
ta2=206; 
%lowpass flitering to use, in Hz
filtering=8;
%some other parameters
nopart=23;
samples=256;

%Component to analyze (in samples)
comp=[94:112] %we need time of MMN - in paper it is between 162 and 232 ms, i.e. the early part of MMN
base=[25:52] %-100 to 0 baseline

%now we want to write a file, in which we copy relevant information
%triplet and colour
filepath = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\data\'];
fid2=fopen(sprintf('%sRstats_vMMN.txt',filepath),'w');

fprintf(fid2,'%s\t %s\t %s\t %s\t %s\t %s\n','participant','Lightness','Colour','Deviancy','Condition no','amplitude');

labels = {'blue','blue','green','green','red','red','yellow','yellow'};

ERPall = zeros(size(conds,1),samples,nopart);

res_mat=[];

for nsub=1:size(subjects,1)
    for ncond=1:size(conds,1)

        FilePath = filepath;
        
%EEG = pop_loadset([ subjects{nsub} '_C' conds{ncond} '.set'], FilePath);
EEG = pop_loadset([ subjects{1} '_C' conds{ncond} '.set'], FilePath);

%for plotting
ERP(ncond,:) = squeeze(mean(mean(EEG.data(electrodes,:,:)),3)); 
%get 95% CI
ERPSD(ncond,:) = 1.96*std(squeeze(mean(EEG.data(electrodes,:,:),1)),0,2)/sqrt(size(EEG.data,3));
%ERPall(ncond,:,:) = squeeze(mean(EEG.data(electrodes,:,:),1)); 
gfp(ncond,:) = std(mean(EEG.data(1:64,:,:),3),0,1); % GFP of the mean across observations

%for stats
       dummy = EEG.data(electrodes,comp,:); 
       basedummy = EEG.data(electrodes,base,:);
       dummy=squeeze(mean(mean(dummy,2),1)) - squeeze(mean(mean(basedummy,2),1));

%for baseline-subtracted plot
ERPbase(ncond,:) = mean((squeeze(mean(EEG.data(electrodes,:,:))) - repmat(squeeze(mean(mean(basedummy,2),1))',256,1)),2)'; 
       
 %strip the colour of the condition from the deviancy
cstr=char(conds(ncond));
colour=str2double(cstr(1)); deviancy=str2double(cstr(2));

       if rem(deviancy, 2) ~= 0 %odd numbers
    sat="standard";
elseif rem(deviancy, 2) == 0 %even numbers
    sat="deviant";
       end

  if rem(colour, 2) ~= 0 %odd numbers
   lightness="dark";
elseif rem(colour, 2) == 0 %even numbers
    lightness="light";
            end

            for np=1:length(dummy)
            fprintf(fid2,'%i\t %s\t %s\t %s\t %s\t %6f\n',np,lightness,labels{colour},sat,cstr,dummy(np));
            end

    end
end

%get further parameters from EEG file
ta=EEG.times; 
srate=EEG.srate;

%choose whether to plot baseline-subtracted or not
plotpar=input('1 - plot baseline corrected, 2 - plot uncorrected)');
if plotpar==1
    ERP=ERPbase;
end

%plot 1
gfp_c1=[ERP(1,:); ERP(2,:); ERP(3,:); ERP(4,:)]; %col 1 standard, dark blue
col_c1={[0,0,0.6];[0.6,0.6,0.85];[0.6,0.6,0.85];[0,0,0.6]};
linestyle={':';'-';':';'-'}; %this will stay the same for all the plots
figure
for i = 1:4
gfp_c1(i,:)=lowpass(gfp_c1(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c1(i,ta1:ta2),'color',col_c1{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('blue');
    legend({'dark stand','light dev','light stand','dark dev'});
    xlim([-100 600])

%plot2
gfp_c2=[ERP(5,:); ERP(6,:);ERP(7,:); ERP(8,:)];
col_c2={[0,0.4,0];[0.2,0.7,0.2];[0.2,0.7,0.2];[0,0.4,0]};

figure
for i = 1:4
gfp_c2(i,:)=lowpass(gfp_c2(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c2(i,ta1:ta2),'color',col_c2{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('green');
    legend({'dark stand','light dev','light stand','dark dev'});
xlim([-100 600])

    %plot3
gfp_c3=[ERP(9,:); ERP(10,:); ERP(11,:); ERP(12,:)];
col_c3={[0.5,0,0];[0.9,0.6,0.6];[0.9,0.6,0.6];[0.5,0,0]};
figure
for i = 1:4
gfp_c3(i,:)=lowpass(gfp_c3(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c3(i,ta1:ta2),'color',col_c3{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('S');
    title('red');
    legend({'dark stand','light dev','light stand','dark dev'});
xlim([-100 600])

        %plot4
gfp_c4=[ERP(13,:); ERP(14,:); ERP(15,:); ERP(16,:)];
col_c4={[0.4,0.4,0];[0.7,0.7,0];[0.7,0.7,0];[0.4,0.4,0]};
figure
for i = 1:4
gfp_c4(i,:)=lowpass(gfp_c4(i,:),filtering,srate); %low pass filter for display
    plot(ta(ta1:ta2),gfp_c4(i,ta1:ta2),'color',col_c4{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([ylim_low ylim_high])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('yellow');
    legend({'dark stand','light dev','light stand','dark dev'});
xlim([-100 600])

    %make difference waves collapsed across lightness
figure
diff(1,:)=(gfp_c1(4,:)-gfp_c1(1,:)+ gfp_c1(2,:)-gfp_c1(3,:))/2; %blue
sdev(1,:)=(ERPSD(4,:)-ERPSD(1,:)+ ERPSD(2,:)-ERPSD(3,:))/2; 
diff(2,:)=(gfp_c2(4,:)-gfp_c2(1,:)+ gfp_c2(2,:)-gfp_c2(3,:))/2; %g
sdev(2,:)=(ERPSD(8,:)-ERPSD(5,:)+ ERPSD(6,:)-ERPSD(7,:))/2; 
diff(3,:)=(gfp_c3(4,:)-gfp_c3(1,:)+ gfp_c3(2,:)-gfp_c3(3,:))/2; %r
sdev(3,:)=(ERPSD(12,:)-ERPSD(9,:)+ ERPSD(10,:)-ERPSD(11,:))/2; 
diff(4,:)=(gfp_c4(4,:)-gfp_c4(1,:)+ gfp_c4(2,:)-gfp_c4(3,:))/2; %y
sdev(4,:)=(ERPSD(16,:)-ERPSD(13,:)+ ERPSD(14,:)-ERPSD(15,:))/2; 

cols={[0,0,1];[0,1,0];[1,0,0];[0.5,0.5,0]};
%first plot all the SD intervals
for in = 1:4 
        patch1 = fill([ta(ta1:ta2),fliplr(ta(ta1:ta2))],[diff(in,ta1:ta2)+squeeze(sdev(in,ta1:ta2)),fliplr(diff(in,ta1:ta2)-squeeze(sdev(in,ta1:ta2)))],cols{in}); hold on
        set(patch1, 'edgecolor', 'none');
        set(patch1, 'FaceAlpha', 0.2);
end
for i = 1:4 %then plot the means
    plot(ta(ta1:ta2),diff(i,ta1:ta2),'color',cols{i},'LineWidth',1);
    ylim([-2 4])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('difference waves');
    legend({'blue','green','red','yellow'});
xlim([-100 600])

%make differences for cool vs warm - this has same power as in the original
%study
figure
diff3(1,:)=((gfp_c1(4,:)-gfp_c1(1,:)+ gfp_c1(2,:)-gfp_c1(3,:))/2+(gfp_c2(4,:)-gfp_c2(1,:)+ gfp_c2(2,:)-gfp_c2(3,:))/2)/2; %blue and green
sdev3(1,:)=((ERPSD(4,:)-ERPSD(1,:)+ ERPSD(2,:)-ERPSD(3,:))/2 + (ERPSD(8,:)-ERPSD(5,:)+ ERPSD(6,:)-ERPSD(7,:))/2)/2; 
diff3(2,:)=((gfp_c3(4,:)-gfp_c3(1,:)+ gfp_c3(2,:)-gfp_c3(3,:))/2+(gfp_c4(4,:)-gfp_c4(1,:)+ gfp_c4(2,:)-gfp_c4(3,:))/2)/2; %r & y
sdev3(2,:)=((ERPSD(12,:)-ERPSD(9,:)+ ERPSD(10,:)-ERPSD(11,:))/2 + (ERPSD(16,:)-ERPSD(13,:)+ ERPSD(14,:)-ERPSD(15,:))/2)/2; 

cols={[0,0,0];[1,0,0]};
%first plot all the SD intervals
for in = 1:2
        patch1 = fill([ta(ta1:ta2),fliplr(ta(ta1:ta2))],[diff3(in,ta1:ta2)+squeeze(sdev3(in,ta1:ta2)),fliplr(diff3(in,ta1:ta2)-squeeze(sdev3(in,ta1:ta2)))],cols{in}); hold on
        set(patch1, 'edgecolor', 'none');
        set(patch1, 'FaceAlpha', 0.2);
end
for i = 1:2 %then plot the means
    plot(ta(ta1:ta2),diff3(i,ta1:ta2),'color',cols{i},'LineWidth',1);
    ylim([-2 4])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('difference waves');
    legend({'cool','warm'});
xlim([-100 600])

    %make difference waves for dark and light separately - this will be
    %noisy
figure
diff2(1,:)=gfp_c1(4,:)-gfp_c1(1,:); %d blue 
diff2(2,:)=gfp_c1(2,:)-gfp_c1(3,:); %l blue
diff2(3,:)=gfp_c2(4,:)-gfp_c2(1,:); %d green
diff2(4,:)=gfp_c2(2,:)-gfp_c2(3,:); %l green
diff2(5,:)=gfp_c3(4,:)-gfp_c3(1,:); %d red
diff2(6,:)=gfp_c3(2,:)-gfp_c3(3,:); %l red
diff2(7,:)=gfp_c4(4,:)-gfp_c4(1,:); %d yellow
diff2(8,:)=gfp_c4(2,:)-gfp_c4(3,:); %l yellow
cols={[0,0,0.4];[0,0,0.7];[0,0.4,0];[0,0.7,0];[0.4,0,0];[0.7,0,0];[0.4,0.4,0];[0.7,0.7,0]};
linestyle={'--';'-';'--';'-';'--';'-';'--';'-'}; %this will stay the same for all the plots

for i = 1:8
    plot(ta(ta1:ta2),diff2(i,ta1:ta2),'color',cols{i},'LineStyle',linestyle{i},'LineWidth',2);
    ylim([-2 4])
    hold on;
end
    yline([0],'--')
    xline([0],'--')
    title('difference waves');
    legend({'d blue','l blue','d green','l green','d red','l red','d yellow','l yellow'});

