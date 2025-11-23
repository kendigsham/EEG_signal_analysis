clear all

%subjects, in order
subjects={'11';'12';'13';'14';'15';'16';'17';'18';'19';'20'};

blocks = {'1';'2';'3';'4';'5';'6';'7';'8'}; 

filepath1 = ['D:\Thierry_Data\dataedited\']


t_minmax = [-0.2 0.8] % Epoch to cut out in a first step
baseline = [-100 0] %in ms, for eeglab baseline subtraction

%-----------------------------------------------------------------------
%ARTIFACT REJECTION USING FASTER - but not doing interpolation
%-----------------------------------------------------------------------

%define limits for blink detection
blinkstart=-0.2; blinkend=0.6; eyemovend=0.6;

for ns = 1: size(subjects,1)
       
    %open a log file to write down rejected epochs
    fid5=fopen([filepath1 'reject_' subjects{ns} '.txt'],'a');
    %set up savename for adjust report
    reportname=['adjust_report_' subjects{ns} '.txt']
    
    eeglab
    
    for nb = 1 : size(blocks,1)
        filenames{nb} = {['p' subjects{ns} '_thierry_bl' blocks{nb} '_e.set']};
    end
    filenames
    
    for jj = 1 : size(filenames,2)
        EEG = pop_loadset(filenames{jj}, filepath1);
        [ALLEEG, EEG, jj] = eeg_store( ALLEEG, EEG);
    end
    
    max_blockno=size(blocks,1);
    
    EEG = pop_mergeset( ALLEEG, [1:max_blockno], 0)
    
    EEG = pop_saveset( EEG, [subjects{ns} '_ALL.set'], filepath1);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    
    eeglab
    %
    EEG = pop_loadset( [subjects{ns} '_ALL.set'], filepath1);
    
     
    %remove ext7 and ext8
    EEG = pop_select( EEG, 'channel',[1:70]); %64+6 externals
   
    %load locations
    EEG=pop_chanedit(EEG,  'load',{ 'C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\processing\biosemi64.elp', 'filetype', 'besa'});
    
    %sort out the eye channels locations manually - left mastoid, right
    %mastoid, left hor, right hor, left above, left below
  
    EEG=pop_chanedit(EEG, 'changefield',{69 'theta' '-108'},'changefield',{69 'radius' '0.639'},'changefield',{69 'X' '-0.28'},'changefield',{69 'Y' '0.862'},'changefield',{69 'Z' '-0.423'},'changefield',{69 'sph_theta' '108'},'changefield',{69 'sph_phi' '-25'});
    EEG=pop_chanedit(EEG, 'changefield',{70 'theta' '108'},'changefield',{70 'radius' '0.639'},'changefield',{70 'X' '-0.28'},'changefield',{70 'Y' '-0.862'},'changefield',{70 'Z' '-0.423'},'changefield',{70 'sph_theta' '-108'},'changefield',{70 'sph_phi' '-25'});       
    EEG=pop_chanedit(EEG, 'changefield',{67 'theta' '-42'},'changefield',{67 'radius' '0.65556'},'changefield',{67 'X' '0.65616'},'changefield',{67 'Y' '0.59081'},'changefield',{67 'Z' '-0.46947'},'changefield',{67 'sph_theta' '42'},'changefield',{67 'sph_phi' '-28'});
    EEG=pop_chanedit(EEG, 'changefield',{68 'theta' '42'},'changefield',{68 'radius' '0.65556'},'changefield',{68 'X' '0.65616'},'changefield',{68 'Y' '-0.59081'},'changefield',{68 'Z' '-0.46947'},'changefield',{68 'sph_theta' '-42'},'changefield',{68 'sph_phi' '-28'});
    EEG=pop_chanedit(EEG, 'changefield',{65 'theta' '-25'},'changefield',{65 'radius' '0.58333'},'changefield',{65 'X' '0.87543'},'changefield',{65 'Y' '0.40822'},'changefield',{65 'Z' '-0.25882'},'changefield',{65 'sph_theta' '25'},'changefield',{65 'sph_phi' '-15'});
    EEG=pop_chanedit(EEG, 'changefield',{66 'theta' '-27'},'changefield',{66 'radius' '0.69444'},'changefield',{66 'X' '0.72987'},'changefield',{66 'Y' '0.37189'},'changefield',{66 'Z' '-0.57358'},'changefield',{66 'sph_theta' '27'},'changefield',{66 'sph_phi' '-35'});
    
    %reset to Fz reference to run FASTER
    EEG = pop_reref( EEG, [38],'keepref','on','exclude',[65:EEG.nbchan] );
    
    %run FASTER
    %----------
    %take out the 32 channels from EEG data to feed into faster
    backupEEG=EEG.data; %store data in a variable as a backup
    backupNumchan=EEG.nbchan;
    backuplocs=EEG.chanlocs;
    %now alter the variables to only have the 64 head channels
    EEG.data=EEG.data(1:64,:,:);
    EEG.nbchan=64;
    EEG.chanlocs=EEG.chanlocs(1:64);
    
    %find bad epochs to reject
    epoch_props = epoch_properties(EEG,1:64);
    epochs_threshold = min_z(epoch_props); % min_z also takes further arguments to allow you to turn on or off specific testing properties or use a Z-score threshold other than 3
    bad_epochs = find(epochs_threshold) % Exceeded threshold also details how many properties of each channel, epoch, component, etc exceeded the threshold, so you could use "bad_X = find(exceeded_threshold >= 2)" to tighten the rejection conditions to only reject Xs that are considered artifact-contaminated by 2 tests, if desired.
    
    %take stock of all the rejected epochs
    fprintf(fid5,'%s\t %i\n','bad_epochs',length(bad_epochs));
    
    
    %now restructure my EEG data to also include external channels
    EEG.data=[EEG.data; backupEEG(65:backupNumchan,:,:)];
    EEG.nbchan=backupNumchan;
    EEG.chanlocs=backuplocs;
    
    %reject the bad epochs
    EEG = pop_rejepoch( EEG, bad_epochs,0);
    
    %reset to average reference
    EEG = pop_reref(EEG,[],'refstate',0,'exclude',[65:EEG.nbchan]); % Average Reference
    
    
    %now do ICA
    %-------------------
EEG = pop_runica(EEG, 'icatype','runica','dataset',1,'options',{'extended' 1},'chanind',[1:70] );

    %automatically detect artifactual components
    [art, horiz, vert, blink, disc,soglia_DV, diff_var, soglia_K, meanK, soglia_SED, SED, soglia_SAD, SAD, ...
        soglia_GDSF, GDSF, soglia_V, nuovaV]=ADJUST(EEG,reportname);
    
    %now put it back to how it was
        EEG.chanlocs=backuplocs;
    
    %EEG = interface_ADJ (EEG,'report');
    
    %take record of these components
    fprintf(fid5,'%s\t %i\n','HEM_components',length(horiz));
    fprintf(fid5,'%s\t %i\n','VEM_components',length(vert));
    fprintf(fid5,'%s\t %i\n','blink_components',length(blink));
    fprintf(fid5,'%s\t %i\n','discontinuity_components',length(disc));
    
    %subtract them from the data
    EEG = pop_subcomp( EEG, art, 0);
    
   
%     EEG=pop_select(EEG,'notrial',Trials2Remove1);
%     EEG=pop_select(EEG,'notrial',Trials2Remove2);
%     %take stock of all the rejected epochs
%     fprintf(fid5,'%s\t %i\n','blinks',length(Trials2Remove1));
%     %take stock of all the rejected epochs
%     fprintf(fid5,'%s\t %i\n','eye movs',length(Trials2Remove2));
    
    
    EEG = pop_saveset( EEG, [subjects{ns} '_ALL_rejected.set'], filepath1);
    [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );
    
    fclose(fid5);
end %subjects

