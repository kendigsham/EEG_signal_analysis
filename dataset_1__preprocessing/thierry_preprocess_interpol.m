clear all

subjects={'27';'28';'29';'32'};

filepath = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\data\'] %this is where we put our data

baseline = [-100 0] %in ms, for eeglab baseline subtraction

for ns = 1: size(subjects,1)
            
            %open a log file to write down rejected epochs
            fid5=fopen([filepath 'reject_' subjects{ns} '_INTERPOL.txt'],'a');

            eeglab

                            EEG = pop_loadset([subjects{ns} '_ALL_rejected.set'], filepath);

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
    
%find bad channels to reject
    channel_props = channel_properties(EEG,1:64,38);
    channel_threshold = min_z(channel_props); % min_z also takes further arguments to allow you to turn on or off specific testing properties or use a Z-score threshold other than 3
    bad_chans = find(channel_threshold) % Exceeded threshold also details how many properties of each channel, epoch, component, etc exceeded the threshold, so you could use "bad_X = find(exceeded_threshold >= 2)" to tighten the rejection conditions to only reject Xs that are considered artifact-contaminated by 2 tests, if desired.
    
    %take stock of all the rejected epochs
    fprintf(fid5,'%s\t %i\n','bad_chans',length(bad_chans));
    
    %interpolate bad channels
    if isempty(bad_chans)==0
        EEG = h_eeg_interp_spl(EEG, bad_chans);
    end

    
            %find epochs that need some channels interpolated
    for ep=1:length(EEG.epoch)
    epochchan_properties = single_epoch_channel_properties(EEG,ep,1:64);
    epochchannel_threshold = min_z(epochchan_properties); % min_z also takes further arguments to allow you to turn on or off specific testing properties or use a Z-score threshold other than 3

%    subjects{ns}
    epochbad_chans = find(epochchannel_threshold) % Exceeded threshold also details how many properties of each channel, epoch, component, etc exceeded the threshold, so you could use "bad_X = find(exceeded_threshold >= 2)" to tighten the rejection conditions to only reject Xs that are considered artifact-contaminated by 2 tests, if desired.
    
    %take stock of all the rejected epochs
    fprintf(fid5,'%s\t epoch %i\t %i\n','badchan_epochs',ep,length(epochbad_chans));
    
    if isempty(epochbad_chans)==0
        %make it into a cell for interpolation purposes
        badcell=num2cell(epochbad_chans);
        %interpolate bad channels in these epochs
        EEG = h_epoch_interp_spl(EEG,badcell);
    end
    end

        %now restructure my EEG data to also include external channels
    EEG.nbchan=backupNumchan;
        EEG.data=[EEG.data; backupEEG(65:EEG.nbchan,:,:)];
    EEG.chanlocs=backuplocs;

                %reset to average reference WITHout EYE CHANNELS - for
                %further analysis
            EEG = pop_reref(EEG,[],'refstate',0,'exclude',[65:EEG.nbchan]); % Average Reference
        
%                    %reject trials which were found by ben to be poor after
% %                    %visual inspection
%                     if subjects{ns}=='400'
% EEG = pop_rejepoch( EEG, [28	30	33	54	74	75	98	116	153	168	216	236	270	293	308	314	328	380	381	398	449	451	470	483	484	488	493	511	516	518	523	525	545	546	547	551	552	554	564	570	575	587	596	612	621	626	632	645	649	671	684	687	716	751	769	804	825	846	857	863	889	991	914	919	920	933	947	948	974	988	996] ,0);
%                      %fuzzy noise residue after ica rejections
%                     elseif subjects{ns}=='401'
%     EEG = pop_rejepoch( EEG, [17	18	19	29	57	73	76	98	102	103	107	228	314	315	316	399	428	486	512	572	633	639	667	696	772	788	823	946] ,0);
%                    elseif subjects{ns}=='402'
%    EEG = pop_rejepoch( EEG, [69	70	86	87	90	119	209	333	419	447	505	654	721	737	803	912	963] ,0);
                   
                   EEG = pop_rmbase( EEG, baseline);        
                   
            EEG = pop_saveset( EEG, [subjects{ns} '_ALL70_interpol.set'], filepath);
            [ALLEEG, EEG, CURRENTSET] = eeg_store( ALLEEG, EEG, 0 );

        fclose(fid5);
                            
end %subjects

    %NOTE!!!!!!!!!!!!!!!!!
%now you need to load each of the files and check if the artifact rejection
%went alright by eye! make a note of the number of extra trials you reject
%so you can calculate the percentage of rejected trials for each
