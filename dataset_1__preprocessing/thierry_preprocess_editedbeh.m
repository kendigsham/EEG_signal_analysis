%subjects, in order
subjects={'11';'12';'13';'14';'15';'16';'17';'18';'19';'20'};
blocks = {'1';'2';'3';'4';'5';'6';'7';'8'}; 

%you will need to change the filepath to your own onedrive folders
filepath1 = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\'] %this is where we put our data
filepath2 = ['C:\Users\jmartino\OneDrive - University of Edinburgh\Thierry_Data\dataedited\'] %this is processed data

t_minmax = [-0.2 0.8] % Epoch to cut out in a first step
baseline = [-100 0] %in ms, for eeglab baseline subtraction

for ns = 1:size(subjects,1)
    
    for nb = 1:size(blocks,1)
        
%          if subjects{ns}=='8' && blocks{nb}=='5'
%              continue;
%          end    
       
        filepath_load = [filepath1];
        filename = ['p' subjects{ns} '_thierry_bl' blocks{nb}];
           disp('working on:')        
        
        %initialise eeglab
        eeglab

               %for participants 6 and 7, we fix up the files in BVA as edf+ and
        %then import, with only stimulus onsets as events
        if strcmp(subjects{ns},'6') || strcmp(subjects{ns},'7') || strcmp(subjects{ns},'18')
        
                 
        [filepath_load filename '.edf']

            EEG = pop_biosig([filepath_load filename '.edf']);                

        else
            
            [filepath_load filename '.bdf']

        %load .bdf biosemi file
         EEG = pop_biosig([filepath_load filename '.bdf']);
%         %get rid of all offset events, i.e. not type 1 (they're called
%         %64512 so we will only keep 64513)
         eventno=[65025, 64513];
         EEG = pop_selectevent( EEG, 'type',eventno,'deleteevents','on');
        end
          
        %downsample to 256 Hz
        EEG = pop_resample( EEG, 256);
        
         %REPLACING EVENTS
        %load file with behavioral data; first i gotta rename this file
        %and delete practice from it!
        behdata = load ( [filepath1, subjects{ns}, '_edited.result'] );
                
        trials_per_block=450; 
        numblocks=8;
          
        %select the appropriate section of result file for each block
        bl_num=str2num(blocks{nb}); %numerical value for block                
        
        if bl_num==1
            act_triggers=behdata(1:trials_per_block,4); %triggers are in column 4
            act_shape=behdata(1:trials_per_block,3); %shape is column 3 - circle 1,2 and square 3,4
            act_corr=behdata(1:trials_per_block,5); %accuracy is in column 6
        elseif bl_num==8
            act_triggers=behdata(((bl_num-1)*trials_per_block+1):end,4);
            act_shape=behdata(((bl_num-1)*trials_per_block+1):end,3);
            act_corr=behdata(((bl_num-1)*trials_per_block+1):end,5);
        else
             act_triggers=behdata(((bl_num-1)*trials_per_block+1):(bl_num*trials_per_block),4);
             act_shape=behdata(((bl_num-1)*trials_per_block+1):(bl_num*trials_per_block),3);
            act_corr=behdata(((bl_num-1)*trials_per_block+1):(bl_num*trials_per_block),5);
        end
            
             %check that the number of events is correct
        if trials_per_block ~= size(EEG.event,2) 
             disp('incorrect number of events - please check why!');
            %return
            cutoff=trials_per_block-size(EEG.event,2); %how many trials were lost?
            %shorten on this basis, cutting off lost trials
            act_triggers=act_triggers((1+cutoff):end);
            act_shape=act_shape((1+cutoff):end);
            act_corr=act_corr((1+cutoff):end);
        end
 
        %overwrite first two triggers with 99, as in male et al. paper
        act_triggers(1:2)=99;
        
        num_trials=length(act_triggers);

        %if it is incorrect, precede it with 9 so we know to throw it away
        for broj=1:num_trials   
            %check if it is standard and no response
            if act_shape(broj)<3 && act_corr(broj)==9
            EEG.event(1,broj).type = act_triggers(broj); 
            %now check if it is an oddball with response
            elseif  act_shape(broj)>2 && act_corr(broj)==1
            EEG.event(1,broj).type = act_triggers(broj);                 
            else %all other answers become 999 - wrong
            EEG.event(1,broj).type = 999;                                 
            end
        end
        
        %the idea is to epochise the oddballs so we have them for erp analyses but to get rid of them prior to the decoding
        %analysis as they are too few for that
     
%it is best to filter continuous data - plus lowpass won't work otherwise
        
        %need to reference before i do anything cause of biosemi's 40db noise thing
    EEG = pop_reref(EEG,[],'exclude',[65:EEG.nbchan]); % Average Reference
       
%highpass - .1 Hz (-6 dB cutoff) highpass Hamming windowed sinc FIR filter
    EEG = pop_eegfiltnew(EEG, 'locutoff',0.1,'plotfreqz',0);
     EEG = eeg_checkset( EEG );
    %lowpass
     EEG = pop_eegfiltnew(EEG, 'hicutoff',40,'plotfreqz',0);
        
        %---------------------
        %now we can epochise
        %---------------------
        relevant_triggers={'11' '111' '1111' '11111' '9211' '9311' '9411' '12' '13' '14' '21' '221' '2221' '22221' '9221' '9321' '9421' '22' '23' '24' '31' '331' '3331' '33331' '9231' '9331' '9431' '32' '33' '34' '41' '441' '4441' '44441' '9241' '9341' '9441' '42' '43' '44' '51' '551' '5551' '55551' '9251' '9351' '9451' '52' '53' '54' '61' '661' '6661' '66661' '9261' '9361' '9461' '62' '63' '64' '71' '771' '7771' '77771' '9271' '9371' '9471' '72' '73' '74' '81' '881' '8881' '88881' '9281' '9381' '9481' '82' '83' '84' '999'}
        EEG = pop_epoch( EEG, relevant_triggers, t_minmax, 'newname', 'ana epochised', 'epochinfo', 'yes');
        EEG = pop_rmbase( EEG, baseline);
        
        EEG = pop_saveset( EEG, 'filename', [filename '_e.set'] , 'filepath', filepath2);
    end
end


