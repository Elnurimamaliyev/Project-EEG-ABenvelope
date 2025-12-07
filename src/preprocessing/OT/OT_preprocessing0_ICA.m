%% Preprocessing - ICA weights
%this script merely computes the ICA weights -> no pre-processing to our
%actual data happens here
%   - pre-processing to obtain the ICA weights is different from the
%   pre-processing of the data the you want to analyze later
%Source: https://sccn.ucsd.edu/wiki/Makoto's_preprocessing_pipeline

%% setup your paths
OT_setup

%% the loop was only done to derive the ICA weights
for s=1:length(sbj)
    sb = sbj{s};
    %setup path file
    PATH = fullfile(DATAPATH, sb,filesep); %is not in the setup script, since it changes dynamically with the participants

    % PATH = fullfile(DATAPATH, sb,filesep);
    cd(PATH)
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab('nogui');
    
    % file = {'wide_game_added_trigger.set'};  %set up name
    file = {'narrow_game_added_trigger.set','wide_game_added_trigger.set'};  %set up name

    badchan = [];

    for i=1:(length(file)) %loop over the two different conditions
        EEG = pop_loadset(fullfile(PATH,file{i}));  %load the set of each condition
        if i ==1
            urchanlocs = EEG.chanlocs;
        end
        
        ALLEEG{i} = EEG;
    end
    
    %merge the data set -> Would you expect artifact locations to differ
    %between conditions? I dont, so i compute the ICA weights over both
    %conditions -> any differences in the conditions is not to different
    %artifacts being removed

    if length(ALLEEG) > 1  % Check if there is more than one dataset
    EEG = pop_mergeset(ALLEEG{1}, ALLEEG{2});  % Merge the first two datasets
    else
        % No action is needed if there is only one dataset in ALLEEG
        disp('Only one dataset available, skipping merge.');
    end


    %%% start the pre-processing
    %downsample
    EEG.data = double(EEG.data); %ICA requires double-precision
    EEG = pop_resample(EEG,250); %resampling is done for computational efficiency -> more samples, more time it takes for the algorithm 
    EEG.data = double(EEG.data); %resampling removed the double precision, so we have to add it again 
    
    %highpass-filter
    EEG = pop_firws(EEG, 'fcutoff', 1, 'ftype', 'highpass', 'wtype', 'hann',...
        'forder', 568);
    
    %lowpass-filter
    EEG = pop_firws(EEG, 'fcutoff', 42, 'ftype', 'lowpass', 'wtype', 'hann',...
        'forder', 128);
    
    %clean the channels 
    EEG = clean_channels(EEG);
    
    if isfield(EEG.chaninfo,'removedchans')
        badchan = EEG.chaninfo.removedchans;
        chan_rej(s,1) = size(badchan,2);
    else
        chan_rej(s,1) = 0;
        
    end
    
%     %reference to average
%     EEG = pop_reref(EEG,[]);
   
    
    %run ICA
    EEG = eeg_regepochs(EEG, 'recurrence', 1);
    EEG = eeg_checkset(EEG, 'eventconsistency');
    EEG.data = double(EEG.data);
    
    % remove epochs with artefacts to improve ICA training
    PRUNE = 3;
    EEG = pop_jointprob(EEG, 1, [1:size(EEG.data,1)], PRUNE, PRUNE, 0, 1, 0);
    EEG = eeg_checkset(EEG, 'eventconsistency');
    EEG.data = double(EEG.data);

    % compute ICA weights
    EEG = pop_runica(EEG, 'icatype', 'runica','extended',1,'interrupt','on','concatenate','on');
    
    %save the ICA weights
    icawinv = EEG.icawinv;
    icasphere = EEG.icasphere;
    icaweights = EEG.icaweights;
    icachansind = EEG.icachansind;
    
    %%% Weight adding
    %previously we ran the ICA and now we have a bunch of weights that need to be save
    %to the actual data of interest
    clear global 
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab('nogui');

    %load the two data sets and add the ICA weights
    for i=1:(length(file)) %loop over the two different conditions
        EEG = pop_loadset(fullfile(PATH,file{i}));

        %since the ICA weights were trained on a subset of channels, if we
        %want to add the weights, we first need to remove the bad channels
        %from earlier again

        if ~isempty(badchan)
            badch = {badchan.labels};
            EEG = pop_select(EEG,'nochannel',badch);
        end

        %sanity check
        EEG = eeg_checkset(EEG, 'eventconsistency');
        EEG.urchanlocs = urchanlocs;
        
        %add the weights to your EEGset
        EEG.icawinv = icawinv;
        EEG.icasphere = icasphere;
        EEG.icaweights = icaweights;
        EEG.icachansind = icachansind;

        %sanity check
        EEG = eeg_checkset( EEG );
        %labels your IC components -> brain, eye, muscle, heart etc.
        EEG = pop_iclabel(EEG, 'default');

        %save the new set with the corresponding ICA labels
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
        pop_saveset(EEG,'filename',sprintf('%s_ica_%s',sbj{s},file{i}),'filepath',PATH);
    end
    clear badchan badch
end