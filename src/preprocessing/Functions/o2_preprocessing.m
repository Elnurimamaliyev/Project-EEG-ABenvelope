function [EEG,PATH] = o2_preprocessing(s,k,sbj,lp)
%% Does the preprocessing for project
%
%Input:
%s = subject number
%k = condition number
%sbj = array of participant indicies
%lp = lowpass frequency range cutoff
%
%
%Output:
%EEG = EEG structure (completely preprocessed)
%
%
%Notes:
%requires all the other previous pre-processing scripts to have run

%set up the path to the data
% o0_setupscript_trf.
OT_setup

%get the path to the file 
PATH = fullfile([DATAPATH], sprintf(sbj{s}),filesep);
% PATH = fullfile([DATAPATH '\data\'], sprintf(sbj{s}),filesep);

file = {sprintf('%s_ica_narrow_game_added_trigger.set',sbj{s}),sprintf('%s_ica_wide_game_added_trigger.set',sbj{s})};

% file = {sprintf('%s_ica_wide_game_added_trigger.set',sbj{s})};
    
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab('nogui'); %we do not want to have no GUI here -> i am running it over many participants
%load the ica set
EEG = pop_loadset(fullfile(PATH,file{k}));

%reject the labeled components that fall into this range of probability
EEG = pop_icflag(EEG, [NaN NaN;0.7 1;0.7 1;0.6 1;0.7 1;0.7 1;NaN NaN]);
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
EEG = pop_subcomp(EEG, [])
[ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);


%% start the preprocessing

EEG = pop_firws(EEG, 'fcutoff', lp, 'ftype', 'lowpass', 'wtype', 'rectangular', 'forder',100)%,'plotfresp',1);  %low pass filter. Freq. response can be plotted
EEG = pop_resample(EEG,100);                                                                                    %downsample to save time
EEG = pop_firws(EEG, 'fcutoff', 0.3, 'ftype', 'highpass', 'wtype', 'hann',...                                   % highpass filter
    'forder', 518)%,'plotfresp',1);
EEG = pop_interp(EEG,EEG.urchanlocs,'spherical');                                                               %interpolate missing channels
EEG = pop_reref(EEG,{'TP9','TP10'},'keepref','on');                                                             %re-reference to mastoids
EEG = pop_select(EEG,'nochannel',{'TP9','TP10'});                                                               % remove the mastoids channels -> due to reference, they do not hold meaningful information 

