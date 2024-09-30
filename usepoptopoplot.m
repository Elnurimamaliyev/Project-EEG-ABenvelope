%% Add path
OT_setup

addpath(genpath(['C:\Users\icbmadmin\Documents\GitLabRep' ...
    '\EEGLAB\eeglab2024.2\'])); % EEGLAB has been now added to Matlabs path
                                % and function can be accessed

%%                                
[fs_eeg, resp, fs_audio, audio_dat, EEG] = LoadEEG(s, k, sbj, task);

%% pop_topoplot function
A = pop_topoplot(EEG)

%% topoplot function
% topoplot
