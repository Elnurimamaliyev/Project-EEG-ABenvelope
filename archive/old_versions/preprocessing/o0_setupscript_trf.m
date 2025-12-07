%% SETUP SCRIPT 
% Important Script to add all the important dependencies (toolboxes) and
% paths -> anything that affects all your scripts, should be added here i.e.
% dont add the data path to all of your analysis scripts later on

%% Paths
% Data and functions
% fig_path = char('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Figures\');       % folder to save all my figures later

MAINPATH  = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope';                     % main path to the project folders 
DATAPATH  = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\UserFirst4Test';                % where the files will be saved
chan_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\UserFirst4Test\Data\chan_info\';  % location of channel information 
raw_data_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\UserFirst4Test\Data\raw_data\'; % where the raw data is stored
task = {'narrow'}; 

% Toolboxes
EEGLAB = 'C:\Users\icbmadmin\Documents\GitLabRep\EEGLAB\eeglab2024.1';     % where EEGLAB is saved on my computer
MTRF   = 'C:\Users\icbmadmin\Documents\GitLabRep\mTRF-Toolbox';            % where mTRF-Toolbox is saved on my computer

%% Adding file directories into MATLAB 
addpath(genpath(MAINPATH)); % Matlab can now access where the data has been saved
addpath(genpath(EEGLAB));   % EEGLAB has been now added to Matlabs path and function can be accessed
addpath(genpath(MTRF));     % MTRF has been now added to Matlabs path and function can be accessed
%% Subject specific prefix
sbj =    {'P001'};%, 'P002','P003','P005', 'P006','P007','P008','P009',...
    %'P010','P012','P013', 'P014','P015','P016','P017', 'P018',...
    %'P019','P020','P021','P022'};
    %i have only provided the data set for the first participant, the
    %scripts are designed for all of them though, so dont be confused why
    %there are for loops over all sbj

%% Miscalleneous
TriggerSoundDelay = 0.019; %the delay between presentation and recording of the system
