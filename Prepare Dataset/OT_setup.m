%setup file
% Add mTRF Toolbox Pathway
addpath(genpath('C:\Users\icbmadmin\Documents\GitLabRep\EEGLAB\eeglab2024.1\')); % EEGLAB has been now added to Matlabs path and function can be accessed
addpath(genpath('C:\Users\icbmadmin\Documents\GitLabRep\mTRF-Toolbox\'));        % MTRF has been now added to Matlabs path and function can be accessed

% Add Data Pathway
DATAPATH = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AllUsers\';

addpath(genpath(DATAPATH));

MAINPATH = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\';
raw_data_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AllUsers\raw_data\';
chan_path = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AllUsers\chan_info\';
fig_path = char('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\Figs\');

task = {'narrow', 'wide'};
sbj =    {'P002','P003','P005', 'P006','P007','P008','P009',...
    'P010','P012','P013', 'P014','P015','P016','P017', 'P018',...
    'P019','P020','P021','P022'};

% sbj =    {'P001', 'P002','P003','P005', 'P006','P007','P008','P009',...
%     'P010','P012','P013', 'P014','P015','P016','P017', 'P018',...
%     'P019','P020','P021','P022'};

%the delay between presentation and recording of the system
TriggerSoundDelay = 0.019;

%%
%%% Original by Thorge - Commented
% DATAPATH = 'C:\data files';
% addpath(genpath(DATAPATH));
% MAINPATH = 'O:\projects\thh_ont\auditory-attention-in-complex-work-related-auditory-envrionments';
% marcmainpath = 'O:\projects\mar_game\';
% raw_data_path = ['O:\projects\mar_game\raw_data\'];
% chan_path = 'O:\projects\thh_ont\auditory-attention-in-complex-work-related-auditory-envrionments\configs\';
% fig_path = char('\\smb.uni-oldenburg.de\home\lorf0331\Documents\MATLAB\Project\OTtracking\Results\figs');
% task = {'narrow', 'wide'};
% 
% sbj =    {'P001', 'P002','P003','P005', 'P006','P007','P008','P009',...
%     'P010','P012','P013', 'P014','P015','P016','P017', 'P018',...
%     'P019','P020','P021','P022'};
% 
% %the delay between presentation and recording of the system
% TriggerSoundDelay = 0.019;
