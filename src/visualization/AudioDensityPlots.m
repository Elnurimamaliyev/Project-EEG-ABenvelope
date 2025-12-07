% Audio Density Plots
%% Cleaning
clear; clc;
% Add Main paths
OT_setup
%% Setup: Initialize Matrices for Amplitude Counts and PSDs
numBins = 8;      % number of bins for amplitude histogram

% Initialize matrices for amplitude histogram counts and PSD storage
Narrow_audio_counts = zeros(length(sbj), numBins);
Wide_audio_counts = zeros(length(sbj), numBins);
psd_narrow_audio = cell(length(sbj), 1);  % PSD for Narrow audio stimulus
psd_wide_audio = cell(length(sbj), 1);    % PSD for Wide audio stimulus
psd_narrow_envelope = cell(length(sbj), 1); % PSD for Narrow envelope
psd_wide_envelope = cell(length(sbj), 1);   % PSD for Wide envelope

%% Loop through Each Subject and Task
for s = 1:length(sbj)
    for k = 1:length(task)
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Display subject and task info
        
        % Load Audio Data
        [fs_eeg, ~, fs_audio, audio_dat, ~] = LoadEEG(s, k, sbj, task);
        % [p, q] = rat(fs_eeg / fs_audio);
        % audio_resampled = resample(audio_dat, p, q);

        % Create Envelope using mTRF Toolbox
        envelope = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Envelope extraction
        EnvNorm = (envelope - min(envelope)) / (max(envelope) - min(envelope)); % Normalize envelope
        
        % Define Bin Edges for Histogram
        binEdges_dB = linspace(0, 0.98, numBins + 1);
        binEdges_amp = 10.^(binEdges_dB / 20); % Convert to amplitude
        normbinEdges_amp = normalize(binEdges_amp, 2, 'range');
        
        % Calculate Histogram Counts
        [DbCounts, ~, ~] = histcounts(EnvNorm, normbinEdges_amp);

        % Assign histogram counts based on task type
        if k == 1 % Narrow audio
            Narrow_audio_counts(s, :) = DbCounts;
        elseif k == 2 % Wide audio
            Wide_audio_counts(s, :) = DbCounts;
        end

        % Calculate and Store Power Spectral Densities (PSD) for Raw Audio
        % [pxx_audio, f_audio] = pwelch(audio_resampled, [], [], [], fs_eeg);
        [pxx_audio, f_audio] = pwelch(audio_dat, [], [], [], fs_audio);

        if k == 1
            psd_narrow_audio{s} = {f_audio, pxx_audio}; % PSD for Narrow audio
        elseif k == 2
            psd_wide_audio{s} = {f_audio, pxx_audio};   % PSD for Wide audio
        end

        % Calculate and Store PSD for Envelope
        [pxx_env, f_env] = pwelch(envelope, [], [], [], fs_eeg);
        if k == 1
            psd_narrow_envelope{s} = {f_env, pxx_env}; % PSD for Narrow envelope
        elseif k == 2
            psd_wide_envelope{s} = {f_env, pxx_env};   % PSD for Wide envelope
        end
    end
end

NormAudiodat = normalize(audio_dat,2,"range",[-1 1]);
%% 
% Group Averages of each condition for Histogram Counts
MeanNarrowAudio = mean(Narrow_audio_counts, 1);
MeanWideAudio = mean(Wide_audio_counts, 1);
% Average PSDs across subjects
avg_psd_narrow_audio = mean(cell2mat(cellfun(@(x) x{2}', psd_narrow_audio, 'UniformOutput', false)), 1);
avg_psd_wide_audio = mean(cell2mat(cellfun(@(x) x{2}', psd_wide_audio, 'UniformOutput', false)), 1);
avg_psd_narrow_envelope = mean(cell2mat(cellfun(@(x) x{2}', psd_narrow_envelope, 'UniformOutput', false)), 1);
avg_psd_wide_envelope = mean(cell2mat(cellfun(@(x) x{2}', psd_wide_envelope, 'UniformOutput', false)), 1);

% Group and Condition Averages for Histogram Counts
AvgAudioDist = (MeanNarrowAudio + MeanWideAudio) / 2;
% Group and Condition Averages for PSD
AvgPSDaudio = (avg_psd_narrow_audio + avg_psd_wide_audio)/2;
AvgPSDenvelope = (avg_psd_narrow_envelope + avg_psd_wide_envelope)/2;

%%
% Save
AudioDescription.norm_audio_dat = NormAudiodat;
AudioDescription.AvgAudioDist = AvgAudioDist;
AudioDescription.AvgPSDaudio = AvgPSDaudio;
AudioDescription.AvgPSDenvelope = AvgPSDenvelope;

AudioDescription_friday= AudioDescription;
% Save the updated norm_audio_dat variable to a .mat file
save('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\AudioDescription_friday.mat', 'AudioDescription_friday');
%% Plotting
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio

% Extra parameters
window_time = [1149 1151];
time_indices = (time_audio >= window_time(1)) & (time_audio <= window_time(2));
time_windowed = time_audio(time_indices);
NormAudiodat_windowed = NormAudiodat(time_indices);
%%
% Colors
% blue = [0 0.4470 0.7410];
% orange = [0.8500 0.3250 0.0980];
% yellow = [0.9290 0.6940 0.1250];

% 3500, 2802, 2401, 1441,  

% Plot 1: Amplitude Histogram for Group & Condition Average

% figure;
subplot(2,2,1)
plot(time_windowed, NormAudiodat_windowed, 'LineWidth', 0.8, 'DisplayName', 'Original Audio (normalized)');  % Original audio with thicker line
% plot(time_audio, NormAudiodat, 'LineWidth', 1, 'DisplayName', 'Original Audio (normalized)');  % Original audio with thicker line
xlabel('Time (s)'); ylabel('Amplitude Bins (Normalized)'); xlim(window_time);
title('Sound Sample');
% grid on; 
ylim([-1, 1]);
hold off; box off
% legend('Location', 'best');  % Add legend

subplot(2, 2, 2);
histogram('BinEdges', normbinEdges_amp, 'BinCounts', AvgAudioDist);
xlabel('Amplitude Bins (Normalized)');
ylabel('Counts');
title('Envelope Amplitude Distribution');
set(gca, 'View', [90 -90]); % Rotate plot
% grid on; 
box off;

% Plot 2: Power Spectral Densities (PSDs) for Audio Stimuli and Envelopes

% Subplot 3: PSD of Narrow Audio Stimulus
subplot(2, 2, 3);
plot(psd_narrow_audio{1}{1}, AvgPSDaudio,  'LineWidth', 0.8);
xlabel('Frequency (Hz)');
ylabel('PSD (a.u.^2/Hz)');
title('PSD of Audio Stimulus');
% grid on;
ylim([0, 0.0000004]);
xlim([0, 6000]);

% Subplot 4: PSD of Wide Audio Envelope
subplot(2, 2, 4);
plot(psd_wide_envelope{1}{1}, AvgPSDenvelope, 'LineWidth', 0.8); 
% LineWidth=2
xlabel('Frequency (Hz)');
ylabel('PSD (a.u.^2/Hz)');
title('PSD Audio Envelope');
ylim([0, 0.001]);
xlim([0,20]);
% grid on;
%%
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 16);
%%
h = findobj(gca, 'Type', 'Line'); % Find line objects in the current axis

for i = 1:length(h)
    if strcmp(h(i).Tag, 'Outliers') % Modify only the outliers
        h(i).Marker = 'o'; % Change marker style (e.g., 'o', '*', '+')
        h(i).MarkerSize = 8; % Change marker size
        h(i).MarkerEdgeColor = 'r'; % Change marker color
    elseif strcmp(h(i).Tag, 'Median') % Modify the median line
        h(i).Color = 'b'; % Change median color
        h(i).LineWidth = 2; % Change line width
    end
end

%%
subplot(2, 2, 3);
xlim([0, 2]);
%% 
subplot(2, 2, 4);
xlim([0, 30]);

%%

subplot(2, 2, 4);
xlim([0, 10]);
ylim([0, 0.001]);
%%
title('');
box off; grid off
set(gca, FontSize=15, LineWidth=2);  % Set the font size to 10
%%
s = 20; % Subject index
k = 2;  % Task index (Wide audio)

% Load audio data for Participant 20, Condition 2
fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k});
[fs_eeg, ~, fs_audio, audio_dat, ~] = LoadEEG(s, k, sbj, task);

% Resample audio to match EEG sampling rate
[p, q] = rat(fs_eeg / fs_audio);
audio_resampled = resample(audio_dat, p, q);

% Extract envelope using mTRF Toolbox and normalize
envelope = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg);
EnvNorm = (envelope - min(envelope)) / (max(envelope) - min(envelope));

% Define bin edges for histogram (using normalized amplitude values)
numBins = 8; % Specify the number of bins
binEdges_dB = linspace(0, 0.98, numBins + 1);
binEdges_amp = 10.^(binEdges_dB / 20);
normbinEdges_amp = normalize(binEdges_amp, 2, 'range');

% Calculate histogram counts for the envelope
[DbCounts, edges, binMask] = histcounts(EnvNorm, normbinEdges_amp);
Wide_audio_counts = DbCounts; % Only storing the wide audio count for subject 20

% Power Spectral Density (PSD) for the audio stimulus
% [pxx_audio, f_audio] = pwelch(audio_resampled, [], [], [], fs_eeg);
[pxx_audio, f_audio] = pwelch(audio_dat, [], [], [], fs_audio);

% Power Spectral Density (PSD) for the envelope
[pxx_env, f_env] = pwelch(envelope, [], [], [], fs_eeg);

% Normalize audio data for plotting
NormAudiodat = normalize(audio_dat, 2, "range", [-1 1]);

%% 
% Subplot 4: PSD of Wide audio envelope
subplot(2, 2, 4);
plot(f_env, pxx_env); % Convert to dB scale
xlabel('Frequency (Hz)');
ylabel('PSD (a.u.^2/Hz)');
title('PSD of Audio Envelope');
grid on;

ylim([0, 0.001]);
xlim([0, 30]);
%% Plotting
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
window_time = [1149 1151]; % Specify time window for zoomed-in view

% Colors for plotting
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
yellow = [0.9290 0.6940 0.1250];

% Plotting the results
figure;

% Subplot 1: Time series of normalized audio data
subplot(2, 2, 1);
plot(time_audio, NormAudiodat, 'LineWidth', 1, 'DisplayName', 'Original Audio (normalized)');  % Original audio with thicker line
xlabel('Time (s)');
ylabel('Amplitude (Normalized)');
xlim(window_time);
title('Sound Sample');
grid on;
box off;

% Subplot 2: Amplitude histogram of the envelope for Wide audio
subplot(2, 2, 2);
histogram('BinEdges', normbinEdges_amp, 'BinCounts', Wide_audio_counts);
xlabel('Amplitude (Normalized)');
ylabel('Counts');
title('Amplitude Distribution of Envelope');
set(gca, 'View', [90 -90]); % Rotate plot
grid on;
box off;
%% 
% Subplot 3: PSD of Wide audio stimulus
subplot(2, 2, 3);
plot(f_audio, pxx_audio, 'Color', blue); % Convert to dB scale
xlabel('Frequency (Hz)');
ylabel('PSD (a.u.^2/Hz)');
title('PSD of Audio Stimulus');
grid on;
%% 
% Subplot 4: PSD of Wide audio envelope
subplot(2, 2, 4);
plot(f_env, pxx_env); % Convert to dB scale
xlabel('Frequency (Hz)');
ylabel('PSD (a.u.^2/Hz)');
title('PSD of Audio Envelope');
grid on;
% xlim([0, 30]); 
%% 
ylim([0, 0.002]);
% xlim([0, 30]);
%%
xlim([0, 50]);


