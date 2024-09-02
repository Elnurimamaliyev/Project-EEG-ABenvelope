%% Amplitude binned envelope



%% Feature extraction and amplitude binning

% Define the number of bins for amplitude binning
num_bins = 5; % Number of bins, adjust as needed

% Compute the amplitude range for binning
min_amp = min(stim_Env(:));
max_amp = max(stim_Env(:));
bin_edges = linspace(min_amp, max_amp, num_bins + 1);

% Initialize the binned envelope matrix
stim_binned = zeros(size(stim_Env));

% Bin the envelope data
for bin_idx = 1:num_bins
    bin_mask = (stim_Env >= bin_edges(bin_idx)) & (stim_Env < bin_edges(bin_idx + 1));
    stim_binned(bin_mask) = bin_idx;
end

% Convert the binned data to a binary matrix where each column corresponds to a bin
stim_binned_matrix = zeros(size(stim_Env, 1), num_bins);

for bin_idx = 1:num_bins
    stim_binned_matrix(:, bin_idx) = (stim_binned == bin_idx);
end

% Ensure the binned matrix is the same length as resp
stim_binned_matrix = stim_binned_matrix(1:size(resp, 2), :);

% Update feature names
feature_names = {'Envelope', 'Amplitude-Binned Envelope'};

% Add the binned envelope to the features list
features = {stim_Env, stim_binned_matrix};

% Using different features to get Accuracies
figure;

for feature_idx = 1:length(features)

    % Assign the current feature set to stim
    stim = features{feature_idx};

    % Optionally, display the name of the current feature set
    fprintf('Processing feature set: %s\n', feature_names{feature_idx});

    % Ensure the stim is the correct size (matching the length of resp)
    stim = stim(1:size(resp, 2), :);

    % Model hyperparameters
    tmin = -100;
    tmax = 400;
    trainfold = 10;
    testfold = 1;

    % Partition the data into training and test data segments
    [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp', trainfold, testfold);

    % Compute model weights
    model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);

    % Test the model on unseen neural data
    [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);

    % Plotting in a 3x2 grid
    % Model weights
    subplot(2, 2, feature_idx * 2 - 1)
    plot(model.t, squeeze(model.w(1, :, :))); % model weights
    title(sprintf('Weights (%s)', feature_names{feature_idx}))
    ylabel('a.u.')
    xlabel('time in ms.')

    % GFP (Global Field Power)
    subplot(2, 2, feature_idx * 2)
    boxplot(stats.r) % channel correlation values
    title(sprintf('GFP (%s)', feature_names{feature_idx}))
    ylabel('correlation')
end

% Adjust the layout
sgtitle('Feature Comparisons') % Add a super title to the figure



%%

% Define time arrays
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
time_env = (0:length(stim_Env) - 1) / fs_eeg;       % Time array for envelope and binned envelope

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds

% Plotting time series
figure;

% Plot original sound
subplot(3, 1, 1);
plot(time_audio, audio_dat);
title('Original Sound');
xlabel('Time (seconds)');
ylabel('Amplitude');
% xlim([start_window end_window]);  % Zoom in on the original sound

% Plot envelope
subplot(3, 1, 2);
plot(time_env, stim_Env);
title('Envelope');
xlabel('Time (seconds)');
ylabel('Envelope Amplitude');
% xlim([start_window end_window]);  % Zoom in on the envelope

% Plot amplitude-binned envelope
subplot(3, 1, 3);
imagesc(time_env, db_bin_edges, stim_binned_matrix'); % Transpose for correct orientation
% axis xy; % Correct orientation of y-axis
title('Amplitude-Binned Envelope');
xlabel('Time (seconds)');
ylabel('Amplitude Bin');
% xlim([start_window end_window]);  % Zoom in on the envelope
% colorbar; % Add colorbar for bin representation

% Adjust the layout
sgtitle('Time Series of Original Sound, Envelope, and Amplitude-Binned Envelope'); % Add a super title to the figure


% Define time arrays
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
time_env = (0:length(stim_Env) - 1) / fs_eeg;       % Time array for envelope and binned envelope

% Define the zoom-in range (in seconds)
start_window = 2.5;  % Start time in seconds
end_window = 15;     % End time in seconds

% Plotting time series
figure;

% Plot original sound
subplot(3, 1, 1);
plot(time_audio, audio_dat);
title('Original Sound');
xlabel('Time (seconds)');
ylabel('Amplitude');
xlim([start_window end_window]);  % Zoom in on the original sound

% Plot envelope
subplot(3, 1, 2);
plot(time_env, stim_Env);
title('Envelope');
xlabel('Time (seconds)');
ylabel('Envelope Amplitude');
xlim([start_window end_window]);  % Zoom in on the envelope

% Plot amplitude-binned envelope
subplot(3, 1, 3);
imagesc(time_env, 1:length(db_bin_edges), stim_binned_matrix'); % Transpose for correct orientation
axis xy; % Correct orientation of y-axis
title('Amplitude-Binned Envelope');
xlabel('Time (seconds)');
ylabel('Amplitude Bin');
xlim([start_window end_window]);  % Zoom in on the envelope
% colorbar; % Add colorbar for bin representation

% Adjust the layout
sgtitle('Time Series of Original Sound, Envelope, and Amplitude-Binned Envelope'); % Add a super title to the figure
%%%
































%%
amp_conti = [1, 4, 5, 6, 6, 8, 2, 3, 1, 8, 2, 4, 5, 1, 8, 5, 7, ...
    6, 1, 8, 8, 6, 2, 4, 2, 7, 9, 8, 3, 4];
t = 1:length(amp_conti);


matrix_bin = AmplitudeBinner(amp_conti)








% 
% % Initialize the matrix to store the modified amplitude values for each level
% matrix_bin = zeros(10, length(t)); % 10 rows for 10 amplitudes, columns for time points
% 
% % Populate the matrix_bin with amplitude values
% for i = 1:10
%     matrix_bin(i, :) = i; % Pre-fill each row with its amplitude level
% end
% 
% % Update the matrix_bin based on the given logic
% for i = 1:length(amp_conti)
%     amp_value = amp_conti(i);
%     if amp_value >= 1 && amp_value <= 10 % Ensure amplitude value is within the valid range
%         matrix_bin(amp_value, i) = matrix_bin(amp_value, i) + 0.5; % Add 0.5 to the current amplitude value
%     end
% end

% Create the first figure for the original amplitude data
figure;
subplot(2,1,1);
plot(t, amp_conti);
title('Amplitude over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([0 10]); % Set y-axis limits

% Create the second figure for the modified amplitude levels
subplot(2,1,2);
plot(t, matrix_bin);
title('Modified Amplitude Levels over Time');
xlabel('Time');
ylabel('Amplitude');
ylim([0 10]); % Set y-axis limits

% Create a legend for the amplitudes
legend(arrayfun(@(x) sprintf('Amplitude %d', x), 1:10, 'UniformOutput', false));
