%%% Permutation Test for EEG Correlation Analysis

% clear;clc;
%% Channel selection
warning('off', 'MATLAB:nearlySingularMatrix');
OT_setup; channel_idx = [9, 7, 6, 8, 10];
%% load("StatsParticipantTask_Permutation_ABenvelope")

OT_setup;
load("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_Permutation_ABenvelope.mat")

%%

% Access the null_distributions cell array for the chosen participant and task
Narroww = zeros(20,5);
Widee = zeros(20,5);

for s = 1:length(sbj) % Iterate through subjects
    null_distributions_narrow = StatsParticipantTask_Permutation_ABenvelope(s, 1).null_distributions{2};
    null_distributions_wide = StatsParticipantTask_Permutation_ABenvelope(s, 2).null_distributions{8};

    Narroww(s,:) = null_distributions_narrow.r;
    Widee(s,:) = null_distributions_wide.r;
end

% % Display the struct fields and contents
% figure;
% plot(Narroww)

%% AB Envelope Permutation Calculation
% This script performs permutation testing to generate null distributions
% for AB Envelope correlation statistics across subjects and tasks.

StatsParticipantTask_Permutation_ABenvelope = struct();

% Define the number of permutations to generate the null distribution
num_permutations = 100;

% Loop over all participants and tasks
for s = 1:length(sbj) % Iterate through subjects
    for k = 1:length(task) % Iterate through tasks
        fprintf('Processing Subject: %s, Task: %s\n', sbj{s}, task{k});

        % Load EEG and audio data for the current subject and task
        [fs_eeg, full_resp, fs_audio, audio_dat, ~] = LoadEEG(s, k, sbj, task);
        resp = full_resp(:, channel_idx); % Extract the selected EEG channels

        % Derive AB envelope features and preprocess data
        [~, ABenvNorm, ~, ~, resp, binEdges_dB] = feature_deriv(audio_dat, fs_audio, fs_eeg, resp);

        % Initialize storage for null distribution statistics
        null_stats = cell(num_permutations, 1);
        ABenvNorm = ABenvNorm(:, 1:8);

        % Perform permutation testing
        for p = 1:num_permutations
            % Randomly shuffle the columns of the AB envelope (null hypothesis)
            permuted_envelopes = {ABenvNorm(:, randperm(size(ABenvNorm, 2)))};

            % Use the permuted envelope as input stimulus for the model
            % stim = permuted_envelopes{1};
            stim = ABenvNorm;
            % Model parameters
            tmin = -100;         % Minimum time lag in ms
            tmax = 400;          % Maximum time lag in ms
            trainfold = 10;      % Number of folds for training
            testfold = 1;        % Number of folds for testing
            Dir = 1;             % Forward modeling direction
            lambdas = logspace(-4, 4, 10); % Regularization parameters

            % Partition data into training and testing sets
            [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);

            % Standardize EEG responses (z-score normalization)
            rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
            rtestz = zscore(rtest, [], 'all');

            % Perform cross-validation to find the optimal regularization parameter
            cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);
            l = mean(cv.r, 3); % Average correlation across channels
            [~, l_idx] = max(mean(l, 1)); % Find lambda index with maximum correlation
            l_opt = lambdas(l_idx); % Optimal regularization parameter

            % Train the TRF model using optimal regularization
            model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);

            % Test the model on the testing set and obtain statistics
            [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

            disp(stats.r); % Display correlation coefficients for each permutation

            % Store the statistics for the current permutation
            null_stats{p} = stats;
        end

        % Save null distribution statistics and IDs in the structure
        StatsParticipantTask_Permutation_ABenvelope(s, k).null_distributions = null_stats;
        StatsParticipantTask_Permutation_ABenvelope(s, k).ParticipantID = s;
        StatsParticipantTask_Permutation_ABenvelope(s, k).TaskID = k;

        disp('Permutation test completed for this subject and task!');
    end
end
%%
% Save the results to a .mat file
savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_Permutation_ABenvelope';
if ~isfolder(fileparts(savePath))
    mkdir(fileparts(savePath)); % Create directory if it does not exist
end
save(savePath, 'StatsParticipantTask_Permutation_ABenvelope'); % Save results

disp('All results saved successfully!');


%% Combined AB and Standard Envelope Permutation Calculation
% This script performs permutation testing for AB and Standard Envelope features
% AB Envelope: Shift bins randomly 100 times.
% Standard Envelope: Use randomly chosen other participants' envelopes 100 times.

StatsParticipantTask_Permutation_Combined = struct();

% Define the number of permutations
num_permutations = 100;

% Loop over all participants and tasks
for s = 1:length(sbj) % Iterate through subjects
    for k = 1:length(task) % Iterate through tasks
        fprintf('Processing Subject: %s, Task: %s\n', sbj{s}, task{k});
        
        % Load EEG and audio data for the current subject and task
        [fs_eeg, full_resp, fs_audio, audio_dat, ~] = LoadEEG(s, k, sbj, task);
        resp = full_resp(:, channel_idx); % Extract the selected EEG channels
        
        % Derive AB and Standard envelope features and preprocess data
        [~, ABenvNorm, ~, ~, resp, binEdges_dB] = feature_deriv(audio_dat, fs_audio, fs_eeg, resp);

        % Initialize storage for null distribution statistics
        null_stats_AB = cell(num_permutations, 1);
        null_stats_Standard = cell(num_permutations, 1);

        % AB Envelope: Shift bins 100 times
        fprintf('ABenvelope - Subject: %s, Task: %s\n', sbj{s}, task{k});
        for p = 1:num_permutations

            % Randomly shuffle the columns of the AB envelope
            permuted_AB = ABenvNorm(:, randperm(size(ABenvNorm, 2)));

            % Use the permuted envelope as input stimulus for the model
            stim_AB = permuted_AB;

            % Model parameters
            tmin = -100;         
            tmax = 400;          
            trainfold = 10;      
            testfold = 1;        
            Dir = 1;             
            lambdas = logspace(-4, 4, 10);

            % Partition data for training and testing
            [strain, rtrain, stest, rtest] = mTRFpartition(stim_AB, resp, trainfold, testfold);

            % Standardize EEG responses
            rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
            rtestz = zscore(rtest, [], 'all');

            % Cross-validation to find the optimal regularization parameter
            cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);
            l = mean(cv.r, 3); 
            [~, l_idx] = max(mean(l, 1)); 
            l_opt = lambdas(l_idx); 

            % Train and test the model
            model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
            [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

            % Store statistics
            null_stats_AB{p} = stats;
        end

        % Standard Envelope: Use other participants' envelopes 100 times
        fprintf('Standard Envelope - Subject: %s, Task: %s\n', sbj{s}, task{k});

        for p = 1:num_permutations
            % Randomly select envelopes from other participants
            other_s = setdiff(1:length(sbj), s); % Exclude current participant
            rand_participant = other_s(randi(length(other_s))); % Select a random participant
            [~, ~, ~, audio_other, ~] = LoadEEG(rand_participant, k, sbj, task); % Load audio
            % [EnvNorm_other, ~, ~, ~, ~, ~] = feature_deriv(audio_other, fs_audio, fs_eeg, resp);
            [EnvNorm_other, ~, ~, ~, resp, binEdges_dB] = feature_deriv(audio_other, fs_audio, fs_eeg, resp);

            % Assign the envelope from another participant
            stim_Standard = EnvNorm_other;

            % Partition data for training and testing
            [strain, rtrain, stest, rtest] = mTRFpartition(stim_Standard, resp, trainfold, testfold);

            % Standardize EEG responses
            rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
            rtestz = zscore(rtest, [], 'all');

            % Cross-validation to find the optimal regularization parameter
            cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);
            l = mean(cv.r, 3); 
            [~, l_idx] = max(mean(l, 1)); 
            l_opt = lambdas(l_idx); 

            % Train and test the model
            model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
            [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

            % Store statistics
            null_stats_Standard{p} = stats;
        end

        % Save results for the current subject and task
        StatsParticipantTask_Permutation_Combined(s, k).null_distributions_AB = null_stats_AB;
        StatsParticipantTask_Permutation_Combined(s, k).null_distributions_Standard = null_stats_Standard;
        StatsParticipantTask_Permutation_Combined(s, k).ParticipantID = s;
        StatsParticipantTask_Permutation_Combined(s, k).TaskID = k;

        disp('Permutation test completed for this subject and task!');
    end
end

% Save the combined results to a .mat file
savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_Permutation_Combined.mat';
if ~isfolder(fileparts(savePath))
    mkdir(fileparts(savePath)); % Create directory if it does not exist
end
save(savePath, 'StatsParticipantTask_Permutation_Combined');

disp('All results saved successfully!');

%%
load("StatsParticipantTask_Permutation_Combined.mat")
OT_setup
%%
% load("StatsParticipantTask_Permutation.mat")
OT_setup
%%
% StatsParticipantTask_Permutation = StatsParticipantTask_Permutation_Combined;
% Number of participants and tasks
num_participants = length(sbj);

% Initialize arrays to store permutation distribution values for each condition
narrow_r_values = cell(num_participants, 1);
wide_r_values = cell(num_participants, 1);

% Loop over each participant and collect the permutation distributions for "narrow" and "wide"
for s = 1:num_participants
    for k = 1:length(task)
        % Check if the task is narrow or wide
        if strcmp(task{k}, 'narrow') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            % Extract standard envelope permutation stats for narrow condition
            narrow_r_values{s} = cellfun(@(x) x.r, StatsParticipantTask_Permutation(s, k).null_distributions(:, 1), 'UniformOutput', false);
            narrow_r_values{s} = cell2mat(narrow_r_values{s}'); % Convert to matrix
        elseif strcmp(task{k}, 'wide') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            % Extract standard envelope permutation stats for wide condition
            wide_r_values{s} = cellfun(@(x) x.r, StatsParticipantTask_Permutation(s, k).null_distributions(:, 1), 'UniformOutput', false);
            wide_r_values{s} = cell2mat(wide_r_values{s}'); % Convert to matrix
        end
    end
end
% Initialize arrays for confidence interval calculations
num_participants = length(sbj);
narrow_mean = zeros(num_participants, 1);
narrow_ci = zeros(num_participants, 2);
wide_mean = zeros(num_participants, 1);
wide_ci = zeros(num_participants, 2);

% Calculate mean and 95% CI for each participant and condition
for s = 1:num_participants
    if ~isempty(narrow_r_values{s})
        data = narrow_r_values{s};
        narrow_mean(s) = mean(data);
        sem = std(data) / sqrt(length(data)); % Standard error of the mean
        narrow_ci(s, :) = [narrow_mean(s) - 1.96 * sem, narrow_mean(s) + 1.96 * sem];
    end
    if ~isempty(wide_r_values{s})
        data = wide_r_values{s};
        wide_mean(s) = mean(data);
        sem = std(data) / sqrt(length(data)); % Standard error of the mean
        wide_ci(s, :) = [wide_mean(s) - 1.96 * sem, wide_mean(s) + 1.96 * sem];
    end
end

% Plot Boxplots for "Narrow" with Confidence Intervals
figure;
subplot(2, 1, 1);
hold on;
for s = 1:length(sbj)
    boxplot(narrow_r_values{s}, 'Positions', s, 'Widths', 0.6);
    % Overlay confidence intervals
    errorbar(s, narrow_mean(s), narrow_mean(s) - narrow_ci(s, 1), narrow_ci(s, 2) - narrow_mean(s), ...
        'k', 'LineWidth', 1.5); % Black error bars
end
title('Boxplots for Narrow Condition with 95% CI');
xlabel('Participants');
ylabel('Values');
grid on;
% Adjust x-ticks for better readability
set(gca, 'XTick', 1:num_participants);
set(gca, 'XTickLabel', 1:num_participants);

% Plot Boxplots for "Wide" with Confidence Intervals
subplot(2, 1, 2);
hold on;
for s = 1:length(sbj)
    boxplot(wide_r_values{s}, 'Positions', s, 'Widths', 0.6);
    % Overlay confidence intervals
    errorbar(s, wide_mean(s), wide_mean(s) - wide_ci(s, 1), wide_ci(s, 2) - wide_mean(s), ...
        'k', 'LineWidth', 1.5); % Black error bars
end
title('Boxplots for Wide Condition with 95% CI');
xlabel('Participants');
ylabel('Values');
grid on;

% Adjust x-ticks for better readability
set(gca, 'XTick', 1:num_participants);
set(gca, 'XTickLabel', 1:num_participants);

%%
% Number of participants
num_participants = length(sbj);
sortIdxmean = [20, 3, 4, 5, 15, 2, 12, 7, 10, 17, 11, 16, 13, 6, 1, 9, 14, 18, 19, 8];

% Initialize arrays to store permutation distribution means for each condition
narrow_r_values = cell(num_participants, 1);
wide_r_values = cell(num_participants, 1);

% Loop over participants and extract mean values for "narrow" and "wide" tasks
for s = 1:num_participants
    for k = 1:length(task)
        if strcmp(task{k}, 'narrow') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            narrow_r_values{s} = cellfun(@(x) mean(x.r), StatsParticipantTask_Permutation(s, k).null_distributions(:, 1));
        elseif strcmp(task{k}, 'wide') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            wide_r_values{s} = cellfun(@(x) mean(x.r), StatsParticipantTask_Permutation(s, k).null_distributions(:, 1));
        end
    end
end

% Combine narrow and wide condition values
combined_r_val = cellfun(@(n, w) (n + w) / 2, narrow_r_values, wide_r_values, 'UniformOutput', false);

% Calculate mean and 95% CI for each participant
AVGcond_Rvals = zeros(num_participants, 1);
combined_R_ci = zeros(num_participants, 2);

for s = 1:num_participants
    if ~isempty(combined_r_val{s})
        AVGcond_Rvals(s) = mean(combined_r_val{s});
        sem = std(combined_r_val{s}) / sqrt(length(combined_r_val{s})); % Standard error of the mean
        combined_R_ci(s, :) = [AVGcond_Rvals(s) - 1.96 * sem, AVGcond_Rvals(s) + 1.96 * sem];
    end
end

% Plot boxplots and confidence intervals
figure;
hold on;
for s = 1:num_participants
    idx = sortIdxmean(s);
    if ~isempty(combined_r_val{idx})
        boxplot(combined_r_val{idx}, 'Positions', idx, 'Widths', 0.6);
        errorbar(s, AVGcond_Rvals(idx), AVGcond_Rvals(idx) - combined_R_ci(idx, 1), ...
                 combined_R_ci(idx, 2) - AVGcond_Rvals(idx), 'k', 'LineWidth', 1.5);
    end
end
hold off;

% Customize the plot
title('Boxplot for Combined Narrow and Wide Conditions with 95% CI');
xlabel('Participants');
ylabel('Values');
set(gca, 'XTick', 1:num_participants, 'XTickLabel', 1:num_participants);
grid on;



%% Standard Envelope Permutation Calculation
StatsParticipantTask_Permutation = struct();
% Define number of permutations
num_permutations = 100;

% Loop over participants and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task
        % Load EEG and Audio
        [fs_eeg, full_resp, fs_audio, audio_dat, ~]  = LoadEEG(s, k, sbj, task);
        resp = full_resp(:, channel_idx); % Select 5 channels


        % Feature derivation
        [EnvNorm, ABenvNorm, ~, ~, resp, binEdges_dB] = feature_deriv(audio_dat, fs_audio, fs_eeg, resp);

        feature_names = {'EnvNorm', 'ABenvNorm'};
        features = {EnvNorm, ABenvNorm};


        % Initialize storage for null distribution stats
        null_stats = cell(num_permutations, length(features));

        % Permutation test
        for p = 1:num_permutations
            % Randomly choose envelopes with replacement for each trial
            permuted_envelopes = cellfun(@(x) x(randperm(size(x, 1)), :), features, 'UniformOutput', false);
            % permuted_envelopes = cellfun(@(x) circshift(x, randperm(size(x, 1)), features, 'UniformOutput', false);

            for feature_idx = 1:length(features)
                stim = permuted_envelopes{feature_idx}; % Assign random envelope to stim

                % Model hyperparameters
                tmin = -100;
                tmax = 400;         % tmax = 500;
                trainfold = 10;
                testfold = 1;
                Dir = 1;            % specifies the forward modeling

                lambdas = linspace(10e-4, 10e4,10);
                % Ensure matching sizes
                % size_check(stim, resp);

                % Partition data for cross-validation
                [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);
                rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
                rtestz = zscore(rtest, [], 'all');

                % Cross-validation to find optimal regularization parameter
                cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);
                l = mean(cv.r,3); % Average across channels
                [~, l_idx] = max(mean(l,1));
                l_opt = lambdas(l_idx);

                % Train and test the model
                model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
                [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);

                % Store stats for null distribution
                null_stats{p, feature_idx} = stats;
            end
        end

        % Store null distributions in StatsParticipantTask structure
        StatsParticipantTask_Permutation(s, k).null_distributions = null_stats;

        % Save Participant and Task IDs
        StatsParticipantTask_Permutation(s, k).ParticipantID = s;
        StatsParticipantTask_Permutation(s, k).TaskID = k;

        disp('Results are saved!');
    end
end

% % Save the StatsParticipantTask structure to a .mat file
savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_Permutation';
save(savePath, 'StatsParticipantTask_Permutation');     % Save the StatsParticipantTask structure to the specified path

%%
load("StatsParticipantTask_Permutation_Combined.mat")
OT_setup
%%

% Number of participants
num_participants = length(sbj);

% Initialize cell array to store combined values
combined_r_values = cell(num_participants, 1);

% Extract and combine narrow and wide values
for s = 1:num_participants
    narrow_vals = [];
    wide_vals = [];
    
    for k = 1:length(task)
        % Check for "narrow" task
        if strcmp(task{k}, 'narrow') && isfield(StatsParticipantTask_Permutation_Combined(s, k), 'null_distributions_AB') && ...
           ~isempty(StatsParticipantTask_Permutation_Combined(s, k).null_distributions_AB)
            narrow_vals = cellfun(@(x) mean(x.r), StatsParticipantTask_Permutation_Combined(s, k).null_distributions_AB);
        end
        
        % Check for "wide" task
        if strcmp(task{k}, 'wide') && isfield(StatsParticipantTask_Permutation_Combined(s, k), 'null_distributions_AB') && ...
           ~isempty(StatsParticipantTask_Permutation_Combined(s, k).null_distributions_AB)
            wide_vals = cellfun(@(x) mean(x.r), StatsParticipantTask_Permutation_Combined(s, k).null_distributions_AB);
        end
    end
    
    % Combine narrow and wide values
    if ~isempty(narrow_vals) && ~isempty(wide_vals)
        combined_r_values{s} = (narrow_vals + wide_vals) / 2;
    end
end

% Plot all combined distributions
figure;
hold on;
colors = lines(num_participants); % Generate distinct colors for each participant

for s = 1:num_participants
    if ~isempty(combined_r_values{s})
        % Scatter plot for each participant's 100 combined values
        scatter(repmat(s, 1, length(combined_r_values{s})), combined_r_values{s}, 36, colors(s, :), 'filled');
    end
end
hold off;

% Customize plot
title('Distribution of Combined Narrow and Wide Null Values Across Participants');
xlabel('Participants');
ylabel('Combined Values');
xlim([0, num_participants + 1]);
xticks(1:num_participants);
grid on;
