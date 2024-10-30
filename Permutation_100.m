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
        [EnvNorm, ABenvNorm, ~, ~, resp] = feature_deriv(audio_dat, fs_audio, fs_eeg, resp);

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
%% 

% % Save the StatsParticipantTask structure to a .mat file
savePath = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask_Permutation';
save(savePath, 'StatsParticipantTask_Permutation');     % Save the StatsParticipantTask structure to the specified path
%%
load("StatsParticipantTask_Permutation.mat")
OT_setup
%%
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

%%
num_participants = length(sbj);

% Initialize arrays to store mean values for each condition
narrow_r_values = cell(num_participants, 1);
wide_r_values = cell(num_participants, 1);

% Loop over each participant and collect the mean values for "narrow" and "wide"
for s = 1:num_participants
    for k = 1:length(task)
        % Check if the task is narrow or wide
        if strcmp(task{k}, 'narrow') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            % Extract mean values from narrow condition
            % Assuming null_distributions is a 100x2 cell array
            struct_values = StatsParticipantTask_Permutation(s, k).null_distributions(:, 1);
            narrow_r_values{s} = zeros(size(struct_values)); % Initialize array to hold mean values
            
            % Calculate the mean value for each struct
            for idx = 1:length(struct_values)
                % Extract the value you want to compute the mean of
                narrow_r_values{s}(idx) = mean(struct_values{idx}.r); % Assuming 'r' is the field you want
            end
            
        elseif strcmp(task{k}, 'wide') && isfield(StatsParticipantTask_Permutation(s, k), 'null_distributions')
            % Extract mean values from wide condition
            struct_values = StatsParticipantTask_Permutation(s, k).null_distributions(:, 1);
            wide_r_values{s} = zeros(size(struct_values)); % Initialize array to hold mean values
            
            % Calculate the mean value for each struct
            for idx = 1:length(struct_values)
                % Extract the value you want to compute the mean of
                wide_r_values{s}(idx) = mean(struct_values{idx}.r); % Assuming 'r' is the field you want
            end
        end
    end
end

% Optional: Display the size of the results to verify
disp(size(narrow_r_values{1})); % Should display 100x1 for the first participant's narrow values
disp(size(wide_r_values{1})); % Should display 100x1 for the first participant's wide values
% Loop through each subject and create a box plot
figure;
subplot(2,1,1)
% Loop through each subject and create a box plot
for s = 1:length(sbj)
    % Create a box plot for the current subject at position 's'
    boxplot(narrow_r_values{s}, 'Positions', s, 'Widths', 0.6); % You can adjust the width if desired
    hold on; % Keep the figure open for the next box plot
end

% Optionally, you can add labels and title
xlabel('Subjects');
ylabel('Values');
title('Boxplots for All Subjects: Narrow');

% Set x-ticks to be at each subject's position
set(gca, 'XTick', 1:length(sbj)); 
set(gca, 'XTickLabel', 1:length(sbj)); % Label x-ticks
grid on; % Optional: Add grid for better visibility

% Loop through each subject and create a box plot
% figure;
subplot(2,1,2)
% Loop through each subject and create a box plot
for s = 1:length(sbj)
    % Create a box plot for the current subject at position 's'
    boxplot(wide_r_values{s}, 'Positions', s, 'Widths', 0.6); % You can adjust the width if desired
    hold on; % Keep the figure open for the next box plot
end

% Optionally, you can add labels and title
xlabel('Subjects');
ylabel('Values');
title('Boxplots for All Subjects: Wide');

% Set x-ticks to be at each subject's position
set(gca, 'XTick', 1:length(sbj)); 
set(gca, 'XTickLabel', 1:length(sbj)); % Label x-ticks
grid on; % Optional: Add grid for better visibility



%%
% Create a figure for box plots and density plots
figure;

% Loop through each subject and create a box plot for narrow condition
for s = 1:length(sbj)
    % Create a box plot for the current subject at position 's'
    boxplot(narrow_r_values{s}, 'Positions', s, 'Widths', 0.4); % Narrow condition box plot
    hold on; % Keep the figure open for the next plot
    
    % Overlay density plot
    % You may need to adjust the bandwidth parameter (0.5 in this case)
    [f, xi] = ksdensity(narrow_r_values{s}, 'Bandwidth', 0.5);
    plot(xi, f * max(get(gca, 'YLim')), 'LineWidth', 2); % Scale density to fit box plot
end

% Loop through each subject and create a box plot for wide condition
for s = 1:length(sbj)
    % Create a box plot for the current subject at position 's'
    boxplot(wide_r_values{s}, 'Positions', s + length(sbj), 'Widths', 0.4); % Wide condition box plot
    
    % Overlay density plot
    [f, xi] = ksdensity(wide_r_values{s}, 'Bandwidth', 0.5);
    plot(xi, f * max(get(gca, 'YLim')), 'LineWidth', 2); % Scale density to fit box plot
end

% Optionally, you can add labels and title
xlabel('Subjects');
ylabel('Values');
title('Boxplots with Density Plots for Narrow and Wide Conditions');

% Set x-ticks to be at each subject's position
set(gca, 'XTick', 1:2:length(sbj)*2); % Set x-ticks for both conditions
set(gca, 'XTickLabel', repmat(1:length(sbj), 1, 2)); % Label x-ticks for both conditions

% Add legend to differentiate between narrow and wide
legend('Narrow Density', 'Wide Density', 'Location', 'best');

grid on; % Optional: Add grid for better visibility


%%
