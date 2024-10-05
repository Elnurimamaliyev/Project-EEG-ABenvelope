%% Cleaning
clear, clc;

% Add Main paths
OT_setup

% Channel selection
[~, ~, ~, ~, EEG] = LoadEEG(1, 1, sbj, task); eeg_channel_labels = {EEG.chanlocs.labels}; selected_channels = {'C3', 'FC2', 'FC1', 'Cz', 'C4'}; [~, channel_idx] = ismember(selected_channels, eeg_channel_labels);  % Find the indices corresponding to the selected channels in your EEG data
clear EEG eeg_channel_labels
%% Search Grid
% Grid of bin ranges from 0 to 80 with different minimum values
min_bin_values = [0, 10, 20, 30, 40, 50, 60, 70];
max_bin_values = [6, 10, 20, 30, 40, 50, 60, 70, 80];
num_bins_values = [4, 5, 6, 8, 9, 10, 12, 15, 16];

save_directory = 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\BinsLoop'; % Specify your desired path here


for min_i = 1:length(min_bin_values)
    for max_i = 1:length(max_bin_values)
        for bin_i = 1:length(num_bins_values)

            % Get the current bin values
            min_bin = min_bin_values(min_i);
            max_bin = max_bin_values(max_i);
            num_bins = num_bins_values(bin_i);

            % Check conditions to exclude invalid combinations
            if min_bin >= max_bin
                continue; % Skip to the next iteration if conditions are not met
            end

            % Create bin edges
            binEdges_dB = linspace(min_bin, max_bin, num_bins + 1)';
            name_struct = struct(); % Initialize the structure

            % Run the search grid function
            name_struct_new = run_SearchGrid(name_struct, channel_idx, sbj, task, binEdges_dB, num_bins);

            % Name of struct to save
            save_name = fullfile(save_directory, sprintf('Min_%d_Max_%d_Bin_%d.mat', min_bin, max_bin, num_bins));
            try
                % Save the current results in a .mat file named after the bin configuration
                save(save_name, 'name_struct_new', '-v7.3'); % Use '-v7.3' for larger datasets if necessary
            catch ME
                fprintf('Error saving %s: %s\n', save_name, ME.message);
            end


        end
    end
end





%%
function [name_struct] = run_SearchGrid(name_struct, channel_idx, sbj, task, binEdges_dB, num_bins )



    for s = 1:length(sbj)
        for k = 1:length(task)
            name_struct(s, k).ParticipantID = s;  % s is the participant ID
            name_struct(s, k).TaskID = k;         % k is the task ID
            fprintf('Subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task
            % Load the EEG and Audio
            [fs_eeg, full_resp, fs_audio, audio_dat, ~]  = LoadEEG(s, k, sbj, task);
            % Select only 5 channels from response
            resp = full_resp(:, channel_idx);
            % Original Envelope Feature Generation   
            Env = mTRFenvelope(double(audio_dat)', fs_audio, fs_eeg); % Extract the envelope using the mTRF toolbox
            [Env, resp] = size_check(Env, resp); % Ensure matching sizes
    
            % Normalize
            EnvNorm = normalize(Env,1,'range');
    
            % bin generation
            binEdges_linear = 10.^(binEdges_dB / 20);
            NormBinEdges = normalize(binEdges_linear,1,'range');
            % Extend bin edges with eps, MATLAB’s smallest floating-point value
            NormBinEdges(end) = NormBinEdges(end) + eps; NormBinEdges(1) = NormBinEdges(1) - eps; % Extend the first and last bin edge slightly

    
            % Calculate the histogram counts and bin indices and amplitude density (Using histcounts)
            [DbCounts, ~, binIndices] = histcounts(EnvNorm, NormBinEdges);
    
            % Initialize the binned envelope matrix
            ABenv = zeros(size(EnvNorm,1), num_bins);
    
            % Binned envelope
            for i = 1:length(binIndices)
                ABenv(i, binIndices(i)) = EnvNorm(i);
            end
    
            % Normalize each bin 
            ABenvNorm = normalize(ABenv, 1, 'range');
    
            % % % % Exclude bins 
            total_counts = sum(DbCounts); % Total number of values across all bins
            bin_percentages = (DbCounts / total_counts) * 100; % Percentage of values in each bin
            low_percentage_threshold = 5;           % Define a threshold for low percentage bins (e.g., less than 5%)
            low_percentage_bins = find(bin_percentages < low_percentage_threshold);        % Find the bins that have a percentage lower than the threshold
    
            %%%% Exclude the bins with low percentages
            % exclude if low_percentage_bins is not the end number
            ABenvRedBin = ABenv(:, low_percentage_bins(end)+1:end);
            ABenvNormRedBin = ABenvNorm(:, low_percentage_bins(end)+1:end);
    
            %%%%%%% Ready Features
            if isempty(ABenvRedBin)
                feature_names = {'EnvNorm', 'ABenv', 'ABenvNorm'};
                features = {EnvNorm, ABenv, ABenvNorm};

            else
                % Add the binned envelope to the features list
                feature_names = {'EnvNorm', 'ABenv', 'ABenvNorm', 'ABenvRedBin', 'ABenvNormRedBin'};
                features = {EnvNorm, ABenv, ABenvNorm, ABenvRedBin, ABenvNormRedBin};
    
            end


            % Model hyperparameters
            tmin = -100;
            tmax = 400;         % tmax = 500;
            trainfold = 10;
            testfold = 1;
            Dir = 1;            %specifies the forward modeling
    
            lambdas = linspace(10e-4, 10e4,10);
            % figure; col_num = length(features); % Dinamic Figure Columns
    
            for feature_idx = 1:length(features)
    
                stim = features{feature_idx};  % Assign the current feature set to stim
                fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set
                % Partition the data into training and test data segments
                [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);
    
                %%% z-score the input and output data
                rtrainz = cellfun(@(x) zscore(x, [], 'all'), rtrain, 'UniformOutput', false);
                rtestz = zscore(rtest, [], 'all');
    
                %%% Use cross-validation to find the optimal regularization parameter
                cv = mTRFcrossval(strain, rtrainz, fs_eeg, Dir, tmin, tmax, lambdas, 'Verbose', 0);
    
                % Get the optimal regression parameter
                l = mean(cv.r,3); %over channels
                [l_val,l_idx] = max(mean(l,1));
                l_opt = lambdas(l_idx);
    
                % Compute model weights
                model = mTRFtrain(strain, rtrainz, fs_eeg, Dir, tmin, tmax, l_opt, 'verbose', 0);
                % Test the model on unseen neural data
                [~, stats] = mTRFpredict(stest, rtestz, model, 'verbose', 0);
    
                % Store mean correlation values
                name_struct(s, k).(sprintf('%sModel', feature_names{feature_idx})) = model;
                name_struct(s, k).(sprintf('%sStats', feature_names{feature_idx})) = stats;    
            end
    
    
            disp ('Another!')
        end
    end

end


