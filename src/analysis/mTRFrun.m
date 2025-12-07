function [stats, model, strain, rtrain, stest, rtest] = mTRFrun(features, feature_names, resp, fs_eeg, tmin, tmax, testfold, trainfold)
    % Using different features to get Accuracies
    figure; col_num = length(features); % Dinamic Figure Columns
        
    for feature_idx = 1:length(features)
        
        stim = features{feature_idx};  % Assign the current feature set to stim
        fprintf('Processing feature set: %s\n', feature_names{feature_idx});   % Optionally, display the name of the current feature set
        % Partition the data into training and test data segments
        [strain, rtrain, stest, rtest] = mTRFpartition(stim, resp, trainfold, testfold);
        
        % Compute model weights
        model = mTRFtrain(strain, rtrain, fs_eeg, 1, tmin, tmax, 0.05, 'verbose', 0);
        % Test the model on unseen neural data
        [~, stats] = mTRFpredict(stest, rtest, model, 'verbose', 0);
    
        % Plotting in a col_num x 2 grid
    
        % Model weights
        subplot(col_num, 2, feature_idx * 2 - 1)
        plot(model.t, squeeze(model.w(1, :, :))); % model weights
        title(sprintf('Weights (%s)', feature_names{feature_idx}))
        ylabel('a.u.')
        xlabel('time in ms.')
    
        % GFP (Global Field Power)
        subplot(col_num, 2, feature_idx * 2)
        boxplot(stats.r) % channel correlation values
        title(sprintf('GFP (%s)', feature_names{feature_idx}))
        ylabel('correlation')
    end
    
    % Adjust the layout
    sgtitle('Feature Comparisons') % Add a main title to the figure

end