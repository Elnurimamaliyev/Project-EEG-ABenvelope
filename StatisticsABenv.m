%%% plot that uses pearson r of normal envelope model and ab envelope over each subject
clear
OT_setup
addpath("C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data")

load('C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-Data\StatsParticipantTask.mat');


%%
% Perform Shapiro-Wilk test for Normal Envelopes
[h_sw_NormEnv, p_sw_NormEnv, stats_sw_NormEnv] = swtest(narrow_norm_env_r_median);
disp(['Participant ' num2str(s) ', Condition ' task{k} ': Shapiro-Wilk Test (NormEnv) p-value = ' num2str(p_sw_NormEnv)]);
        
% narrow_norm_env_r_median

% narrow_ab_env_r_median
%%
figure;
histogram(narrow_ab_env_r_median,5);
title(['Histogram of Correlation Values (NormEnv) - Participant ' num2str(s) ', Condition ' task{k}]);
xlabel('Correlation');
ylabel('x');



%% Statistics - Normality test

for s = 1:length(sbj)  
    for k = 1:length(task)  
        % Extract correlation values for Normal Envelopes and AB Envelopes
        corr_values_NormEnv = StatsParticipantTask(s, k).NormEnvStats.r;  % Assuming this structure exists
        corr_values_ABenv = StatsParticipantTask(s, k).ABenvStats.r;

        % Create a new figure for each participant and condition
        figure;

        % Left column: Histogram for Normal Envelopes
        subplot(2, 2, 1);  % 2 rows, 2 columns, 1st plot
        histogram(corr_values_NormEnv);
        title(['Histogram of Correlation Values (NormEnv) - Participant ' num2str(s) ', Condition ' task{k}]);
        xlabel('Correlation');
        ylabel('Frequency');

        % Right column: Histogram for AB Envelopes
        subplot(2, 2, 2);  % 2 rows, 2 columns, 3rd plot
        histogram(corr_values_ABenv);
        title(['Histogram of Correlation Values (ABenv) - Participant ' num2str(s) ', Condition ' task{k}]);
        xlabel('Correlation');
        ylabel('Frequency');

        % Left column: Q-Q plot for Normal Envelopes
        subplot(2, 2, 3);  % 2 rows, 2 columns, 2nd plot
        qqplot(corr_values_NormEnv);
        title(['Q-Q Plot of Correlation Values (NormEnv) - Participant ' num2str(s) ', Condition ' task{k}]);

        % Right column: Q-Q plot for AB Envelopes
        subplot(2, 2, 4);  % 2 rows, 2 columns, 4th plot
        qqplot(corr_values_ABenv);
        title(['Q-Q Plot of Correlation Values (ABenv) - Participant ' num2str(s) ', Condition ' task{k}]);

        % Perform Shapiro-Wilk test for Normal Envelopes
        [h_sw_NormEnv, p_sw_NormEnv, stats_sw_NormEnv] = swtest(corr_values_NormEnv);
        disp(['Participant ' num2str(s) ', Condition ' task{k} ': Shapiro-Wilk Test (NormEnv) p-value = ' num2str(p_sw_NormEnv)]);
        
        % % Perform Kolmogorov-Smirnov test for Normal Envelopes
        % [h_ks_NormEnv, p_ks_NormEnv, stats_ks_NormEnv] = kstest(corr_values_NormEnv);
        % disp(['Participant ' num2str(s) ', Condition ' task{k} ': Kolmogorov-Smirnov Test (NormEnv) p-value = ' num2str(p_ks_NormEnv)]);
        
        % Perform Shapiro-Wilk test for AB Envelopes
        [h_sw_ABenv, p_sw_ABenv, stats_sw_ABenv] = swtest(corr_values_ABenv);
        disp(['Participant ' num2str(s) ', Condition ' task{k} ': Shapiro-Wilk Test (ABenv) p-value = ' num2str(p_sw_ABenv)]);
        
        % % Perform Kolmogorov-Smirnov test for AB Envelopes
        % [h_ks_ABenv, p_ks_ABenv, stats_ks_ABenv] = kstest(corr_values_ABenv);
        % disp(['Participant ' num2str(s) ', Condition ' task{k} ': Kolmogorov-Smirnov Test (ABenv) p-value = ' num2str(p_ks_ABenv)]);
    end
end
%%

for s = 1:length(sbj)  
    for k = 1:length(task)  
        % Extract correlation values for Normal Envelopes and AB Envelopes
        corr_values_NormEnv = StatsParticipantTask(s, k).NormEnvStats.r;  % Assuming this structure exists
        corr_values_ABenv = StatsParticipantTask(s, k).ABenvStats.r;
        
        % Perform Wilcoxon Signed-Rank Test
        [p, h, stats] = signrank(corr_values_NormEnv, corr_values_ABenv);
        
        % Display the results
        fprintf('p-value: %.4f\n', p);
        fprintf('Test Statistic (z): %.4f\n', stats.zval);
        if h == 0
            fprintf('No significant difference (fail to reject H0).\n');
        else
            fprintf('Significant difference (reject H0).\n');
        end


    end 
end


%% Statistics -  test

for s = 1:length(sbj)
    for k = 1:length(task)
        % Extract correlation values for Normal Envelopes and AB Envelopes
        corr_values_NormEnv = StatsParticipantTask(s, k).NormEnvStats.r; 
        corr_values_ABenv = StatsParticipantTask(s, k).ABenvStats.r;

        % Perform paired t-test
        [h_ttest, p_ttest] = ttest(corr_values_ABenv, corr_values_NormEnv);

        % Display results
        disp(['Participant ' num2str(s) ', Condition ' task{k} ': t-test p-value = ' num2str(p_ttest)]);
    end
end
%%
% Initialize arrays to store mean correlation values for each condition
for s = 1:length(sbj)
    for k = 1:length(task)
        % Extract correlation values for Normal Envelopes and AB Envelopes
        corr_values_NormEnv = StatsParticipantTask(s, k).NormEnvStats.r; 
        corr_values_ABenv = StatsParticipantTask(s, k).ABenvStats.r;

        % Perform paired t-test
        [h_ttest, p_ttest] = ttest(corr_values_ABenv, corr_values_NormEnv);

        % Display results
        disp(['Participant ' num2str(s) ', Condition ' task{k} ':']);
        disp(['   t-test p-value = ' num2str(p_ttest)]);
        
        % Check significance at alpha level of 0.05
        if p_ttest < 0.01
            disp('   Result: Significant difference (reject H0)');
        else
            disp('   Result: No significant difference (fail to reject H0)');
        end
    end
end


%% Plotting with zoom-in for correlation
% Extract and create time arrays
time_array = (0:wav.audio_strct.pnts-1) / wav.audio_strct.srate;  % Original time array (seconds)
time_array_stim = (0:length(stim_Env) - 1) / fs_eeg;  % Time array for the envelope (seconds)
%% Define the zoom-in range (in seconds)
start_window = 226;  % Start time in seconds
end_window = 242;     % End time in seconds
% Define the zoom-in range (in seconds)
% start_window = 233.5;  % Start time in seconds
% end_window = 235.5;     % End time in seconds
%% For plotting populate the matrix_bin with amplitude values
bins= 0:num_bins-1;
bins = bins(:)*8;

plot_stim_ABenv = repmat(bins, 1, length(stim_ABenv)); 
% plot_stim_ABenv = repmat(bins(:)*8, 1, length(stim_ABenv)); 
plot_stim_ABenv = plot_stim_ABenv' + stim_ABenv*4;

figure;

subplot(4,1,1);
plot(time_array_stim, resp)
title('EEG')
xlabel('Time (s)');
ylabel('Voltage (μV)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,2);
plot(time_array, audio_dat)
title('Original Sound')
xlabel('Time (s)');
ylabel('Amplitude (a.u)')
xlim([start_window end_window]);  % Zoom in on the original sound

subplot(4,1,3);
plot(time_array_stim, stim_Env)
title('Envelope')
xlabel('Time (s)');
ylabel('Amplitude (a.u)')
xlim([start_window end_window]);  % Zoom in on the envelope
hold on

% Mark onsets on the envelope plot for better visualization
subplot(4,1,3);
onset_times = find(stim_Onset == 1) / fs_eeg;
onset_indices = round(onset_times * fs_eeg); % Round to nearest integer
stem(onset_times, stim_Env(onset_indices), 'r');
hold off;
legend('Env', 'Ons');

subplot(4,1,4);
% plot((1:length(stim_binned))/fs_eeg, stim_binned_matrix);
plot((1:length(plot_stim_ABenv))/fs_eeg, plot_stim_ABenv);
title('AB Envelope');
xlabel('Time (s)');
ylabel('Binned stim');
xlim([start_window end_window]);
%% Next
%%% Histogram Distribution 
addpath 'C:\Users\icbmadmin\Documents\GitLabRep\Project-EEG-ABenvelope\Thorge\AB_envelope'
% AB_envelope


%% Will not be used probably
% UnusedFeatures % Combine other features (e.g., )



