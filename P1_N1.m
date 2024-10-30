% StatsParticipantTask_27_10
% close;clear;clc;
%% 
OT_setup
load("StatsParticipantTask_27_10.mat")

StatsParticipantTask_N1_P1 = struct();
%% 
% Define time windows for P1 and N1 peaks (in ms)
P1_window = [0, 130];  % ms
N1_window = [70, 210];  % ms

%%
% Loop over participants and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        Modelfeatures = {'standardEnvelopeModel'};
        Statsfeatures = {'standardEnvelopeStats'};

        for feature_idx = 1:length(Modelfeatures)
            Model = StatsParticipantTask_27_10(s, k).(Modelfeatures{feature_idx});
            Stats = StatsParticipantTask_27_10(s, k).(Statsfeatures{feature_idx});

            % Extract weight values
            t_vector = Model.t; 
            weights = Model.w;

            % Calculate mean weights by channel
            mean_weights = mean(weights, 3); % Mean over the third dimension (trials)

            % Find P1 peaks (positive peaks) across the entire time vector
            [Positive_peak_vals_all, Positive_peak_time_idx_all] = findpeaks(mean_weights); % Find all local maxima
            if ~isempty(Positive_peak_time_idx_all)
                % Get corresponding peak values and times
                Positive_peak_all = Positive_peak_vals_all; % All peak values
                Positive_peak_time_all = t_vector(Positive_peak_time_idx_all); % Corresponding times
            else
                Positive_peak_all = NaN;
                Positive_peak_time_all = NaN;
            end
            
            % Find N1 peaks (negative peaks) across the entire time vector
            [N1_peak_vals_all, N1_peak_time_idx_all] = findpeaks(-mean_weights); % Find all local minima by negating the weights
            if ~isempty(N1_peak_time_idx_all)
                % Get corresponding peak values and times
                Negative_peak_all = -N1_peak_vals_all; % All peak values (negated back)
                Negative_peak_time_all = t_vector(N1_peak_time_idx_all); % Corresponding times
            else
                Negative_peak_all = NaN;
                Negative_peak_time_all = NaN;
            end

            % Find first positive P1 peak after time 0
            valid_P1_indices = find(Positive_peak_time_all > 0 & Positive_peak_all > 0);
            if ~isempty(valid_P1_indices)
                P1_peak_value = Positive_peak_all(valid_P1_indices(1));
                P1_peak_time = Positive_peak_time_all(valid_P1_indices(1));
            else
                P1_peak_value = NaN;
                P1_peak_time = NaN;
            end
            
            % Find first positive N1 peak after time 0
            valid_N1_indices = find(Negative_peak_time_all > 0 & Negative_peak_all < 0);
            if ~isempty(valid_N1_indices)
                N1_peak_value = Negative_peak_all(valid_N1_indices(1));
                N1_peak_time = Negative_peak_time_all(valid_N1_indices(1));
            else
                N1_peak_value = NaN;
                N1_peak_time = NaN;
            end

            % % Plot mean weights by channel
            % figure;
            % subplot(2, 2, feature_idx * 2 - 1);
            % plot(t_vector, mean_weights, 'LineWidth', 2.0, 'Color', 'k', 'DisplayName', 'Mean all Channels'); % Plot mean weights for each channel
            % hold on
            % plot(t_vector, squeeze(Model.w(1, :, :)), 'Color', [0.5 0.5 0.5]); % Model weights
            % title(sprintf('Weights (%s)', Statsfeatures{feature_idx}));
            % ylabel('Weights (a.u.)'); xlabel('Time (ms)');
            % legend('Mean all Channels'); % Channel labels
            % 
            % % Mark all P1 Peaks
            % if ~isnan(Positive_peak_all)
            %     plot(first_P1_peak_time, first_P1_peak_value, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'P1 Peaks');
            % end
            % 
            % % Mark all N1 Peaks
            % if ~isnan(Negative_peak_all)
            %     plot(first_N1_peak_time, first_N1_peak_value, 'bo', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'N1 Peaks');
            % end
            % 
            % % GFP (Global Field Power)
            % subplot(2, 2, feature_idx * 2);
            % boxplot(Stats.r); ylim([-0.05 0.20]); % Channel correlation values
            % title(sprintf('GFP (%s)', Statsfeatures{feature_idx})); ylabel('correlation');

            % Store all peaks and the first valid peaks in the StatsParticipantTask_N1_P1 structure
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak', Modelfeatures{feature_idx})) = Positive_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak', Modelfeatures{feature_idx})) = Negative_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_time', Modelfeatures{feature_idx})) = Positive_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_time', Modelfeatures{feature_idx})) = Negative_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_value', Modelfeatures{feature_idx})) = P1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_time', Modelfeatures{feature_idx})) = P1_peak_time;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_value', Modelfeatures{feature_idx})) = N1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_time', Modelfeatures{feature_idx})) = N1_peak_time;
        end
    end
end


%%
% Loop over participants and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        Modelfeatures = {'standardEnvelopeModel'};
        Statsfeatures = {'standardEnvelopeStats'};

        for feature_idx = 1:length(Modelfeatures)
            Model = StatsParticipantTask_27_10(s, k).(Modelfeatures{feature_idx});
            Stats = StatsParticipantTask_27_10(s, k).(Statsfeatures{feature_idx});

            % Extract weight values
            t_vector = Model.t; 
            weights = Model.w;

            % Calculate mean weights by channel
            mean_weights = mean(weights, 3); % Mean over the third dimension (trials)

            % Find P1 peaks (positive peaks) within 0-130 ms
            P1_window = (t_vector >= 0 & t_vector <= 130); % Define time window for P1
            [P1_peak_vals_all, P1_peak_time_idx_all] = findpeaks(mean_weights(P1_window)); % Find positive peaks within window
            
            if ~isempty(P1_peak_time_idx_all)
                % Get corresponding peak values and times for P1 within window
                Positive_peak_all = P1_peak_vals_all; % All peak values within window
                Positive_peak_time_all = t_vector(P1_window); % Corresponding times within window
                Positive_peak_time_all = Positive_peak_time_all(P1_peak_time_idx_all); % Adjust to actual time points
            else
                Positive_peak_all = NaN;
                Positive_peak_time_all = NaN;
            end
            
            % Find N1 peaks (negative peaks) within 70-210 ms
            N1_window = (t_vector >= 70 & t_vector <= 210); % Define time window for N1
            [N1_peak_vals_all, N1_peak_time_idx_all] = findpeaks(-mean_weights(N1_window)); % Find negative peaks within window by negating weights
            
            if ~isempty(N1_peak_time_idx_all)
                % Get corresponding peak values and times for N1 within window
                Negative_peak_all = -N1_peak_vals_all; % Negate back to get true peak values
                Negative_peak_time_all = t_vector(N1_window); % Corresponding times within window
                Negative_peak_time_all = Negative_peak_time_all(N1_peak_time_idx_all); % Adjust to actual time points
            else
                Negative_peak_all = NaN;
                Negative_peak_time_all = NaN;
            end

            % Find first valid P1 peak within 0-130 ms
            valid_P1_indices = find(Positive_peak_time_all > 0);
            if ~isempty(valid_P1_indices)
                P1_peak_value = Positive_peak_all(valid_P1_indices(1));
                P1_peak_time = Positive_peak_time_all(valid_P1_indices(1));
            else
                P1_peak_value = NaN;
                P1_peak_time = NaN;
            end
            
            % Find first valid N1 peak within 70-210 ms
            valid_N1_indices = find(Negative_peak_time_all > 70);
            if ~isempty(valid_N1_indices)
                N1_peak_value = Negative_peak_all(valid_N1_indices(1));
                N1_peak_time = Negative_peak_time_all(valid_N1_indices(1));
            else
                N1_peak_value = NaN;
                N1_peak_time = NaN;
            end

            % Store all peaks and the first valid peaks in the StatsParticipantTask_N1_P1 structure
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak', Modelfeatures{feature_idx})) = Positive_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak', Modelfeatures{feature_idx})) = Negative_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_time', Modelfeatures{feature_idx})) = Positive_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_time', Modelfeatures{feature_idx})) = Negative_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_value', Modelfeatures{feature_idx})) = P1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_time', Modelfeatures{feature_idx})) = P1_peak_time;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_value', Modelfeatures{feature_idx})) = N1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_time', Modelfeatures{feature_idx})) = N1_peak_time;
        end
    end
end

%%
% Loop over participants and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        Modelfeatures = {'standardEnvelopeModel'};
        Statsfeatures = {'standardEnvelopeStats'};

        for feature_idx = 1:length(Modelfeatures)
            Model = StatsParticipantTask_27_10(s, k).(Modelfeatures{feature_idx});
            Stats = StatsParticipantTask_27_10(s, k).(Statsfeatures{feature_idx});

            % Extract weight values
            t_vector = Model.t; 
            weights = Model.w;

            % Calculate mean weights by channel
            mean_weights = mean(weights, 3); % Mean over the third dimension (trials)

            % Find P1 peaks (positive peaks) within 0-150 ms
            P1_window = (t_vector >= 0 & t_vector <= 150); % Define time window for P1
            [P1_peak_vals_all, P1_peak_time_idx_all] = findpeaks(mean_weights(P1_window)); % Find positive peaks within window
            
            if ~isempty(P1_peak_time_idx_all)
                % Get corresponding peak values and times for P1 within window
                Positive_peak_all = P1_peak_vals_all; % All peak values within window
                Positive_peak_time_all = t_vector(P1_window); % Corresponding times within window
                Positive_peak_time_all = Positive_peak_time_all(P1_peak_time_idx_all); % Adjust to actual time points
            else
                Positive_peak_all = NaN;
                Positive_peak_time_all = NaN;
            end
            
            % Find N1 peaks (negative peaks) within 50-210 ms
            N1_window = (t_vector >= 50 & t_vector <= 210); % Define time window for N1
            [N1_peak_vals_all, N1_peak_time_idx_all] = findpeaks(-mean_weights(N1_window)); % Find negative peaks within window by negating weights
            
            if ~isempty(N1_peak_time_idx_all)
                % Get corresponding peak values and times for N1 within window
                Negative_peak_all = -N1_peak_vals_all; % Negate back to get true peak values
                Negative_peak_time_all = t_vector(N1_window); % Corresponding times within window
                Negative_peak_time_all = Negative_peak_time_all(N1_peak_time_idx_all); % Adjust to actual time points
            else
                Negative_peak_all = NaN;
                Negative_peak_time_all = NaN;
            end

            % Find first valid P1 peak within 0-150 ms
            valid_P1_indices = find(max(Positive_peak_all));
            if ~isempty(valid_P1_indices)
                P1_peak_value = Positive_peak_all(valid_P1_indices(1));
                P1_peak_time = Positive_peak_time_all(valid_P1_indices(1));
            else
                P1_peak_value = NaN;
                P1_peak_time = NaN;
            end
            
            % Find first valid N1 peak within 50-210 ms
            valid_N1_indices = find(min(Negative_peak_all));

            if ~isempty(valid_N1_indices)
                N1_peak_value = Negative_peak_all(valid_N1_indices(1));
                N1_peak_time = Negative_peak_time_all(valid_N1_indices(1));
            else
                N1_peak_value = NaN;
                N1_peak_time = NaN;
            end

            % Store all peaks and the first valid peaks in the StatsParticipantTask_N1_P1 structure
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak', Modelfeatures{feature_idx})) = Positive_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak', Modelfeatures{feature_idx})) = Negative_peak_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_time', Modelfeatures{feature_idx})) = Positive_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_time', Modelfeatures{feature_idx})) = Negative_peak_time_all;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_value', Modelfeatures{feature_idx})) = P1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_time', Modelfeatures{feature_idx})) = P1_peak_time;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_value', Modelfeatures{feature_idx})) = N1_peak_value;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_time', Modelfeatures{feature_idx})) = N1_peak_time;

            % Plot mean weights by channel
            figure;
            subplot(2, 1, 1);
            hold on;
            fill([50 50 210 210], [-30 30 30 -30], [0.6 0.6 1], 'EdgeColor', 'none', 'FaceAlpha', 0.4); % N1 time window
            fill([0 0 150 150], [-30 30 30 -30], [1 0.6 0.6], 'EdgeColor', 'none', 'FaceAlpha', 0.2); % P1 time window

            red_color = [1 0.1 0.1];
            blue_color = [0.1 0.1 1];
            xline(0, Color=red_color)
            xline(150, Color=red_color)

            xline(50, Color=blue_color)
            xline(210, Color=blue_color)

            subplot(2, 1, 1);
            % Plot P1 and N1 time windows as shaded regions
            hold on;
            plot(t_vector, mean_weights, 'LineWidth', 2.0, 'Color', 'k', 'DisplayName', 'Mean all Channels'); % Plot mean weights for each channel
            plot(t_vector, squeeze(Model.w(1, :, :)), 'Color', [0.5 0.5 0.5]); % Plot individual channels
            title(sprintf('Mean Weights (%s)', Statsfeatures{feature_idx}));
            ylabel('Weights (a.u.)'); xlabel('Time (ms)');
            legend('Mean all Channels'); % Channel labels
            % Mark first P1 and N1 Peaks
            if ~isnan(P1_peak_value)
                plot(P1_peak_time, P1_peak_value, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'First P1 Peak');
            end

            if ~isnan(N1_peak_value)
                plot(N1_peak_time, N1_peak_value, 'bo', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'First N1 Peak');
            end
            legend;

            % % GFP (Global Field Power)
            % subplot(2, 1, 2);
            % boxplot(Stats.r); ylim([-0.05 0.20]); % Channel correlation values
            % title(sprintf('GFP (%s)', Statsfeatures{feature_idx})); ylabel('Correlation')

        end
    end
end

%%
% Initialize arrays for magnitudes and latencies of the first P1 and N1 peaks
P1_Peaks_magnitude_narrow = zeros(length(sbj), 1);
% N1_Peaks_magnitude_narrow = zeros(length(sbj), 1);
P1_Peaks_latency_narrow = zeros(length(sbj), 1);
N1_Peaks_latency_narrow = zeros(length(sbj), 1);

P1_Peaks_magnitude_wide = zeros(length(sbj), 1);
N1_Peaks_magnitude_wide = zeros(length(sbj), 1);
P1_Peaks_latency_wide = zeros(length(sbj), 1);
N1_Peaks_latency_wide = zeros(length(sbj), 1);

% Loop over participants to extract the first P1 and N1 peak magnitudes and latencies
for s = 1:length(sbj)
    % Narrow task
    P1_Peaks_magnitude_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_first_P1Peak_value;
    N1_Peaks_magnitude_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_first_N1Peak_value;
    P1_Peaks_latency_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_first_P1Peak_time;
    N1_Peaks_latency_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_first_N1Peak_time;

    % Wide task
    P1_Peaks_magnitude_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_first_P1Peak_value;
    N1_Peaks_magnitude_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_first_N1Peak_value;
    P1_Peaks_latency_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_first_P1Peak_time;
    N1_Peaks_latency_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_first_N1Peak_time;
end

%% Step 4: Plotting

% Create a figure for the magnitudes
figure;

% P1 Magnitudes
subplot(2, 1, 1); % 2 rows, 1 column, 1st plot
hold on;
plot(1:length(sbj), P1_Peaks_magnitude_narrow, 'ro-', 'DisplayName', 'P1 Magnitude Narrow');
plot(1:length(sbj), P1_Peaks_magnitude_wide, 'bo-', 'DisplayName', 'P1 Magnitude Wide');
ylabel('P1 Magnitude (a.u.)');
xlabel('Participants');
title('P1 Magnitude by Participant');
legend();
grid on;

% N1 Magnitudes
subplot(2, 1, 2); % 2 rows, 1 column, 2nd plot
hold on;
plot(1:length(sbj), N1_Peaks_magnitude_narrow, 'ro-', 'DisplayName', 'N1 Magnitude Narrow');
plot(1:length(sbj), N1_Peaks_magnitude_wide, 'bo-', 'DisplayName', 'N1 Magnitude Wide');
ylabel('N1 Magnitude (a.u.)');
xlabel('Participants');
title('N1 Magnitude by Participant');
legend();
grid on;

% Create a new figure for the latencies
figure;

% P1 Latencies
subplot(2, 1, 1); % 2 rows, 1 column, 1st plot
hold on;
plot(1:length(sbj), P1_Peaks_latency_narrow, 'ro-', 'DisplayName', 'P1 Latency Narrow');
plot(1:length(sbj), P1_Peaks_latency_wide, 'bo-', 'DisplayName', 'P1 Latency Wide');
ylabel('P1 Latency (ms)');
xlabel('Participants');
title('P1 Latency by Participant');
legend();
grid on;

% N1 Latencies
subplot(2, 1, 2); % 2 rows, 1 column, 2nd plot
hold on;
plot(1:length(sbj), N1_Peaks_latency_narrow, 'ro-', 'DisplayName', 'N1 Latency Narrow');
plot(1:length(sbj), N1_Peaks_latency_wide, 'bo-', 'DisplayName', 'N1 Latency Wide');
ylabel('N1 Latency (ms)');
xlabel('Participants');
title('N1 Latency by Participant');
legend();
grid on;

%% Step 5: Plotting P1 - N1 Differences
figure;
hold on;
% Calculate P1 - N1 differences for both narrow and wide tasks
P1_N1_diff_narrow = P1_Peaks_magnitude_narrow - N1_Peaks_magnitude_narrow;
P1_N1_diff_wide = P1_Peaks_magnitude_wide - N1_Peaks_magnitude_wide;

% Plotting differences
plot(1:length(sbj), P1_N1_diff_narrow, 'ro-', 'DisplayName', 'P1 - N1 Difference Narrow');
plot(1:length(sbj), P1_N1_diff_wide, 'bo-', 'DisplayName', 'P1 - N1 Difference Wide');
ylabel('P1 - N1 Difference (a.u.)');
xlabel('Participants');
title('P1 - N1 Magnitude Difference by Participant');
legend();
grid on;



%%
%% Step 1: Calculate Mean Magnitudes and Latencies across Participants

% Calculate mean values while ignoring NaNs
mean_P1_magnitude_narrow = nanmean(P1_Peaks_magnitude_narrow);
mean_N1_magnitude_narrow = nanmean(N1_Peaks_magnitude_narrow);
mean_P1_latency_narrow = nanmean(P1_Peaks_latency_narrow);
mean_N1_latency_narrow = nanmean(N1_Peaks_latency_narrow);

mean_P1_magnitude_wide = nanmean(P1_Peaks_magnitude_wide);
mean_N1_magnitude_wide = nanmean(N1_Peaks_magnitude_wide);
mean_P1_latency_wide = nanmean(P1_Peaks_latency_wide);
mean_N1_latency_wide = nanmean(N1_Peaks_latency_wide);

% Display results
disp(['Mean P1 Magnitude (Narrow): ', num2str(mean_P1_magnitude_narrow)]);
disp(['Mean N1 Magnitude (Narrow): ', num2str(mean_N1_magnitude_narrow)]);
disp(['Mean P1 Latency (Narrow): ', num2str(mean_P1_latency_narrow)]);
disp(['Mean N1 Latency (Narrow): ', num2str(mean_N1_latency_narrow)]);

disp(['Mean P1 Magnitude (Wide): ', num2str(mean_P1_magnitude_wide)]);
disp(['Mean N1 Magnitude (Wide): ', num2str(mean_N1_magnitude_wide)]);
disp(['Mean P1 Latency (Wide): ', num2str(mean_P1_latency_wide)]);
disp(['Mean N1 Latency (Wide): ', num2str(mean_N1_latency_wide)]);


%% Step 2: Group-Averaged AB Envelope TRF Plot

% Assuming AB envelope TRF data is stored in `groupAvg_ABenv_TRF`
% Adjust for participant trace differences

figure;
plot(groupAvg_ABenv_TRF, 'LineWidth', 1.5);
ylabel('Amplitude (a.u.)');
xlabel('Time (ms)');
title('Group-Averaged AB Envelope TRF');
grid on;

%% Step 3: N1 Peak Latencies and P1-N1 Peak-to-Peak Amplitudes across Bins

% Assuming `AB_env_bins` contains bin values for the AB envelope TRF

% Calculate N1 peak latencies and P1-N1 peak-to-peak amplitudes across bins
N1_peak_latencies_bins = mean(reshape(N1_Peaks_latency_wide, [], length(AB_env_bins)), 1);
P1_N1_peak_to_peak_amplitude_bins = mean(reshape(P1_Peaks_magnitude_wide - N1_Peaks_magnitude_wide, [], length(AB_env_bins)), 1);

% Plot N1 Peak Latencies across bins
figure;
subplot(2, 1, 1);
plot(AB_env_bins, N1_peak_latencies_bins, 'bo-', 'LineWidth', 1.5);
ylabel('N1 Latency (ms)');
xlabel('AB Envelope TRF Bins');
title('N1 Peak Latencies across AB Envelope TRF Bins');
grid on;

% Plot P1-N1 Peak-to-Peak Amplitudes across bins
subplot(2, 1, 2);
plot(AB_env_bins, P1_N1_peak_to_peak_amplitude_bins, 'ro-', 'LineWidth', 1.5);
ylabel('P1-N1 Peak-to-Peak Amplitude (a.u.)');
xlabel('AB Envelope TRF Bins');
title('P1-N1 Peak-to-Peak Amplitudes across AB Envelope TRF Bins');
grid on;




%% AB env wieght plot

%% AB env weight plot with bin-specific averaging

for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        Modelfeatures = {'ABenvNormModel'};
        Statsfeatures = {'ABenvNormStats'};

        for feature_idx = 1:length(Modelfeatures)
            Model = StatsParticipantTask_27_10(s, k).(Modelfeatures{feature_idx});
            Stats = StatsParticipantTask_27_10(s, k).(Statsfeatures{feature_idx});

            % Extract weight values
            t_vector = Model.t; 
            weights = Model.w; % Dimensions are now (8, nChannels, nSamples)

            % Calculate mean weights by averaging over channels
            mean_weights_bins = squeeze(mean(weights, 2)); % Average over 2nd dimension (channels) to keep bins and time
            
            % Now, mean_weights_bins should have dimensions (8, nSamples)

            % Find P1 and N1 peaks for each bin separately
            for bin_idx = 1:size(mean_weights_bins, 1)
                % Extract the time series for the current bin
                bin_weights = mean_weights_bins(bin_idx, :);

                % Find P1 peaks (positive peaks) across the entire time vector
                [Positive_peak_vals_all, Positive_peak_time_idx_all] = findpeaks(bin_weights); % Find all local maxima
                if ~isempty(Positive_peak_time_idx_all)
                    % Get corresponding peak values and times
                    Positive_peak_all = Positive_peak_vals_all; % All peak values
                    Positive_peak_time_all = t_vector(Positive_peak_time_idx_all); % Corresponding times
                else
                    Positive_peak_all = NaN;
                    Positive_peak_time_all = NaN;
                end

                % Find N1 peaks (negative peaks) across the entire time vector
                [N1_peak_vals_all, N1_peak_time_idx_all] = findpeaks(-bin_weights); % Find all local minima by negating the weights
                if ~isempty(N1_peak_time_idx_all)
                    % Get corresponding peak values and times
                    Negative_peak_all = -N1_peak_vals_all; % All peak values (negated back)
                    Negative_peak_time_all = t_vector(N1_peak_time_idx_all); % Corresponding times
                else
                    Negative_peak_all = NaN;
                    Negative_peak_time_all = NaN;
                end

                % Find first positive P1 peak after time 0
                valid_P1_indices = find(Positive_peak_time_all > 0 & Positive_peak_all > 0);
                if ~isempty(valid_P1_indices)
                    P1_peak_value = Positive_peak_all(valid_P1_indices(1));
                    P1_peak_time = Positive_peak_time_all(valid_P1_indices(1));
                else
                    P1_peak_value = NaN;
                    P1_peak_time = NaN;
                end

                % Find first positive N1 peak after time 0
                valid_N1_indices = find(Negative_peak_time_all > 0 & Negative_peak_all < 0);
                if ~isempty(valid_N1_indices)
                    N1_peak_value = Negative_peak_all(valid_N1_indices(1));
                    N1_peak_time = Negative_peak_time_all(valid_N1_indices(1));
                else
                    N1_peak_value = NaN;
                    N1_peak_time = NaN;
                end

                % Store peaks for each bin in StatsParticipantTask_N1_P1 structure
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_bin%d', Modelfeatures{feature_idx}, bin_idx)) = Positive_peak_all;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_bin%d', Modelfeatures{feature_idx}, bin_idx)) = Negative_peak_all;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_time_bin%d', Modelfeatures{feature_idx}, bin_idx)) = Positive_peak_time_all;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_time_bin%d', Modelfeatures{feature_idx}, bin_idx)) = Negative_peak_time_all;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_value_bin%d', Modelfeatures{feature_idx}, bin_idx)) = P1_peak_value;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_P1Peak_time_bin%d', Modelfeatures{feature_idx}, bin_idx)) = P1_peak_time;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_value_bin%d', Modelfeatures{feature_idx}, bin_idx)) = N1_peak_value;
                StatsParticipantTask_N1_P1(s, k).(sprintf('%s_first_N1Peak_time_bin%d', Modelfeatures{feature_idx}, bin_idx)) = N1_peak_time;
            end
        end
    end
end
%%



%%
%% Plot 8 bins weight of each bin

for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        Modelfeatures = {'ABenvNormModel'};
        
        for feature_idx = 1:length(Modelfeatures)
            Model = StatsParticipantTask_27_10(s, k).(Modelfeatures{feature_idx});
            
            % Extract time vector and mean weights across channels for each bin
            t_vector = Model.t;
            weights = Model.w; % Dimensions (8, nChannels, nSamples)
            mean_weights_bins = squeeze(mean(weights, 3)); % Average over channels, resulting in (8, nSamples)

            % Plot each bin separately
            figure;
            hold on;
            colors = lines(8); % Generate distinct colors for each bin

            step_size = 1/8;
            for bin_idx = 1:8
                plot(t_vector, mean_weights_bins(bin_idx, :)*step_size+bin_idx, 'LineWidth', 1.5, 'Color', colors(bin_idx, :), ...
                     'DisplayName', sprintf('Bin %d', bin_idx));
            end

            title(sprintf('Mean Weights for Each Bin (Participant %d, Task %d)', s, k));
            xlabel('Time (ms)');
            ylabel('Mean Weight (a.u.)');
            legend show;
            hold off;
        end
    end
end
