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

% Loop over participants and tasks
for s = 1:length(sbj)
    for k = 1:length(task)
        % Process each feature
        % Modelfeatures = {'standardEnvelopeModel', 'ABenvNormModel'};
        % Statsfeatures = {'standardEnvelopeStats', 'ABenvNormStats'};
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

            % % Find P1 peak (maximum) in the P1 window
            % P1_idx = t_vector >= P1_window(1) & t_vector <= P1_window(2);
            % [P1_peak, P1_peak_time_idx] = max(mean_weights(:, P1_idx), [], 2); % Max within P1 window
            % 
            % P1_peak_time = t_vector(P1_idx);
            % P1_peak_time = P1_peak_time(P1_peak_time_idx); % Get corresponding time
            % 
            % % Find N1 peak (minimum) in the N1 window
            % N1_idx = t_vector >= N1_window(1) & t_vector <= N1_window(2);
            % [N1_peak, N1_peak_time_idx] = min(mean_weights(:, N1_idx), [], 2); % Min within N1 window
            % N1_peak_time = t_vector(N1_idx);
            % N1_peak_time = N1_peak_time(N1_peak_time_idx); % Get corresponding time

            % Find P1 peak (positive peak) in the P1 window
            P1_idx = t_vector >= P1_window(1) & t_vector <= P1_window(2);
            [~, P1_peak_time_idx] = findpeaks(mean_weights(:, P1_idx), 'MinPeakHeight', 0); % Find local maxima
            if ~isempty(P1_peak_time_idx)
                P1_peak = mean_weights(:, P1_idx);
                P1_peak = P1_peak(P1_peak_time_idx); % Get corresponding peak values
                P1_peak_time = t_vector(P1_idx);
                P1_peak_time = P1_peak_time(P1_peak_time_idx); % Get corresponding time
            else
                P1_peak = NaN;
                P1_peak_time = NaN;
            end
            
            % Find N1 peak (negative peak) in the N1 window
            N1_idx = t_vector >= N1_window(1) & t_vector <= N1_window(2);
            [~, N1_peak_time_idx] = findpeaks(-mean_weights(:, N1_idx), 'MinPeakHeight', 0); % Find local minima by negating the weights
            if ~isempty(N1_peak_time_idx)
                N1_peak = mean_weights(:, N1_idx);
                N1_peak = -N1_peak(N1_peak_time_idx); % Get corresponding negative peak values
                N1_peak_time = t_vector(N1_idx);
                N1_peak_time = N1_peak_time(N1_peak_time_idx); % Get corresponding time
            else
                N1_peak = NaN;
                N1_peak_time = NaN;
            end

            % Plot mean weights by channel
            figure;
            subplot(2, 2, feature_idx * 2 - 1);
            plot(t_vector, mean_weights, 'LineWidth', 2.0, 'Color', 'k', 'DisplayName', 'Mean all Channels'); % Plot mean weights for each channel
            hold on
            plot(t_vector, squeeze(Model.w(1, :, :)), 'Color', [0.5 0.5 0.5]); % Model weights
            title(sprintf('Weights (%s)', Statsfeatures{feature_idx}));
            ylabel('Weights (a.u.)'); xlabel('Time (ms)');
            legend('Mean all Channels'); % Channel labels

            % Mark P1 Peak
            if ~isnan(P1_peak)
                plot(P1_peak_time, P1_peak, 'ro', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'P1 Peak');
            end

            % Mark N1 Peak
            if ~isnan(N1_peak)
                plot(N1_peak_time, N1_peak, 'bo', 'MarkerSize', 8, 'LineWidth', 2, 'DisplayName', 'N1 Peak');
            end

            % GFP (Global Field Power)
            subplot(2, 2, feature_idx * 2);
            boxplot(Stats.r); ylim([-0.05 0.20]); % Channel correlation values
            title(sprintf('GFP (%s)', Statsfeatures{feature_idx})); ylabel('correlation');

            % Store the P1 and N1 peaks in the StatsParticipantTask structure
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak', Modelfeatures{feature_idx})) = P1_peak;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak', Modelfeatures{feature_idx})) = N1_peak;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_P1Peak_time', Modelfeatures{feature_idx})) = P1_peak_time;
            StatsParticipantTask_N1_P1(s, k).(sprintf('%s_N1Peak_time', Modelfeatures{feature_idx})) = N1_peak_time;
        end
    end
end


%%

P1_Peaks_all_participants_narrow = zeros(20,1);
N1_Peaks_all_participants_Narrow = zeros(20,1);
P1_Peaks_all_participants_wide = zeros(20,1);
N1_Peaks_all_participants_wide = zeros(20,1);


% Loop over participants and tasks
for s = 1:length(sbj)

P1_Peaks_all_participants_narrow(s) = [StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_P1Peak];
N1_Peaks_all_participants_Narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_N1Peak;

P1_Peaks_all_participants_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_P1Peak;
N1_Peaks_all_participants_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_N1Peak;

end



magnetude and latency
plot n values x is participants
plot p values x is participants

plot p1-n1 

%%
% Initialize arrays for magnitudes and latencies
P1_Peaks_magnitude_narrow = zeros(length(sbj), 1);
N1_Peaks_magnitude_narrow = zeros(length(sbj), 1);
P1_Peaks_latency_narrow = zeros(length(sbj), 1);
N1_Peaks_latency_narrow = zeros(length(sbj), 1);

P1_Peaks_magnitude_wide = zeros(length(sbj), 1);
N1_Peaks_magnitude_wide = zeros(length(sbj), 1);
P1_Peaks_latency_wide = zeros(length(sbj), 1);
N1_Peaks_latency_wide = zeros(length(sbj), 1);

% Loop over participants and tasks to extract magnitudes and latencies
for s = 1:length(sbj)
    % Narrow task
    P1_Peaks_magnitude_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_P1Peak;
    N1_Peaks_magnitude_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_N1Peak;
    P1_Peaks_latency_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_P1Peak_time;
    N1_Peaks_latency_narrow(s) = StatsParticipantTask_N1_P1(s, 1).standardEnvelopeModel_N1Peak_time;

    % Wide task
    P1_Peaks_magnitude_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_P1Peak;
    N1_Peaks_magnitude_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_N1Peak;
    P1_Peaks_latency_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_P1Peak_time;
    N1_Peaks_latency_wide(s) = StatsParticipantTask_N1_P1(s, 2).standardEnvelopeModel_N1Peak_time;
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
