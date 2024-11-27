function EnvelopePlots()

%%
clc
%% 
OT_setup
s=20; k=2;
fprintf('Plotting the subject: %s \nTask: %s\n', sbj{s}, task{k}); % Print Subject and Task
[fs_eeg, full_resp, fs_audio, audio_dat, ~]  = LoadEEG(s, k, sbj, task);
[EnvNorm, ABenvNorm, ~, onsetEnvelope, ~, binEdges_dB] = feature_deriv(audio_dat, fs_audio, fs_eeg, full_resp);
numBins = size(ABenvNorm, 2);                                   % Number of dB bins

%%
% norm_onsetEnvelope = normalize(onsetEnvelope,1,'range');

norm_audio_dat = normalize(audio_dat,2,"range",[-1 1]);
time_audio = (0:length(audio_dat) - 1) / fs_audio;  % Time array for original audio
time_env = (0:length(EnvNorm) - 1) / fs_eeg;   % Time array for envelope and binned envelope

% Extra parameters
window_time = [1149 1151];
step_size=0.98/8;
multiplystep_size = step_size*0.9;

% Colors
blue = [0 0.4470 0.7410];
orange = [0.8500 0.3250 0.0980];
yellow = [0.9290 0.6940 0.1250];

% Plot Binned Envelopes
figure;
subplot(3,1,1)
plot(time_audio, norm_audio_dat, 'color', 'k', 'LineWidth', 1, 'DisplayName', 'Original Audio (normalized)');  % Original audio with thicker line
xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
grid on; hold off; box off
legend('Location', 'best');  % Add legend
set(gca, 'FontSize', 13);

title('Original Audio');

subplot(3,1,2)
hold on
plot(time_env, EnvNorm, 'color', blue, 'LineWidth', 2, 'DisplayName', 'mTRFenvelope (normalized)');  % Standard envelope with thicker line
plot(time_env, onsetEnvelope, 'LineWidth', 2,'color',yellow, 'DisplayName', 'Onset Envelope (normalized)');  % Onset envelope

xlim(window_time);
xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);

box off; grid on; hold off;
legend('Location', 'best');  % Add legend
set(gca, 'FontSize', 13);

title('Envelope models');

subplot(3,1,3)
for i = 1:numBins
    plot_norm_binnedEnvelopes_dB = (ABenvNorm(:, i)*multiplystep_size) + binEdges_dB(i);
    hold on
    plot(time_env, plot_norm_binnedEnvelopes_dB, 'LineWidth', 2);
end

title('Binned Envelope');
xlabel('Time (s)'); ylabel('Amplitude(a.u)'); xlim(window_time);
grid on;  % Add grid for better visibility
box off
xlim(window_time);
set(gca, 'FontSize', 13);