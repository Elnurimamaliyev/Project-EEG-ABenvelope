function [mEnv, mOnsetEnv] = EnvelopeGenerators(audio_dat, resp, fs_audio, fs_eeg)
    %%% Envelope Generation

    % We extract the envelope
    mEnv = mTRFenvelope(double(audio_dat),fs_audio,fs_eeg); % in order to use the mTRF toolbox the eeg data and stimulus need to be the same length 

    % Assign to the stimulus variable
    stim_Env = mEnv;
    
    % Are stim and resp the same length?
    stim_Env = stim_Env(1:size(resp,2),:);
    
    %%% Onset Generator
    % Resample the audio data to match the EEG sampling rate if not already done
    % if fs_audio ~= fs_eeg
    %     audio_dat_res = resample(audio_dat, fs_eeg, fs_audio);
    % end
    
    % Threshold-based peak detection on the envelope
    % Adjust the threshold and minimum peak distance based on your data
    threshold = 0.3; % Set a threshold for peak detection
    min_peak_distance = 0.15 * fs_eeg; % Minimum distance between peaks in samples
    [~, onset_locs] = findpeaks(mEnv, 'MinPeakHeight', threshold, 'MinPeakDistance', min_peak_distance);
    
    % Create onset feature vector
    onsets = zeros(size(mEnv));
    onsets(onset_locs) = 1;
    
    % Trim or pad onset feature to match the length of the EEG data
    stim_Onset = onsets(1:size(resp,2));

    %% Transpose
    mOnsetEnv = stim_Onset';
    mEnv = stim_Env';
end