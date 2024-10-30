function [mOnsetStim] = OnsetGenerator(mEnv, min_peak_distance, onset_threshold)
    % OnsetGenerator: Generates a binary onset feature vector from an envelope.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Inputs:
    %   mEnv - The envelope of the audio signal.
    %   min_peak_distance - (Optional) Minimum distance between peaks (in samples).
    %   onset_threshold - (Optional) Threshold for peak detection.
    %
    % Outputs:
    %   mOnsetStim - A binary vector representing onsets detected in the envelope.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Default Values
    if nargin < 3
        onset_threshold = 0.3; % Default threshold for peak detection
    end
    if nargin < 2
        min_peak_distance = 15; % Default minimum distance between peaks
    end
    %% Onset Feature Generation
    % Find peaks in the envelope that correspond to onsets using specified threshold and minimum peak distance.
    [~, onset_locs] = findpeaks(mEnv, 'MinPeakHeight', onset_threshold, 'MinPeakDistance', min_peak_distance);
    
    % Create a binary onset feature vector with the same length as mEnv
    mOnsetStim = zeros(size(mEnv));
    mOnsetStim(onset_locs) = 1;
    
    %% Transpose Outputs
    % Transpose the outputs to match expected format
    % mOnsetStim = mOnsetStim';
end
