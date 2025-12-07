function [stim_ABenv,NormEnv, NormBinEdges, BinEdges] = ABenvelopeGenerator_V1(stim_Env, num_bins)

    % Amplitude binned envelope
    % functionAmplitude_binned_vol1
    
    % Feature extraction and amplitude binning
    
    % Define the number of bins for amplitude binning
    if nargin < 2
    num_bins = 8; % Number of bins, adjust as needed
    end 

    % Compute the amplitude range for binning
    min_amp = min(stim_Env(:));
    max_amp = max(stim_Env(:));
    
    % bin_edges = linspace(min_amp, max_amp, num_bins + 1);
    BinEdges = logspace(log10(min_amp), log10(max_amp), num_bins + 1);
    
    % tRANSfer it to linear scale 
    
    % binEdges_linear = 10.^(binEdges_dB / 20);

    BinEdges= BinEdges';
    
    NormEnv = normalize(stim_Env,1,'range');
    NormBinEdges = normalize(BinEdges,1,'range');

    % Initialize the binned envelope matrix
    stim_binned = zeros(size(NormEnv,1), num_bins);
    
    % Bin the envelope data
    for bin_idx = 1:num_bins
        BinMask = (NormEnv >= NormBinEdges(bin_idx)) & (NormEnv < NormBinEdges(bin_idx + 1));
        % sum non-zero nums for differrent bins 
        a = BinMask==1;

        stim_binned(BinMask, bin_idx) = NormEnv(BinMask);
    end
    
    stim_ABenv = normalize(stim_binned,1,'range');
    
end

