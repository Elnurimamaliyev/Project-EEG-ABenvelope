function  [fs_eeg, resp, fs_audio, audio_dat]  = LoadEEG(s, k, sbj, task)
        % Load the EEG
        EEGFileName = sprintf('Preprocessed_EEG_Subject_%s_Task_%s.mat', sbj{s}, task{k});
        load(EEGFileName);          % Load the EEG

        % extract the EEG features
        fs_eeg = EEG.srate;         % Get the final EEG sample rate
        resp = EEG.data';           % Assign the EEG data to the resp variable
        
        % Load the audio 
        AudioFileName = sprintf('%s_%s_audio_strct.mat', sbj{s}, task{k});
        % AudioFileName = [sbj{s},'_',task{k},'_audio_strct.mat'];
        wav = load(AudioFileName); % for instance P001_narrow_audio_strct.mat
        fs_audio = wav.audio_strct.srate; % Get the final audio sample rate
        audio_dat = wav.audio_strct.data; % Assign the audio data to the audio_dat variable
end