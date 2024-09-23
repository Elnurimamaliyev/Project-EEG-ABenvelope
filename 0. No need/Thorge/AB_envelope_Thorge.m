%% AB envelope
%got to the folder
% cd ('\\daten.w2kroot.uni-oldenburg.de\home\lorf0331\Desktop\TRFPP\')

%% load the EEG
OT_setup
% [EEG,PATH] = OT_preprocessing(1,1,sbj,20);
%% Continue once you have a running pre-processing pipeline
%get the final EEG srate
fs_eeg = EEG.srate;

%assign the EEG data to the resp variable
resp = EEG.data';

wavnarrow = load('P001_narrow_audio_strct.mat');
fs_audio = wavnarrow.audio_strct.srate;
audio_dat = wavnarrow.audio_strct.data;

%FEATURE extraction 
%we extract the envelope
%%% Envelope 1: TRF provided
menv = mTRFenvelope(double(audio_dat)',fs_audio,fs_eeg);
%%
%%% Envelope 2: Manual with Hilbert Transform
wav_h = abs( hilbert(double(wavnarrow.audio_strct.data)) );
[b,a] = butter(3,15/(wavnarrow.audio_strct.srate/2),'low');
wav_hf = filtfilt( b,a,wav_h );
wav_hfd = downsample(wav_hf, 441)';
menv = wav_hfd;


%normalize the envelope
menv_norm = (menv - min(menv)) / ...
                      (max(menv) - min(menv));

% Define the bin edges in dB
binEdges_dB = 30:8:80;  % Binning in 8 dB steps up to 64 dB (adjust as needed)
nBins = length(binEdges_dB);

% Convert dB edges to linear scale by taking 10^(binEdges_dB/20)
binEdges_linear = 10.^(binEdges_dB / 20);

% Normalize the bin edges to be between 0 and 1
binEdges_normalized = (binEdges_linear - min(binEdges_linear)) / ...
                      (max(binEdges_linear) - min(binEdges_linear));

% Calculate the histogram counts and bin indices using histcounts
[counts, ~, binIndices] = histcounts(menv_norm, binEdges_normalized)

%look at the different frequency counts
figure
histogram('BinEdges',binEdges_dB,'BinCounts',counts)
set(gca,'view',[90 -90])
%binned envelope
env_bin = zeros(length(binEdges_dB),length(menv));

for i = 1:length(binIndices)
    env_bin(binIndices(i),i) = menv_norm(i);
end

%normalize each bin 
env_bin_norm = normalize(env_bin,2,'range');

figure
for i = 1:length(binEdges_dB)
    subplot(length(binEdges_dB),1,i)
    plot(env_bin_norm(i,1:100));
    box off
    
    set(gca, 'color', 'none')
    ylabel(sprintf('%d',binEdges_dB(i)))    
end

stim_col = {menv,env_bin_norm'};
nfold = 10;
testfold = 1;
Dir = 1; %specifies the forward modeling
tmin = -100;
tmax = 500;
lambdas = linspace(10e-4,10e4,10);
for  s = 1:length(stim_col)
    stim = stim_col{1,s};
    %TRF prediction
    if size(resp,1)>size(stim,1)
        resp = resp(1:size(stim,1),:);
    elseif size(resp,1)<size(stim,1)
        stim = stim(1:size(resp,1),:);
    end
    
    %partition the data set
    [strain,rtrain,stest,rtest] = mTRFpartition(stim,resp,nfold,testfold);
    
    %% z-score the input and output data
    strainz = strain;
    stestz = stest;
    
    
    rtrainz = cellfun(@(x) zscore(x,[],'all'),rtrain,'UniformOutput',false);
    rtestz = zscore(rtest,[],'all');
    
    %% use cross-validation
    fs = EEG.srate;
    
    
    cv = mTRFcrossval(strainz,rtrainz,fs,Dir,tmin,tmax,lambdas,'Verbose',0);
    
    %get the optimal regression parameter
    l = mean(cv.r,3); %over channels
    [l_val,l_idx] = max(mean(l,1));
    l_opt = lambdas(l_idx);
    
    %train the neural model on the optimal regularization parameter
    model_train = mTRFtrain(strainz,rtrainz,fs,Dir,tmin,tmax,l_opt,'verbose',0);
    %predict the neural data
    [PRED,STATS] = mTRFpredict(stestz,rtestz,model_train,'verbose',0);
    
    reg(s,:) = STATS.r;
    mod_w{s,1} = squeeze(model_train.w); 
end

%plot prediction accuracies
figure
boxplot(reg','Labels',{'menv','AB menv'})
ylabel('Prediction Accuracy')
bins = binEdges_dB;
%plot the model weights
for i = 1:size(mod_w,1)
    modw = mod_w{i,1};
    
    if ndims(modw) > 2
        w_m = squeeze(mean(modw,3));
        w_m(sum(w_m,2) == 0,:) = [];
        bins(sum(w_m,2) == 0,:) = [];
        figure
        tiledlayout(size(w_m,1),1)
        for b = 1:size(w_m,1)
            nexttile
            plot(model_train.t,w_m(b,:),'k')
            box off
            
            set(gca, 'color', 'none')
            ylabel(sprintf('%d',bins(b)))
            
        end
        
    else
        figure
        plot(model_train.t,mean(modw,2),'k')
    end
    
    
end




