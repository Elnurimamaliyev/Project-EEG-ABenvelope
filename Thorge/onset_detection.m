function peaks = onset_detection(x,varagin)
%% this function allows to derive peaks using 4 different ways
%
%input: 
%x = signal of interest to derive the peaks
%1. Local Energy function 
%   - N (default 2048) =  
%2. Spectral Energy
%3. Phase
%4. Complex Domain 

%% local energy function 
Fs = fs;
duration = length(audio_mono)/Fs;

N = 2048;
w = hann(N);
H = 128;
fs_nov = fs/H;

%compute the local energy
wav_square = audio_mono.^2;
energy_local = conv(wav_square,w,'same');
energy_local = energy_local(1:H:end,1);

%take the log
energy_local = log(1 + 10 .* energy_local);

%take the derivative
energy_local_diff = diff(energy_local);
energy_local_diff = cat(1,energy_local_diff,zeros(1,1));

%set all neg values to 0
novelty_energy = energy_local;
novelty_energy(novelty_energy < 0 ) = 0 ; 

%normalize
max_val = max(novelty_energy);
novelty_energy_norm = novelty_energy/max_val;

%derive the peaks
peak = simp_peak(novelty_energy_norm,0.2);

figure(2),clf
subplot(5,1,1)
plot(audio_mono)
set(gca,'Xlim',[0 500000])
title('audio')

subplot(5,1,2)
plot(energy_local)
set(gca,'Xlim',[0 4000])
title('smoothed,squared audio signal')

subplot(5,1,3)
plot(energy_local_diff)
set(gca,'Xlim',[0 4000])
title('rate of change')

subplot(5,1,4)
plot(novelty_energy_norm)
set(gca,'Xlim',[0 4000])
title('novelty energy')

subplot(5,1,5)
plot(peaks)
set(gca,'Xlim',[0 4000])
title('peaks')

%% spectral based novelty
%motivated by polyphonic musical events
%1. convert into frequency bands
%2. capture frequency changes
N = 1024;
H = 768;
gamma = 10;
%short fourier transformation
X = stft(audio_mono,fs,'window',hann(N),'OverlapLength',H,'FFTLength',N);

%take the log
Y = log(1 + gamma * abs(X));

%take the derivative
Y_diff = diff(Y,1,2);
Y_diff(Y_diff < 0) = 0;
nov = sum(Y_diff,1);
nov = cat(2,nov, zeros(size(Y,2)-size(Y_diff,2),1)');

%moving average
fs_nov = fs/(N-H);
M = (0.1*fs_nov);
l_avg = cmp_loc_avg(nov,ceil(M));
nov_norm =nov - l_avg;
nov_norm(nov_norm<0) = 0;

%normalize 
nov_norm = nov_norm/max(nov_norm);

sec = 15;


figure(2),clf
subplot(4,1,1)
plot(audio_mono)
set(gca,'Xlim',[0 sec*fs])
title('audio')

subplot(4,1,2)
plot(nov)
set(gca,'Xlim',[0 sec*fs_nov])
title('rate of change summed over frequencies')

subplot(4,1,3)
plot(nov_norm)
set(gca,'Xlim',[0 sec*fs_nov])
title('normalized and smoothed rate of change')

subplot(4,1,4)
plot(simp_peak(nov_norm,0.2))
set(gca,'Xlim',[0 sec*fs_nov])
title('peaks')

%% phase based novelty
%
%

N = 1024;
H = 768;
gamma = 10;
%short fourier transformation
X = stft(audio_mono,fs,'window',hann(N),'OverlapLength',H,'FFTLength',N);

fs_nov = fs/(N-H);

phase = angle(X) / (pi*2);
phase_diff = normalize(diff(phase,1,2),'range',[-0.5,0.5]);
phase_diff2 = normalize(diff(phase_diff,1,2),'range',[-0.5,0.5]);

novelty_phase = sum(abs(phase_diff2),1);

novelty_phase = cat(2,novelty_phase, zeros(size(phase,2)-size(phase_diff,2),1)');

M = (0.01*fs_nov);

if ~isempty(M)
    l_avg = cmp_loc_avg(novelty_phase,ceil(M));
    novelty_phase_norm =novelty_phase - l_avg;
    novelty_phase_norm(novelty_phase_norm<0) = 0;
end

if norm
    novelty_phase_norm = novelty_phase_norm/max(novelty_phase_norm);
end



figure(3),clf
subplot(5,1,1)
plot(audio_mono)
set(gca,'Xlim',[0 sec*fs])
title('audio')

subplot(5,1,2)
imagesc(phase)
set(gca,'Xlim',[0 sec*fs_nov])
title('phase')

subplot(5,1,3)
imagesc(phase_diff2)
set(gca,'Xlim',[0 sec*fs_nov])
title('2 derivative of phase')

subplot(5,1,4)
plot(novelty_phase_norm)
set(gca,'Xlim',[0 sec*fs_nov])
title('peaks')

subplot(5,1,5)
plot(simp_peak(novelty_phase_norm,0.2))
set(gca,'Xlim',[0 sec*fs_nov])
title('peaks')


%% Complex Domain novelty
%
%
N = 1024;
H = 64;
gamma = 10;
%short fourier transformation
X = stft(audio_mono,fs,'window',hann(N),'OverlapLength',H,'FFTLength',N);
fs_nov = fs/(N-H);

Y = abs(X);
%take the log
if take_log
    Y = log(1 + gamma * Y);
end


phase = angle(X) / (pi*2);
phase_diff = diff(phase,1,2);
phase_diff = cat(2,phase_diff, zeros(size(phase,2)-size(phase_diff,2),size(phase,1))');

X_hat = Y .* exp(2*pi*1i*(phase+phase_diff));
X_prime = abs(X_hat-X);
X_plus = repmat(X_prime,1);

for i = 2:size(X,1)
    idx = Y(i,:) < Y(i-1,:);
    X_plus(i,idx) = 0;
end
novelty_complex = sum(X_plus,1);

%smooth the function
if ~isempty(M) 
    l_avg = cmp_loc_avg(novelty_complex,ceil(M));
    novelty_complex =novelty_complex - l_avg;
    novelty_complex(novelty_complex<0) = 0;
end

%normalize
if norm
    novelty_complex_norm = novelty_complex/max(novelty_complex);
end

sec = 12; 

figure(4),clf
subplot(7,1,1)
plot(audio_mono)
set(gca,'Xlim',[0 sec*fs])
title('audio')

subplot(7,1,2)
imagesc(phase)
set(gca,'Xlim',[0 sec*fs_nov])
title('phase')

subplot(7,1,3)
imagesc(real(X_hat))
set(gca,'Xlim',[0 sec*fs_nov])
title('magnitude')

subplot(7,1,4)
imagesc(X_prime)
set(gca,'Xlim',[0 sec*fs_nov])
title('measure of novelty')

subplot(7,1,5)
plot(novelty_complex)
set(gca,'Xlim',[0 sec*fs_nov])
title('novelty function')

subplot(7,1,6)
plot(novelty_complex_norm)
set(gca,'Xlim',[0 sec*fs_nov])
title('novelty function normalized')

ons = resample(stims(1,:),ceil(fs_nov),100);
subplot(7,1,7)
plot(simp_peak(novelty_complex_norm,0.05))
hold on
plot(ons)
set(gca,'Xlim',[0 sec*fs_nov])
title('novelty function')








