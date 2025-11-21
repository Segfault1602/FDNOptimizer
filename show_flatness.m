clearvars; close all;

[ir, fs] = audioread("initial_ir.wav");

figure(1);
plot(ir);
xlabel('Sample Index');
ylabel('Amplitude');
title('Impulse Response');

WIN_LEN = 2^15;
OVL_LEN = round(0.1*WIN_LEN);
UPPER_FREQ = 20000;

figure(2);
spectralFlatness(ir, fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,fs/2]);

[final_ir, fs] = audioread("final_ir.wav");

figure(3);
plot(final_ir);
xlabel('Sample Index');
ylabel('Amplitude');
title('Final Impulse Response');


figure(4);
spectralFlatness(final_ir, fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,fs/2]);

hold on;
spectralFlatness(ir, fs, Window=rectwin(WIN_LEN), OverlapLength=OVL_LEN, ...
              Range=[0,fs/2]);
hold off;
ylim([0 1]);
legend("Optimized", "Original");
