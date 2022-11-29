randn('seed',0);

N = 50
dA = 0 + (1-0)*rand(N,1) %[0,1]
dB =  0.25 + (1-0.25)*rand(2*N,1) %[0.25,1.15]
dC = normrnd(0.5, 1, [3*N,1]); %[mu:0.5,sig:1]

[h,p] = ttest2(dA,dB)
[h1,p1] = ttest2(dA,dC)
[h2,p2] = ttest2(dC,dB)
pr = ranksum(dA,dB)
p1_r= ranksum(dA,dC)
p2_r = ranksum(dC,dB)
figure
plot(dA)
hold on
plot(dB)
hold on 
plot(dC)
legend('vector A','vector B', 'vector C')
