x = -5:.1:5;
pd = makedist('Normal','mu',0,'sigma',0.1);
pdf_normal = pdf(pd,x);
clf; hold on
plot(x,pdf_normal,'LineWidth',2);
