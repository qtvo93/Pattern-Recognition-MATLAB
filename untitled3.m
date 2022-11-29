close('all');
clear

ima = imread('./portraitGallery/a.jpg');
ima = mat2gray(rgb2gray(ima));

imb = imread('./portraitGallery/b.jpg');
imb = mat2gray(rgb2gray(imb));

imc = imread('./portraitGallery/c.jpg');
imc = mat2gray(rgb2gray(imc));

fdrab = FisherDiscriminant(ima,imb);
fdrbc = FisherDiscriminant(imb,imc);
fdrac = FisherDiscriminant(ima,imc);

function fdr = FisherDiscriminant(ima,imb)
    mua = mean(ima(:));
    mub = mean(imb(:));
    sigmaa = std(ima(:));
    sigmab = std(imb(:));
    fdr = (mua - mub )^2/ (sigmaa^2 + sigmab^2);
end