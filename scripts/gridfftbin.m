clear all
close all

nplanes = 1;
% xaxis = 2048;
% yaxis = 2048;
% suffix = 'identity.txt';
% suffix = 'grid.txt';
% suffix = '_cuda_2048_1_oneside-';
% id = '4';
suffix = '';
id = '';
d
atadir = "/Users/cgheller/Work/Gridding/data/";
filename1 = strcat(datadir,"fft_real",suffix,id,'.bin');
filename2 = strcat(datadir,"fft_img",suffix,id,'.bin');
s = dir(filename1);
xaxis = sqrt(s.bytes/8.0);
yaxis = xaxis;
fid = fopen(filename1, 'rt');
vreal = fread(fid,'double');
fclose(fid)
fid = fopen(filename2, 'rt');
vimag = fread(fid,'double');
fclose(fid)

residual = complex(vreal,vimag);
% residual = vimag;
% clims = [0 800];

figure(1)
cumul2d = reshape(residual,[xaxis,yaxis,nplanes]);
for i=1:nplanes
    gridded = squeeze(cumul2d(:,:,i));
    figure(i)
%     image(uu,vv,log10(gridded),'CDataMapping','scaled',clims);
%      image(uu,vv,gridded,'CDataMapping','scaled');
%     imagesc(gridded,clims);
%     imagesc(uu,vv,log10(gridded),clims);
    imagesc((abs(fftshift(gridded))));%,clims);
    axis image
    colormap hot
    a = colorbar;
    a.Label.String = 'norm(FFT)';
    hold on
end

ax = gca;
ax.FontSize = 20;
% xlim([1000,7000]);
% ylim([1000,7000]);
xlabel('cell');
ylabel('cell');
pngimage = strcat(datadir,'fft-2freq',suffix,id,'.png');
saveas(gcf,pngimage);
