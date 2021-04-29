close all; clear; clc; 

im = imread("data/mydata/img/00000001.jpg");
im = imrotate(im, -90);
figure;
imshow(im); 