clc;
clear;
path='C:\Users\user\Desktop\ResNet80\data\test\';
%%
nx=6;
dx=0.8;
Output_Mag_ic=csvread([path,'output_Mag_ic_',num2str(nx),'_',num2str(dx),'_80.csv']);
Output_Mag_ec=csvread([path,'output_Mag_ec_',num2str(nx),'_',num2str(dx),'_80.csv']);
Output_Mag_cc=csvread([path,'output_Mag_cc_',num2str(nx),'_',num2str(dx),'_80.csv']);
Output_Pha_ic=csvread([path,'output_Pha_ic_',num2str(nx),'_',num2str(dx),'_80.csv']);
Output_Pha_ec=csvread([path,'output_Pha_ec_',num2str(nx),'_',num2str(dx),'_80.csv']);
Output_Pha_cc=csvread([path,'output_Pha_cc_',num2str(nx),'_',num2str(dx),'_80.csv']);
Mag_SED_Coef=Combine3(Output_Mag_ic,Output_Mag_ec,Output_Mag_cc);
Pha_SED_Coef=Combine3(Output_Pha_ic,Output_Pha_ec,Output_Pha_cc);
SED_Coef=[Mag_SED_Coef,Pha_SED_Coef];
csvwrite([path,'SED_Coef','_',num2str(nx),'_',num2str(dx),'_80.csv'],SED_Coef);
l=length(Mag_SED_Coef);
Mag_SED_Coef=reshape(Mag_SED_Coef,l^0.5,l^0.5);
Pha_SED_Coef=reshape(Pha_SED_Coef,l^0.5,l^0.5);
% Output_ic=reshape(Output_ic,18,18);
figure (1);
% imagesc(Mag_SED_Coef);
surf(Mag_SED_Coef,'EdgeColor','None');%绘制z的3D图  
shading interp;
%%
% Mag_SED_totalCoef=csvread(['E:\Tool\patch_v2\data_0809\output_Mag_total_',num2str(nx),'_',num2str(dx),'.csv']);
% Pha_SED_totalCoef=csvread(['E:\Tool\patch_v2\data_0809\output_Pha_total_',num2str(nx),'_',num2str(dx),'.csv']);
% l=length(Mag_SED_totalCoef);
% SED_totalCoef=[Mag_SED_totalCoef,Pha_SED_totalCoef];
% csvwrite(['E:\Tool\patch_v2\data_0809\SED_totalCoef_',num2str(nx),'_',num2str(dx),'.csv'],SED_totalCoef);
% Mag_SED_totalCoef=reshape(Mag_SED_totalCoef,l^0.5,l^0.5);
% Pha_SED_totalCoef=reshape(Pha_SED_totalCoef,l^0.5,l^0.5);
% figure (2)
% surf(Pha_SED_totalCoef,'EdgeColor','None');%绘制z的3D图  
% shading interp;
%%
% i_sed=load('E:\Tool\New_ASED_PEC\I_data_40_0.900000.txt');
