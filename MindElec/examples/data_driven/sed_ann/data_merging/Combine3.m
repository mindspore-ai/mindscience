function I=Combine3(ic,ec,cc)
% clc;
% clear;
% ic=[6;7;10;11];
% ec=[2;3;5;8;9;12;14;15];
% cc=[1;4;13;16];
[l_ic,c]=size(ic);
[l_ec,c]=size(ec);
[l_cc,c]=size(cc);
coef_concat=[ic;ec;cc];
ind_ic=[1:1:l_ic]';
ind_ec=[l_ic+1:1:l_ic+l_ec]';
ind_cc=[l_ic+l_ec+1:1:l_ic+l_ec+l_cc]';
l=l_ic+l_ec+l_cc;
ic=reshape(ind_ic,l_ic^0.5,l_ic^0.5);
ec1=ind_ec(1:l_ec/4);
ec2=reshape(ind_ec(l_ec/4+1:3*l_ec/4),2,l_ec/4);
ec3=ind_ec(3*l_ec/4+1:l_ec);
ind=[ind_cc(1),ec2(1,:),ind_cc(3);ec1,ic,ec3;ind_cc(2),ec2(2,:),ind_cc(4)];
ind=reshape(ind,[],1);
I=coef_concat(ind,:);
return