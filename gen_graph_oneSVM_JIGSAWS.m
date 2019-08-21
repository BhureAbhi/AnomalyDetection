% load /media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/tst_train.mat
% load /media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED1/tst_test.mat
% 
load /media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED2/tst_train.mat

load /media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED2/tst_test.mat
addpath(genpath('/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED2/graph_kernels'));

ntrain=size(y,1);
ntest=size(y1,1);
for v=1:ntrain
   G(v).am=double(W{v}>(mean(W{v}(:))+2*std(W{v}(:))));
   G(v).nl.values = round(V{v}*100);
   k=0;
   for j=1:size(G(v).am,1)
       G(v).al{j,1}=[find(G(v).am(j,:))];
   end
   disp(v);
end

for v=1:ntest
   G(ntrain+v).am=double(W1{v}>(mean(W1{v}(:))+2*std(W1{v}(:))));
   G(ntrain+v).nl.values = round(V1{v}*100);
   k=0;
   for j=1:size(G(ntrain+v).am,1)
       G(ntrain+v).al{j,1}=[find(G(ntrain+v).am(j,:))];
   end
   disp(v);
end

[K,runtime] = WL(G,10,1);
Kn=normalizekm(K{11});
X=Kn(1:ntrain,1:ntrain);
Xt=Kn(ntrain+1:ntrain+ntest,1:ntrain);

mdl={};
r=zeros(ntest,2);
res=zeros(ntest,2);
for c=1:2
    Xc=X(y==c,:);
    Xc1=X(y~=c,:);
    nXc=size(Xc,1);
    nXc1=size(Xc1,1);
    idx=randperm(nXc1,nXc);
    Xc1=Xc1(idx,:);
    mdl=fitcsvm([Xc;Xc1],[ones(nXc,1);zeros(nXc,1)],'KernelFunction','rbf');
    %y_pred=predict(mdl,X);
    %acc=sum((y==c)==y_pred)/ntrain;
    y_pred1=predict(mdl,Xt);
    tacc=sum((y1==c)==y_pred1)/ntest;
    disp([sum(y1==c)/ntest,tacc]);
end

y_pred={};
y_score={};
for c=1:2
    Xc=X(y==c,:);
    Xc1=X(y~=c,:);
    nXc=size(Xc,1);
    nXc1=size(Xc1,1);
    idx=randperm(nXc1,nXc);
    Xc1=Xc1(idx,:);
    mdl=fitcsvm([Xc;Xc1],[ones(nXc,1);-ones(nXc,1)],'KernelFunction','rbf');
    %y_pred=predict(mdl,X);
    %acc=sum((y==c)==y_pred)/ntrain;
    [y_pred{c},y_score{c}]=predict(mdl,Xt);
    tacc=sum(((y1==c)-(y1~=c))==y_pred{c})/ntest;
    disp([sum(y1==c)/ntest,tacc]) ;
end

y_score1=[];
for i=1:2
    y_score1=[y_score1,y_score{i}(:,2)];
end

csvwrite('/media/sdd/anomaly_sumit/STIP-MIL/UCSD/PED2/le2sci_2_y_scores.csv',y_score1);
