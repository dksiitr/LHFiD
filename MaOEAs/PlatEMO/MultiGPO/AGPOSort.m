function FrontNo = AGPOSort(PopObj,phi)

%% loosening coefficient
origObj=PopObj;
[nPop,nObj]=size(origObj);
delta = tan(pi*phi/180);

%% mapping to new objectives
newObj=zeros(nPop,nObj);
for i=1:nObj
    mask = ones(1,nObj);
    mask(i) = 0;
    newObj(:,i) = origObj(:,i)+sqrt(nObj-1)*sum(origObj(:,mask>0),2)...
        *delta(i)/(nObj-1);
end

[FrontNo,~] = NDSort(newObj,Inf); % Non-dominated sorting

end
