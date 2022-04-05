function [Population,zmin,znad] = EnvironmentalSelection(Population,zmin,znad,N,gpophi,delta)

PopObj = Population.objs;
[N2,M] = size(PopObj); % N2 is the size of combined population (N2=2*N)
[FrontNo,~] = NDSort(PopObj,Inf); %Non-dominated sorting
zmin = min(zmin,min(PopObj,[],1));

%% Sorting with the "(M-1)+1"-GPO
phi=ones(1,M)*gpophi;
FrontGPO_All=zeros(M,N2);
for i=1:M
    phi2=phi;
    phi2(1,i)=0;
    FrontGPO_All(i,:)= AGPOSort(PopObj,phi2);
end

%% Normalization method adopted by theta-DEA
[PopObj,znad] = Normalization(PopObj,zmin,znad);

%% Select the corner solutions by ASF
W = zeros(M) + 1e-6;
W(logical(eye(M))) = 1;
ASF = zeros(N2,M);
for i = 1 : M
    ASF(:,i) = max(abs(PopObj-repmat(zmin,N2,1))./repmat(W(i,:),N2,1),[],2);
end
[~,extreme] = min(ASF,[],1);

FitAll=sum(PopObj,2);
FitExtreme=sum(PopObj(extreme,:),2);
IdExtreme=FitExtreme<1.5*median(FitAll); %% here 1.5 is a parameter value from [1,2] or larger
extreme=extreme(IdExtreme); %% or extreme=[];

%% Select extreme or randomly select a solution
if ~isempty(extreme)
    Select=unique(extreme);
else
    FrontNoGPO=FrontGPO_All(randperm(M,1),:);
    pool0=find(FrontNoGPO==min(FrontNoGPO));
    Select=pool0(randperm(length(pool0),1));
end

%% Radomly assigen the size of selected solution set for each processor
R=1:size(PopObj,1); 
S=Select;
R=setdiff(R,S); % Set of remaining solutions

N3=fix((N-length(Select))/M);
SubPopSize=N3*ones(M,1); % N3 or N3+1 for each processor
n3=N-N3*M-length(Select);
if n3>0
   SubPopSize(randperm(M,n3),1)=N3+1;
end

Angle_distance = pdist2(PopObj+0.0001,PopObj+0.0001,'cosine'); %% Note: no much difference with/without narmalization 

%% Select solution one-by-one with GPO
pool_i=randperm(M,M);
for j=1:M
    i=pool_i(j);
    t=0;
    FrontNoGPO=FrontGPO_All(i,:);
    while t<SubPopSize(i) && length(S)<N
        t=t+1;
        RN=R(FrontNo(R)==min(FrontNo(R)));    % Extract non-dominated solution set RN
        D_min=min(Angle_distance(RN,S),[],2); % The minmum pairwise distances between RN and selected set S
        [~,Angle_Sort]=sort(D_min,'descend');
        Len=ceil(delta*(N2-length(S)));     % the size of candicate solutions for survival
        %%% or Len=ceil(delta*N); or Len=max(ceil(delta*N*(1/2+1/2*(N-length(S))/N)),10); 
        Mean_D=D_min(Angle_Sort(min(Len,length(Angle_Sort))));
        index=find(D_min>(Mean_D-1e-6));
        pool=find(FrontNoGPO(RN(index))==min(FrontNoGPO(RN(index))));
        if length(pool)==1
            r=RN(index(pool));
        else
            index2=index(pool);
            id=find(D_min(index2)==max(D_min(index2)));
            if length(id)>1
                id2=id(randperm(length(id),1));
                r=RN(index2(id2));
            else
                r=RN(index2(id));
            end
        end
        if ~isempty(r)
            S=[S,r];     % Add the selected individual r to S
            R(R==r)=[];  % Remove r from R
        else
            continue
        end
    end
end

Population = Population(S);

end

%% Normalization
function [PopObj,znad,extreme] = Normalization(PopObj,z,znad)
% Normalize the population and update the ideal point and the nadir point
[N,M] = size(PopObj);

%%%% Update the nadir point
% Identify the extreme points
W = zeros(M) + 1e-6;
W(logical(eye(M))) = 1;
ASF = zeros(N,M);
for i = 1 : M
    ASF(:,i) = max(abs((PopObj-repmat(z,N,1))./(repmat(znad-z,N,1)))./repmat(W(i,:),N,1),[],2);
end
[~,extreme] = min(ASF,[],1);
% Calculate the intercepts
Hyperplane = (PopObj(extreme,:)-repmat(z,M,1))\ones(M,1);
a = (1./Hyperplane)' + z;
if any(isnan(a)) || any(a<=z)
    a = max(PopObj,[],1);
end

znad = a;

%%%% Normalize the population
if any(znad==0)
    znad(znad==0)=10^(-10);
end
PopObj = (PopObj-repmat(z,N,1))./(repmat(znad-z,N,1));

end
