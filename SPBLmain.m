clear;
ss = tic;
%% Set the train ratio and noise ratio
train_ratio = 0.5;
n_ratio = 0;

addpath L-BFGS-B-C-master/Matlab/;
load('Data/fea.mat','fea','gnd');
% l2-normalized the feature. If not, comment out the next line
% fea = NormalizeFea(fea,1);
load(['Data/Split',num2str(train_ratio*100),'.mat'],'trainidx','testidx','valiidx');

nSmp = length(trainidx);
nClass = length(unique(gnd));
%% Set hyper-parameters
% # of base learners:
maxBase = 300;
% weight of l2-regularization of \theta_h for learning each new h():
base_rw = 1e-6 * nSmp;
% # of added weak model h() at each iteration:
EachAug = 1;

base_options = struct('type','logistic','regu_weight',base_rw,'EachAug',EachAug);
niu = 1e-4 * nSmp; % weights of l1 norm of W
epsilon = 1e-2*niu;
fprintf('nSmp = %d, nClass = %d, niu = %.4f, train ratio = %.2f\n',...
    nSmp,nClass,niu,train_ratio);
Wopts = struct('regutype','l21');

%% Initialization
trainfea = fea(trainidx,:);
if n_ratio == 0
    traingnd = gnd(trainidx);
else
    load(['Data/n',num2str(n_ratio*100),'_traingnd.mat'],'n_traingnd');
    traingnd = n_traingnd;
end
TrainObj = struct('idx',trainidx,'fea',fea(trainidx,:),'gnd',traingnd,'nClass',nClass);
ValiObj = struct('idx',valiidx,'fea',fea(valiidx,:),'gnd',gnd(valiidx),'nClass',nClass);
TestObj = struct('idx',testidx,'fea',fea(testidx,:),'gnd',gnd(testidx),'nClass',nClass);

ini_rr = 0.6; miu = 1.003; max_rr = 1;
% Specify the SPL reweighting scheme and the SPL parameters, where we set a SPLopts.zeta
% as the constant ratio of \zeta over \lambda for mixtrue reweighting.
SPLopts = struct('Type','mix','zeta',2);
ini_idx = Inisel(TrainObj.gnd,ini_rr);
v = zeros(nSmp,1); v(ini_idx) = 1;
u = zeros(nSmp,nClass); u(ini_idx,:) = 1/nClass;
rankratio = ini_rr;

%% Self-Paced Boosting
if exist('Base_Para','var'), clear Base_Para; end
H = zeros(nSmp,0);
ResObj = struct('train_err',zeros(0,2),'vali_err',zeros(0,2),...
    'test_err',zeros(0,2),'dualgap',[],'dualviolation',[],...
    'allW',cell(1,1),'allv',v);
fprintf('Initializing the first base classifier ...\n');
k = 1; mbase = 1;
while k <= maxBase %|| rankratio < max_rr % At k-th iteration
    fprintf('k %d(%d~%d): base training ... ',k,mbase,mbase+EachAug-1);
    if k > 1
        loss = Eachloss(W,H,traingnd);
        [v,SPLlambda] = SPLreweighting(loss,rankratio,SPLopts);
        ResObj.allv(:,k) = v;
    else SPLlambda = 0;
    end
    [new_Base, newH] = TrainBase(trainfea,traingnd,u,base_options);
    fprintf('        vio/niu = %.2f/%.2f, max(|newbase.w|) = %.3f, newbase.b = %.3f\n',...
        new_Base(1).maxScore,niu,max(abs(new_Base(1).w)),new_Base(1).b);
    % It's OK to comment out the convergence conditions (next 4 lines)
    if k > 1 && new_Base(1).maxScore < niu + epsilon && rankratio >= max_rr
        fprintf('violation < niu + epsilon, break\n');
        break;
    end
    if k == 1, Base_Para = new_Base;
    else Base_Para = [Base_Para;new_Base];
    end
    H = [H,newH];
    
    s2 = tic;
    fprintf('        Update W: ... ');
    [W,fval] = UpdateW(H,TrainObj,v,niu,Wopts);
    ResObj.allW{k,1} = W;
    etime2 = toc(s2);
    fprintf('%d s. max(W) = %.4f, fval = %.2f\n',round(etime2),max(max(W)),fval);
    [u,ResObj.dualgap(k,1),ResObj.dualviolation(k,1),ResObj.train_err(k,:)] =...
        DualTrans(H,W,TrainObj.gnd,v,fval,niu,Wopts);
    fprintf('        dualvio/dualgap | rr/SPL = %.3f(%.3f)/%.3f | %.3f/%.3f\n',...
        ResObj.dualviolation(k),max(ResObj.dualviolation),ResObj.dualgap(k),rankratio,SPLlambda);
    allW{k,1} = W;
    
    ResObj.test_err(k,:) = Predict(Base_Para,W,TestObj);
    if ~isempty(valiidx)
        ResObj.vali_err(k,:) = Predict(Base_Para,W,ValiObj);
        [opt_top1,opt_k] = min(ResObj.vali_err(:,1));
        fprintf('        top1: train/vali/opt = %.3f/%.3f/(%.3f,%d,%d)\n',...
            ResObj.train_err(k,1),ResObj.vali_err(k,1),...
            opt_top1,opt_k,size(ResObj.allW{opt_k},1));
    else
        [opt_top1,opt_k] = min(ResObj.test_err(:,1));
        fprintf('        top1: train/test/opt = %.3f/%.3f/(%.3f,%d,%d)\n',...
            ResObj.train_err(k,1),ResObj.test_err(k,1),...
            opt_top1,opt_k,size(ResObj.allW{opt_k},1));
    end
    rankratio = min(rankratio*miu,max_rr); % Annealing
    k = k + 1;
    mbase = mbase + EachAug;
    if rankratio == max_rr
        EachAug = 1; base_options.EachAug = 1;
    end
end
fprintf('Relative max dualviolation = %.3f\n',max(ResObj.dualviolation));
fprintf('Relative max dualgap = %.3f\n',max(ResObj.dualgap));
fprintf('Test top-1 error rate = %.4f, opt_mbase = %d\n',...
    ResObj.test_err(opt_k,1),size(ResObj.allW{opt_k},1));
if ~isempty(valiidx)
    [~,opt_k5] = min(ResObj.vali_err(:,2));
else [~,opt_k5] = min(ResObj.test_err(:,2));
end
fprintf('Test top-5 error rate = %.4f, opt_mbase = %d\n',...
    ResObj.test_err(opt_k5,1),size(ResObj.allW{opt_k5},1));
OptRes = struct('top1',struct('err',ResObj.test_err(opt_k,1),'k',opt_k,'mbase',size(ResObj.allW{opt_k},1)),...
    'top5',struct('err',ResObj.test_err(opt_k5,1),'k',opt_k5,'mbase',size(ResObj.allW{opt_k5},1)));

if ~exist(['n',num2str(n_ratio*100)],'dir')
    mkdir(['n',num2str(n_ratio*100)]);
end
save(['n',num2str(n_ratio*100),'/SPBLRes.mat'],'Base_Para','ResObj','OptRes','base_options','niu',...
    'epsilon','train_ratio','trainidx','testidx','nClass');

toc(ss);

