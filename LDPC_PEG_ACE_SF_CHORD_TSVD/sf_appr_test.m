function res = sf_appr_test(A, max_iter,Check_LDPC,irregular_profile,maxDv)
if ~exist('max_iter', 'var') || isempty(max_iter)
    max_iter = 1e5;
end

N = size(A,1);
M = ceil(log2(N));





% LDPC nonstructured with ACE maximization
if Check_LDPC
Dv1=ones(1,N); % degree distribution per each node
if irregular_profile==1 %construct irregular LDPC
%Dv1(1,1:ceil(N/10))=Dv1(1,1:ceil(N/10))* maxDv; % 10 percent of max weight
%Dv1(1,ceil(N/10)+1:end)=Dv1(1,ceil(N/10)+1:end)* 3; % 90 percent weigh 3
Dv1(1,1:ceil(N/10))=Dv1(1,1:ceil(N/10))*24 ; %  25%  of  weight 3
Dv1(1,ceil(N/10)+1:ceil(N/10)+1+ceil(N/4))=Dv1(1,ceil(N/10)+1:ceil(N/10)+1+ceil(N/4))*3 ; %  25%  of  weight 3
Dv1(1,ceil(N/10)+2+ceil(N/4):end)=Dv1(1,ceil(N/10)+2+ceil(N/4):end)* 3; % 75%  of max weight
Dv1(1,ceil(N/10)+2+ceil(N/4):end)=Dv1(1,ceil(N/10)+2+ceil(N/4):end)* 3; 
end
if irregular_profile==-1 %construct irregular LDPC
check=chord_mask_mat(N, 2);

degree_cage=(sum(check));
Dv1=degree_cage(1)*ones(1,N); 
end

if irregular_profile==0 % regular ldpc with maximal weight maxDv
Dv1=maxDv*ones(1,N); 
end



H1 = ProgressiveEdgeGrowthACE(N,N , Dv1, 1);
mask=double(H1);
%spy(H1)
else
% original decompose matrix chord (cage like)
mask = chord_mask_mat(N, 2);
end


%For a fair comparison,we  has nearly the same number and no fewer
%non-zeros than cage 
%5.1. Exploring real-world matrices for SF
check=chord_mask_mat(N, 2);

non_zero_cage=sum(sum(check));



Nonzero_LDPC= sum(sum((mask)));


if non_zero_cage~=Nonzero_LDPC
%to show that LDPC  can supersede SF, TSVD in F-norm by using the same number
%of non-zeros

error()



end













%QC MET-LDPC Product Tensor for Structured Sparse Transformer Factorization
 %Dv1=4*ones(1,N)
 %H1 = ProgressiveEdgeGrowthACE(N,N , Dv1, 1)
 %Dv2=ones(1,N)
 %H2 = ProgressiveEdgeGrowthACE(N,N , Dv2, 1)
 
% High Rate (Low Temperature) - low noise regime
% QC multi-edge LDPC sparse subTensor
%mask1=
% QC multi-edge repetition (cage) subTensor
%mask2=

% Low Rate (High Temperature) - high noise regime
% QC multi-edge LDPC sparse subTensor
%mask3=
% QC multi-edge repetition (cage) subTensor
%mask4=

% BCH and RS binary-image   High complexity low reconstruction error
% QC multi-edge LDPC sparse subTensor
%mask5=
% QC multi-edge repetition (cage) subTensor





maskn = mask ./ sum(mask,2);



svd_loss = check_svds_error(A);

rng(1, 'twister');

Ws = cell(M,1);
for m=1:M
    Ws{m} = maskn;
    noise = rand(N, N);
    noise = noise ./ sum(noise, 2) * 1e-2;
    Ws{m}(mask>0) = Ws{m}(mask>0) + noise(mask>0);
end

obj = sf_obj(Ws, A);
fprintf('iter=0, obj=%.10f\n', obj);

Wsvec = Ws2Wsvec(Ws, mask);

options = optimset('LargeScale', 'off', ...
    'MaxIter', max_iter, ...
    'OutputFcn', @output_func, ...
    'GradObj', 'on', ...
    'TolX', 1e-10);

Wsvec_opt = fminunc(@(Wsvec)sf_obj_grad_sparse(Wsvec, A, M, mask), Wsvec, options);

Ws = Wsvec2Ws(Wsvec_opt, M, mask);
[obj, norm_loss] = sf_obj(Ws, A);
fprintf('final obj=%.10f, norm_loss=%.10f\n', obj, norm_loss);

res.Ws = Ws;
res.obj = obj;
res.norm_loss = norm_loss;
res.svd_loss = svd_loss;
%iter=13200/20000, obj=1556976.762280, norm_loss=1764.639772

    function stop = output_func(x, optimValues, ~)
        stop = false;
        if mod(optimValues.iteration, max(1, round(max_iter/100)))==0
            [obj_disp, norm_loss_disp] = sf_obj(Wsvec2Ws(x, M, mask), A);
            fprintf('iter=%d/%d, obj=%.6f, norm_loss=%.6f\n', optimValues.iteration, max_iter, obj_disp, norm_loss_disp);
        end
    end
end

function svd_loss = check_svds_error(A)
N = size(A,1);
M = ceil(log2(N));
r = ceil(M * M / 2);
[U, S, V] = svds(A, r);
Ahat = U * S * V';
svd_loss = norm(A-Ahat, 'fro');
fprintf('SVDs error = %.6f\n', svd_loss);
end

function Wsvec = Ws2Wsvec(Ws, mask)
M = length(Ws);
nnz_mask = nnz(mask);
ind_mask = mask>0;
Wsvec = zeros(nnz_mask * M, 1);
for m=1:M
    Wsvec((m-1)*nnz_mask+1:m*nnz_mask) = Ws{m}(ind_mask);
end
end

function Ws = Wsvec2Ws(Wsvec, M, mask)
[I, J] = find(mask);
N = size(mask,1);
nnz_mask = nnz(mask);
Ws = cell(M,1);
for m=1:M
    Ws{m} = full(sparse(I, J, Wsvec((m-1)*nnz_mask+1:m*nnz_mask), N, N));
end
end

function [obj, norm_loss] = sf_obj(Ws, A)
M = length(Ws);
Ahat = Ws{1};
for m=2:M
    Ahat = Ahat * Ws{m};
end
obj = 0.5*norm(A-Ahat, 'fro').^2;
if nargout>1
    norm_loss = norm(A-Ahat, 'fro');
end
end

function [obj, grad] = sf_obj_grad_sparse(Wsvec, A, M, mask)
N = size(A,1);
ind_mask = mask>0;
nnz_mask = nnz(mask);

Ws = Wsvec2Ws(Wsvec, M, mask);
obj = sf_obj(Ws, A);

if nargout > 1
    grad = zeros(nnz_mask*M,1);
    Rs = cell(M,1);
    for m=M:-1:1
        if m==M
            Rs{m} = Ws{M};
        else
            Rs{m} = Ws{m} * Rs{m+1};
        end
    end
    for m=1:M
        if m==1
            L = eye(N);
        else
            L = L * Ws{m-1};
        end
        if m==M
            R = eye(N);
        else
            R = Rs{m+1};
        end
        
        W = Ws{m};
        gradk = (-L' * A * R' + L' * L * W * R * R') .* mask;
        
        grad((m-1)*nnz_mask+1:m*nnz_mask) = gradk(ind_mask);
    end
end
end