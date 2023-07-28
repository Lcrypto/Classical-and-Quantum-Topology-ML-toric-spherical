%Copyright(c) 2023, Usatyuk Vasiliy 
%All rights reserved.
%
%Redistribution and use in source and binary forms, with or without
%modification, are permitted provided that the following conditions are met :
%*Redistributions of source code must retain the above copyright
%notice, this list of conditions and the following disclaimer.
%* Redistributions in binary form must reproduce the above copyright
%notice, this list of conditions and the following disclaimer in the
%documentation and / or other materials provided with the distribution.
%* Neither the name of the <organization> nor the
%names of its contributors may be used to endorse or promote products
%derived from this software without specific prior written permission.
%
%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED.IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
%DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Core contribution LDPC based Tensor Factorization for future use to
% simplify Complex Transformer Deep Neural Neutwork which contain billions parameters
% based on the Matlab platform from https://github.com/RuslanKhalitov/SparseFactorization
%Ruslan Khalitov, Tong Yu, Lei Cheng, Zhirong Yang,
%Sparse factorization of square matrices with application to neural attention modeling,
%Neural Networks,Volume 152,2022, Pages 160-168,


di=17;
%di=36
%directory with result file
addpath('results');
%optimization result in result folder
%load sf_appr_test_di36.mat
load sf_appr_test_di17.mat

A = load_square_matrix(categories{di}, filenames{di});
categories{di}
filenames{di}
check_svds_error(A)

[~, norm_loss_of_LDPC_factorization] = sf_obj(res.Ws, A)



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

function svd_loss = check_svds_error(A)
N = size(A,1);
M = ceil(log2(N));
r = ceil(M * M / 2);
[U, S, V] = svds(A, r);
Ahat = U * S * V';
svd_loss = norm(A-Ahat, 'fro');
fprintf('SVDs error = %.6f\n', svd_loss);
end