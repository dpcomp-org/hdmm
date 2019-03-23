sens = @(A)max(sum(abs(A)));
error = @(W,A)sens(A)*sqrt(trace((W'*W)*pinv(A'*A)));


n = 1024;
Q = zeros(n-32+1, n);
for i = 1:n-32+1
    Q(i, i:i+32-1) = 1.0;
end

%[~, A] = LowRankDP(Q);
%err1 = error(Q, A);
%err2 = error(Q, eye(n));
%fprintf('Width 32 Range, %.2f, %.2f \n', err1, err2);

%for n = [32, 64, 128, 256, 512, 1024]
%for n = [2,4,6,8,10]
for n = [2048, 4096]
    W = tril(ones(n,n));
    %W = kron(W, kron(W,W));
    tic;
    [~,A] = LowRankDP(W);
    time = toc;
    err1 = error(W, A);
    err2 = error(W, eye(size(W,2)));
    fprintf('Prefix 3D, %d, %.2f, %.2f, %.2f \n', n^3, err1, err2, time);
end
    