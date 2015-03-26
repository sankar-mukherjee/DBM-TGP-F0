function nnTfeature = nnTransform(nn, x)
%nnTransform performs a feedforward pass tranformation of input features

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;

    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');
            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
                % [Sankar] soft rectified liniar unit, rectified liniar unit
                %see(http://code.google.com/p/cuda-convnet/wiki/NeuronTypes)
            case 'softrelu'
                nn.a{i} = log(1+exp(nn.a{i - 1} * nn.W{i - 1}'));
            case 'relu'
                nn.a{i} = max(0,(nn.a{i - 1} * nn.W{i - 1}'));
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
    end
   nnTfeature = nn.a{n-1}(:,2:end);
end
