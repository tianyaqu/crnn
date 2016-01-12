function getConfig()
    local config = {
        --nClasses         = 36,
        nClasses         = 36,
        maxT             = 30,
        displayInterval  = 100,
        testInterval     = 1000,
        nTestDisplay     = 15,
        trainBatchSize   = 2,
        valBatchSize     = 2,
        snapshotInterval = 500,
        maxIterations    = 2000000,
        optimMethod      = optim.adadelta,
        optimConfig      = {},
        --trainSetPath     = '../data/synth90k_train_lmdb/data.mdb',
        --valSetPath       = '../data/synth90k_val_lmdb/data.mdb',                
        trainSetPath     = '../data/so/data.mdb',
        valSetPath       = '../data/so_val/data.mdb',
        useCPU           = true,
    }
    return config
end


function createModel(config)
    local nc = config.nClasses
    local nl = nc + 1
    local nt = config.maxT
    local useCPU = config.useCPU or false

    local ks = {3, 3, 3, 3, 3, 3, 2}
    local ps = {1, 1, 1, 1, 1, 1, 0}
    local ss = {1, 1, 1, 1, 1, 1, 1}
    local nm = {64, 128, 256, 256, 512, 512, 512}
    local nh = {256, 256}

    function convRelu(i, batchNormalization,useCPU)
        batchNormalization = batchNormalization or false
        useCPU = useCPU or false
        
        local nIn = nm[i-1] or 1
        local nOut = nm[i]
        local subModel = nn.Sequential()
        local conv
        
        if not useCPU then
            conv = cudnn.SpatialConvolution(nIn, nOut, ks[i], ks[i], ss[i], ss[i], ps[i], ps[i])
        else
            conv = nn.SpatialConvolution(nIn, nOut, ks[i], ks[i], ss[i], ss[i], ps[i], ps[i])
        end
        
        subModel:add(conv)
        if batchNormalization then
            subModel:add(nn.SpatialBatchNormalization(nOut))
        end
        
        if not useCPU then
            subModel:add(cudnn.ReLU(true))
        else
            subModel:add(nn.ReLU(true))
        end
        
        return subModel
    end

    function bidirectionalLSTM(nIn, nHidden, nOut, maxT,useCPU)
        useCPU = useCPU or false
        
        local fwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0, false, useCPU)
        local bwdLstm = nn.LstmLayer(nIn, nHidden, maxT, 0, true, useCPU)
        local ct = nn.ConcatTable():add(fwdLstm):add(bwdLstm)
        local blstm = nn.Sequential():add(ct):add(nn.BiRnnJoin(nHidden, nOut, maxT))
        return blstm
    end

    -- model and criterion
    local model = nn.Sequential()
    
    if not useCPU then
        model:add(nn.Copy('torch.ByteTensor', 'torch.CudaTensor', false, true))
        model:add(nn.AddConstant(-128.0))
        model:add(nn.MulConstant(1.0 / 128))
        model:add(convRelu(1))
        model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 64x16x50
        model:add(convRelu(2))
        model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 128x8x25
        model:add(convRelu(3, true))
        model:add(convRelu(4))
        model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 256x4x?
        model:add(convRelu(5, true))
        model:add(convRelu(6))
        model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 512x2x26
        model:add(convRelu(7, true))                         -- 512x1x26
        model:add(nn.View(512, -1):setNumInputDims(3))       -- 512x26
        model:add(nn.Transpose({2, 3}))                      -- 26x512
        model:add(nn.SplitTable(2, 3))
        model:add(bidirectionalLSTM(512, 256, 256, nt))
        model:add(bidirectionalLSTM(256, 256,  nl, nt))
        model:add(nn.SharedParallelTable(nn.LogSoftMax(), nt))
        model:add(nn.JoinTable(1, 1))
        model:add(nn.View(-1, nl):setNumInputDims(1))
        model:add(nn.Copy('torch.CudaTensor', 'torch.FloatTensor', false, true))
        model:add('torch.CudaTensor')
        model:cuda()
    else
        model:add(nn.Copy('torch.ByteTensor', 'torch.FloatTensor', false, true))
        model:add(nn.AddConstant(-128.0))
        model:add(nn.MulConstant(1.0 / 128))
        model:add(convRelu(1, false, true))
        --model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 64x16x50
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2))       -- 64x16x50
        model:add(convRelu(2, false, true))
        --model:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))       -- 128x8x25
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2))       -- 128x8x25
        model:add(convRelu(3, true, true))
        model:add(convRelu(4, false, true))
        --model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 256x4x?
        model:add(nn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 256x4x?
        model:add(convRelu(5, true, true))
        model:add(convRelu(6, false, true))
        --model:add(cudnn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 512x2x26
        model:add(nn.SpatialMaxPooling(2, 2, 1, 2, 1, 0)) -- 512x2x26
        model:add(convRelu(7, true, true))                         -- 512x1x26
        model:add(nn.View(512, -1):setNumInputDims(3))       -- 512x26
        model:add(nn.Transpose({2, 3}))                      -- 26x512
        model:add(nn.SplitTable(2, 3))
        model:add(bidirectionalLSTM(512, 256, 256, nt, true))
        model:add(bidirectionalLSTM(256, 256,  nl, nt, true))
        model:add(nn.SharedParallelTable(nn.LogSoftMax(), nt))
        model:add(nn.JoinTable(1, 1))
        model:add(nn.View(-1, nl):setNumInputDims(1))
    end
    local criterion = nn.CtcCriterion()

    return model, criterion
end
