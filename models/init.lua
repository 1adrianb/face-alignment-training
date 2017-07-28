local M = {}

function  M.setup(opt, checkpoint)
    local model 
    if checkpoint then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model = torch.load(modelPath)
    elseif opt.retrain ~= 'none' then
        local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
        assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
        print('=> Resuming model from ' .. modelPath)
        model = torch.load(modelPath)
        if preprocess == false then
            return model, nil
        end
    else
        print('=> Creating model from file: models/' .. opt.netType .. '.lua')
        model = require('models/' .. opt.netType)(opt)
    end

    if torch.type(model) == 'nn.DataParallelTable' then
        model = model:get(1)
    end

    -- Set the CUDNN flags
    if opt.cudnn == 'fastest' then
        cudnn.fastest = true
        cudnn.benchmark = true
    elseif opt.cudnn == 'deterministic' then
        -- Use a deterministic convolution implementation
        model:apply(function(m)
            if m.setMode then m:setMode(1, 1, 1) end
        end)
    end

    if opt.nGPU > 1 then
        local gpus = torch.range(1, opt.nGPU):totable()
        local fastest, benchmark = cudnn.fastest, cudnn.benchmark

        local dpt = nn.DataParallelTable(1, true, true)
            :add(model, gpus)
            :threads(function()
                local cudnn = require 'cudnn'
                require 'nngraph'
                require 'newLayers.BinActiveZ'
                cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end)
        dpt.gradInput = nil

        model = dpt:cuda()
    end

    local criterion
    if opt.nStacks>1 then
        criterion = nn.ParallelCriterion()
        for i=1,opt.nStacks do
            criterion:add(nn.MSECriterion())
        end
    else
        criterion = nn.MSECriterion()
    end

    return model:cuda(), criterion:cuda()
end

return M