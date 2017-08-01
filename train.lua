require 'cunn'
local optim = require 'optim'

local lr_policy = {
    {0,50,2.5e-4},
    {50,70,1e-4},
    {70,90,5e-5},
    {90,100,1e-5},
    {100,110,5e-6}
}

local M = {}
local Trainer = torch.class('Trainer', M)

function Trainer:__init(model,criterion,opt,optimState)
        self.model = model
        self.criterion = criterion
        self.optimState = optimState or {
                learningRate = opt.LR,
                learningRateDecay = 0.0,
                momentum = opt.momentum,
                epsilon = 1e-8,
                weightDecay = opt.weightDecay,
        }

        self.opt = opt

        self.params, self.gradParams = model:getParameters()
end

function Trainer:train(epoch, dataloader)
        local avgLoss, avgAcc = 0.0, 0.0
        self.optimState.learningRate = self:learningRate(epoch)

        local timer = torch.Timer()
        local dataTimer = torch.Timer()

        local function feval()
                return self.criterion.output, self.gradParams
        end

        local trainSize = dataloader:size()
        local N = 0

        print('=> Training epoch # '..epoch)

        self.model:training()

        for n, sample in dataloader:run() do
                local dataTime = dataTimer:time().real
                self:copyInputs(sample)

                self.model:zeroGradParameters()
                local output = self.model:forward(self.input)

                local loss = self.criterion:forward(output, self.label)

                self.criterion:backward(self.model.output, self.label)

                self.model:backward(self.input,self.criterion.gradInput)

                optim.rmsprop(feval, self.params, self.optimState)
                
                avgLoss = avgLoss + loss
                N = N + 1

                print((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f'):format(
                        epoch, n, trainSize, timer:time().real, dataTime, loss))

                -- check that the storage didn't get changed do to an unfortunate getParameters call
                assert(self.params:storage() == self.model:parameters()[1]:storage())
                collectgarbage()
                timer:reset()
                dataTimer:reset()
        end

        return avgLoss / N, avgAcc / N
end

function Trainer:learningRate(epoch)
        local decay = 0
        for i=1, #lr_policy do
                if (epoch>lr_policy[i][1]) and (lr_policy[i][2]>=epoch) then
                        print(string.format('Using lr_rate: %f',lr_policy[i][3]))
                        return lr_policy[i][3]
                end
        end
end

function Trainer:copyInputs(sample)
    -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
    -- if using DataParallelTable. The target is always copied to a CUDA tensor
    self.input = self.input or (self.opt.nGPU == 1
      and torch.CudaTensor()
      or cutorch.createCudaHostTensor())
    label = label or torch.CudaTensor()

    self.input:resize(sample.input[{{},{},{},{}}]:size()):copy(sample.input[{{},{},{},{}}])
    label:resize(sample.label:size()):copy(sample.label)

    -- Adjust the input accordingly to the network arhitecture 
    if self.opt.nStacks>1 then
        local tempLabel = {}
        for i=1,self.opt.nStacks do
            table.insert(tempLabel, label)
        end

        self.label = tempLabel
    else 
        self.label = label
    end
end

return M.Trainer



