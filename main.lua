require 'torch'
require 'paths'
require 'nn'
require 'nngraph'

local DataLoader = require 'datalaoder'
local checkpoints = require 'checkpoints'
local models = require 'models/init'
local Trainer = require 'train'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
cutorch.setDevice(1)
torch.setheaptracking(true)

torch.manualSeed(opt.manualSeed)
cutorch.manualSeed(opt.manualSeed)

--Load previous checkpoints, if it exists
local checkpoint, optimState = checkpoints.latest(opt)
local optimState = checkpoint and torch.load(checkpoint.optimFile) or nil

--Create model
local model, criterion = models.setup(opt, checkpoint, true)

print('=> Model size: ', model:getParameters():size(1))

--Data loading
local trainLoader = DataLoader.create(opt,'train')

local trainer  = Trainer(model, criterion, opt, optimState, netLogger)

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber

for epoch = startEpoch, opt.nEpochs do
        -- Train for a single epoch
        local trainLoss, trainAcc = trainer:train(epoch, trainLoader)
        print(string.format(' *Results loss: %6.6f acc: %6.6f ',trainLoss, trainAcc))

        if opt.snapshot ~= 0 and epoch % opt.snapshot == 0 then
                checkpoints.save(epoch, model:clearState(), trainer.optimState, bestModel)
        end
end


