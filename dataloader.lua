local datasets = require 'dataset-init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('DataLoader', M)

function DataLoader.getDataFaces(opt, split)
    print('=> Building dataset...')
    base_dir = '/mnt/rogue/udisk/psxasj/db/3DDFA-300W/300W_LP/landmarks/'
    dirs = paths.dir(base_dir)
    lines = {}

    helentest = {}
    for f in paths.files('/mnt/rogue/db1/psxab5/faces/testset/','.pts') do
                helentest[#helentest+1] = f:sub(1,#f-4)
    end

    for i=1,#dirs do
        if string.sub(dirs[i],1,1) ~= '.' then
            for f in paths.files(base_dir..dirs[i],'.mat') do
                if not string.find(f, "test") then
                    addme = true;
                    for j=1,#helentest do
                        if string.find(f,helentest[j]) then
                            addme=false
                            break
                        end
                    end
                    if addme then
                        lines[#lines+1] = f
                    end
                end
            end
        end
    end
    print('=> Dataset built. '..#lines..' images were found.')
    return lines
end

function DataLoader.create(opt,split)
    local _dataset = nil
    local dataAnnot = DataLoader.getDataFaces(opt,split) 

    return M.DataLoader(_dataset,opt,split,dataAnnot)
end

function DataLoader:__init(_dataset, opt, split, dataAnnot)
    local manualSeed = opt.manualSeed

    local function init()
        local datasets = require 'dataset-init'

        trainLoader, valLoader = datasets.create(opt,split,dataAnnot)
    end

    local function main(idx)
        if manualSeed ~= 0 then
            torch.manualSeed(manualSeed + idx)
        end
        torch.setnumthreads(1)
        _G.dataset = trainLoader
        return trainLoader:size()
    end

    local threads, sizes = Threads(opt.nThreads,init, main)
    self.nCrops = 1
    self.threads = threads
    self.__size = sizes[1][1]
    self.batchSize = math.floor(opt.batchSize / self.nCrops)
    self.opt = opt
    self.split = split
    self.dataAnnot = dataAnnot
end

function DataLoader:size()
        return math.ceil(self.__size/self.batchSize)
end

function DataLoader:annot()
        trainLoader = datasets.create(self.opt,self.split,self.dataAnnot)
        return  trainLoader.annot
end

function DataLoader:run()
    local threads = self.threads
    local size, batchSize = self.__size, self.batchSize
    local perm = torch.randperm(size)

    local idx, sample = 1, nil
    local function enqueue()
        while idx <= size and threads:acceptsjob() do
            local indices = perm:narrow(1,idx,math.min(batchSize,size-idx+1))
            threads:addjob(
                function(indices,nCrops)
                    local sz = indices:size(1)
                    local batch, imageSize
                    local target
                    local indicesCopy = indices
                    for i,idx in ipairs(indices:totable()) do
                        local sample, label = _G.dataset:get(false,idx)
                        local input, label = _G.dataset:preprocess(sample, label)
                        if not batch then
                            imageSize = input:size():totable()
                            if nCrops > 1 then table.remove(imageSize,1) end
                                batch = torch.FloatTensor(sz,nCrops, table.unpack(imageSize))
                            end
                            if not target then
                                targetSize = label:size():totable()
                                target = torch.FloatTensor(sz,nCrops, table.unpack(targetSize))
                            end
                            batch[i]:copy(input)
                            target[i]:copy(label)
                        end
                        collectgarbage()
                        return {
                            input = batch:view(sz*nCrops,table.unpack(imageSize)),
                            label = target:view(sz*nCrops,table.unpack(targetSize)),
                            indx = indicesCopy ,
                        }
                        end,
                        function(_sample_)
                            sample = _sample_
                            end,
                            indices,
                            self.nCrops
                        )
                        idx = idx + batchSize
                end
        end
    local n = 0
    local function loop()
        enqueue()
        if not threads:hasjob() then
            return nil
        end
        threads:dojob()
        if threads:haserror() then
            threads:synchronize()
        end
        enqueue()
        n = n+1
        return n, sample
    end

    return loop
end
return M.DataLoader
