-- Face Alignment Network
--
-- How far are we from solving the 2D \& 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)
-- Adrian Bulat and Georgios Tzimiropoulos
-- ICCV 2017
--

local cudnn = require 'cudnn'

-- Define some short names
local conv = cudnn.SpatialConvolution
local batchnorm = nn.SpatialBatchNormalization
local relu = cudnn.ReLU
local upsample = nn.SpatialUpSamplingNearest

-- Opts
local nModules = 1
local nFeats = 256
local nStack = 8


local function convBlock(numIn, numOut, order)
    local cnet = nn.Sequential()
        :add(batchnorm(numIn,1e-5,false))
        :add(relu(true))
        :add(conv(numIn,numOut/2,3,3,1,1,1,1):noBias())
        :add(nn.ConcatTable()
            :add(nn.Identity())
            :add(nn.Sequential()
                :add(nn.Sequential()
                    :add(batchnorm(numOut/2,1e-5,false))
                    :add(relu(true))
                    :add(conv(numOut/2,numOut/4,3,3,1,1,1,1):noBias())
                )
                :add(nn.ConcatTable()
                    :add(nn.Identity())
                    :add(nn.Sequential()
                        :add(batchnorm(numOut/4,1e-5,false))
                        :add(relu(true))
                        :add(conv(numOut/4,numOut/4,3,3,1,1,1,1):noBias())
                    )
                )
                :add(nn.JoinTable(2))
            )
        )
        :add(nn.JoinTable(2))
    return cnet
end

-- Skip layer
local function skipLayer(numIn,numOut)
    if numIn == numOut  then
        return nn.Identity()
    else
        return nn.Sequential()
            :add(batchnorm(numIn,1e-5,false))
            :add(relu(true))
            :add(conv(numIn,numOut,1,1):noBias())
    end
end

-- Residual block
local function Residual(numIn,numOut)
    return nn.Sequential()
        :add(nn.ConcatTable()
            :add(convBlock(numIn,numOut))
            :add(skipLayer(numIn,numOut)))
        :add(nn.CAddTable(true))
end

local function lin(numIn,numOut,inp)
    -- Apply 1x1 convolution, stride 1, no padding
    local l = conv(numIn,numOut,1,1,1,1,0,0)(inp)
    return relu(true)(batchnorm(numOut)(l))
end

local function hourglass(n, f)
        local model = nn.Sequential()

        local branch = nn.ConcatTable()
        local b1 = nn.Sequential()
        local b2 = nn.Sequential()

        for i = 1,nModules do b1:add(Residual(f,f)) end
        b2:add(nn.SpatialMaxPooling(2,2,2,2))

        if n>1 then
                for i = 1,nModules do b2:add(Residual(f,f)) end
        else
                for i = 1,nModules do b2:add(Residual(f,f)) end
        end

        if n>1 then
                b2:add(hourglass(n-1,f))
        else
                for i = 1,nModules do b2:add(Residual(f,f)) end
        end

        if n>1 then
                for i = 1,nModules do b2:add(Residual(f,f)) end
        else
                for i=1,nModules do b2:add(Residual(f,f)) end
        end
        b2:add(upsample(2))

        branch:add(b1):add(b2)
        model:add(branch)

        return model:add(nn.CAddTable())
end

local function hourglass(n, f, inp)
    -- Upper branch
    local up1 = inp
    for i = 1,nModules do up1 = Residual(f,f)(up1) end

    -- Lower branch
    local low1 = cudnn.SpatialMaxPooling(2,2,2,2)(inp)
    for i = 1,nModules do low1 = Residual(f,f)(low1) end
    local low2

    if n > 1 then low2 = hourglass(n-1,f,low1)
    else
        low2 = low1
        for i = 1,nModules do low2 = Residual(f,f)(low2) end
    end

    local low3 = low2
    for i = 1,nModules do low3 = Residual(f,f)(low3) end
    local up2 = nn.SpatialUpSamplingNearest(2)(low3)

    -- Bring two branches together
    return nn.CAddTable()({up1,up2})
end



function createModel(opt)
    nModules = opt.nModules
    nFeats = opt.nFeats
    nStack = opt.nStacks

    local inp = nn.Identity()()

    -- Initial processing of the image
    local cnv1_ = conv(3,64,7,7,2,2,3,3)(inp)           -- 128
    local cnv1 = relu(true)(batchnorm(64)(cnv1_))
    local r1 = Residual(64,128)(cnv1)
    local pool = nn.SpatialMaxPooling(2,2,2,2)(r1)                       -- 64
    local r4 = Residual(128,128)(pool)
    local r5 = Residual(128,nFeats)(r4)

    local out = {}
    local inter = r5

    for i = 1,nStack do
        local hg = hourglass(4,nFeats,inter)

        -- Residual layers at output resolution
        local ll = hg
        for j = 1,nModules do ll = Residual(nFeats,nFeats)(ll) end
        -- Linear layer to produce first set of predictions
        ll = lin(nFeats,nFeats,ll)

        -- Predicted heatmaps
        local tmpOut = conv(nFeats,68,1,1,1,1,0,0)(ll)
        table.insert(out,tmpOut)

        -- Add predictions back
        if i < nStack then
            local ll_ = conv(nFeats,nFeats,1,1,1,1,0,0)(ll)
            local tmpOut_ = conv(68,nFeats,1,1,1,1,0,0)(tmpOut)
            inter = nn.CAddTable()({inter, ll_, tmpOut_})
        end
    end

    -- Final model
    local model = nn.gModule({inp}, out)

    return model

end

return createModel



