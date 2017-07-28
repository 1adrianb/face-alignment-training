local image = require('image')
require 'utils'

local M = {}
local DatasetImages = torch.class('DatasetImages', M)

function DatasetImages:__init( opt, split, annot )
    self.total = #annot
    self.nParts = 68
    self.annot = annot
    self.opt = opt
    self.typeOfData = split
end

function DatasetImages:get(shuffle,i)
    local inp, out, pts, c, s = generateSampleFace(i,self)
    self.pts, self.c, self.s = pts,c,s
    return inp, out
end

function DatasetImages:size()
    return self.total
end

function DatasetImages:preprocess(input, label)
    if self.typeOfData == 'train'  then
        local s = torch.randn(1):mul(self.opt.scaleFactor):add(1):clamp(1-self.opt.scaleFactor,1+self.opt.scaleFactor)[1]
        local r = torch.randn(1):mul(self.opt.rotFactor):clamp(-2*self.opt.rotFactor,2*self.opt.rotFactor)[1]

        -- Scale/rotation
        if torch.uniform() <= .6 then r = 0 end
        local inp,out = self.opt.inputRes, self.opt.outputRes
        local divideBy = 200

        input = crop(input, {(inp+1)/2,(inp+1)/2}, inp*s/divideBy, r, inp)
        label = crop(label, {(out+1)/2,(out+1)/2}, out*s/divideBy, r, out)

        -- Emulate row resolution
        if torch.uniform()<=.2 and false then --.35
            input = image.scale(input,96,96)
            input = image.scale(input,256,256)
        end

        -- Add jpeg artefacts
        --[[
        if torch.uniform()<=.2 and false then
            local onlyImg = input[{{1,3},{},{}}]
            onlyImg = image.compressJPG(onlyImg,30)
            onlyImg = image.decompressJPG(onlyImg)
            input[{{1,3},{},{}}] = onlyImg
        end
        ]]--

        -- Add random translation
        --[[
        wh_t = torch.Tensor(2):random(0,80)-40
        input = image.translate(input,wh_t[1],wh_t[2])
        label = image.translate(label,wh_t[1]/4.0,wh_t[2]/4.0)
        ]]--

        -- Add some gaussian blue
        --[[
        if torch.uniform()<.4 and false  then
            gauss_s = torch.Tensor(1):random(10,30):int()
            local kernel_gauss = image.gaussian(gauss_s[1])
            input = image.convolve(input, kernel_gauss, 'same')/255.0
        end
        ]]--

        local flip_ = customFlip or flip

        local shuffleLR_ = customShuffleLR or shuffleLR
        if torch.uniform() <= .5 then
            input = flip_(input)
            label = flip_(shuffleLR_(label))
        end

        -- Color augumentation
        input[{1, {}, {}}]:mul(torch.uniform(0.7, 1.3)):clamp(0, 1)
        input[{2, {}, {}}]:mul(torch.uniform(0.7, 1.3)):clamp(0, 1)
        input[{3, {}, {}}]:mul(torch.uniform(0.7, 1.3)):clamp(0, 1)
    end
    return input, label
end

return M.DatasetImages
