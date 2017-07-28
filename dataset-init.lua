local M = {}

function M.create(opt, split, annot)
   local Dataset = require('dataset-images')
   return Dataset(opt, split, annot)
end

return M





