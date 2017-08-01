local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('2D-FAN and 3D-FAN Training script')
   cmd:text('Visit https://www.adrianbulat.com for more details')
   cmd:text()
   cmd:text('Options:')
    ------------ General options --------------------
   cmd:option('-data',       'dataset/300W-LP/',         'Path to dataset')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       1,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   cmd:option('-snapshot',    3, 'save a snapshot every n epochs')
   ------------- Data options ------------------------
   cmd:option('-nThreads',        2, 'number of data loading threads')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         100,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       10,      'mini-batch size (1 = pure stochastic)')
   ------------- Checkpointing options ---------------
   cmd:option('-save',            'checkpoints', 'Directory in which to save checkpoints')
   cmd:option('-resume',          'none',        'Resume from the latest checkpoint in this directory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',              0.00025,   'initial learning rate')
   cmd:option('-momentum',        0.0,   'momentum')
   cmd:option('-weightDecay',     0.0,  'weight decay')
   ---------- Model options ----------------------------------
   cmd:option('-netType',      'fan', 'Options: fan')
   cmd:option('-nModules',       1,       'Number of modues per level')
   cmd:option('-nStacks',         4,       'Number of stacked networks')
   cmd:option('-nFeats',         256,     'BLock width (# channels)')

   cmd:option('-retrain',      'none',   'Path to model to retrain with')
   cmd:option('-optimState',   'none',   'Path to an optimState to reload from')
   ---------- Augumentation options ----------------------------------
   cmd:option('-scaleFactor',        0.3,   'scaling factor')
   cmd:option('-rotFactor',        30,   'rotation factor (in degrees)')

   cmd:text()

   local opt = cmd:parse(arg or {})

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   return opt
end

return M