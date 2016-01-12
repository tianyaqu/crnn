require('cutorch')
require('nn')
require('optim')
require('paths')
require('nngraph')

require('libcrnn')
require('utilities')
require('inference')
require('CtcCriterion')
require('DatasetLmdb')
require('LstmLayer')
require('BiRnnJoin')
require('SharedParallelTable')

if not gConfig.useCPU then
    require('cutorch')
    require('cunn')
    require('cudnn')
    
    cutorch.setDevice(1)
end

cutorch.setDevice(1)
torch.setnumthreads(4)
torch.setdefaulttensortype('torch.FloatTensor')

print('Loading model...')
local modelDir = '../model/crnn_demo/'
paths.dofile(paths.concat(modelDir, 'config.lua'))
local modelLoadPath = paths.concat(modelDir, 'crnn_demo_model.t7')
gConfig = getConfig()
gConfig.modelDir = modelDir
gConfig.maxT = 0
local model, criterion = createModel(gConfig)
local snapshot = torch.load(modelLoadPath)
loadModelState(model, snapshot)
model:evaluate()
print(string.format('Model loaded from %s', modelLoadPath))

local imagePath = '../data/demo.png'
local img = loadAndResizeImage(imagePath)
local text, raw = recognizeImageLexiconFree(model, img)
print(string.format('Recognized text: %s (raw: %s)', text, raw))

