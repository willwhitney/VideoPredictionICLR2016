-- print('in kitti', opt_t)

require 'image'
require 'paths'
require 'utils.base'
local res, debugger = pcall(require, 'fb.debugger')

local KITTIDataset = torch.class('KITTILoader')

if torch.getmetatable('dataLoader') == nil then
   torch.class('dataLoader')
end

local hostname = sys.execute('hostname')
local is_dgx = (hostname:find('dgx') ~= nil)
local is_vine4 = (hostname:find('vine4') ~= nil)
local dir
if is_dgx then
  dir = '/raid/cilvr/denton/KITTI/dataset'
  print('on dgx1, using: ' .. dir)
elseif is_vine4 then
  dir = '/scratch/denton/data/KITTI/dataset'
  print('on vine4, using: ' .. dir)
else
  dir = '/misc/vlgscratch3/FergusGroup/denton/KITTI/dataset'
end
  
function KITTIDataset:__init(opt, train)
  self.opt = opt or {}
  self.opt.chunkSize = opt.chunkSize or 1
  self.train = train
  self.dir = dir
  if opt.cropSize > 128 then
    self.imgtype = ''
  else
    self.imgtype = '_128x426'
  end
  self.min_d = 1 -- max distance between frames
  self.max_d = 7 -- max distance between frames
  if self.opt.chunkSize > 1 then
    self.max_d = 1
  end
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
  if self.train then
    self.indices = torch.linspace(0, 8, 9)
    self:loadPose(self.indices)
  else
    self.indices = torch.linspace(9, 10, 2)
    self:loadPose(self.indices)
  end
  if opt.meanNormalize then
    self:computeMeanStd()
  end
  self.nbins = self.opt.nClass or 2 
  if self.nbins == 2 then
    self.breakpoints_z = torch.Tensor({0})
    self.breakpoints_y = torch.Tensor({0}) 
  elseif self.nbins == 10 then
    self.breakpoints_y = torch.linspace(-6, 6, 9)
    self.breakpoints_z = torch.linspace(-10, 10, 9)
  end
  --self:computeStats(100000)
end

function KITTIDataset:processTransform(t)
  local vz = torch.Tensor(self.nbins):zero()
  local vy = torch.Tensor(self.nbins):zero()
  local z, y = t[3], t[5]
  local iz = interval_index(self.breakpoints_z, z)
  local iy = interval_index(self.breakpoints_y, y)
  vz[iz] = 1
  vy[iy] = 1
  return vz, vy, iz, iy
end

function KITTIDataset:loadPose(seq)
  self.N = 0
  self.poses = {}
  for i = 1,seq:size(1) do
    local fname = ('%s/poses/raw/%02d.txt'):format(self.dir, seq[i])
    assert(paths.filep(fname), 'directory does not exist: ' .. fname)
    local file = io.open(fname, 'r')
    local t = {}
    while true do
      local line = file:read()
      if not line then break end
      line = strsplit(line)
      local rs = {}
      for i = 1,#line do rs[i] = tonumber(line[i]) end
      table.insert(t, torch.FloatTensor(rs):reshape(3, 4))
    end
    local nl
    if self.train then
      nl = self.opt.nLabelled or 1
    else
      nl = 1
    end
    local N = math.floor(#t*nl)
    print(('sequence %d: using %d / %d labels (%d%%)'):format(seq[i], N, #t, 100*nl))
    self.N = self.N + N
    pose = {}
    pose.labelledN = N
    pose.N = #t
    pose.rotation = torch.FloatTensor(N, 3, 3)
    pose.translation = torch.FloatTensor(N, 3)
    for i = 1,N do
      pose.rotation[i]:copy(t[i]:sub(1, 3, 1, 3))
      pose.translation[i]:copy(t[i]:sub(1, 3, 4, 4))
    end
    self.poses[i] = pose
  end
end

function KITTIDataset:computeMeanStd(nex)
  if paths.filep(self.dir .. '/meanstd.t7') then
    local tmp = torch.load(self.dir .. '/meanstd.t7')
    self.mean = tmp.mean
    self.std = tmp.std
    print('mean = ' .. self.mean[1] .. ' ' .. self.mean[2] .. ' ' .. self.mean[3])
    print('std =  ' .. self.std[1] .. ' ' .. self.std[2] .. ' ' .. self.std[3])
  else
    local nex = nex or 10000
    local mean = torch.Tensor(3):zero()
    local std = torch.Tensor(3):zero()
    print('computing mean...')
    for i = 1, nex/2 do
      xlua.progress(i, nex/2)
      local im1, im2 = self:getUnlabelled()
      for c = 1,3 do
        mean[c] = mean[c] + im1[c]:mean()
        mean[c] = mean[c] + im2[c]:mean()
      end
    end
    mean:div(nex)
    print('computing std...')
    for i = 1, nex/2 do
      xlua.progress(i, nex/2)
      local im1, im2 = self:getUnlabelled()
      for c = 1,3 do
        std[c] = std[c] + im1[c]:add(-mean[c]):std()
        std[c] = std[c] + im2[c]:add(-mean[c]):std()
      end
    end
    std:div(nex)
    print('mean = ' .. mean[1] .. ' ' .. mean[2] .. ' ' .. mean[3])
    print('std =  ' .. std[1] .. ' ' .. std[2] .. ' ' .. std[3])
    self.mean = mean
    self.std = std
    torch.save(self.dir .. '/meanstd.t7', {mean=mean, std=std})
  end
end

function KITTIDataset:normalize(x)
  if self.opt.meanNormalize then
    return adjust_meanstd(x, self.mean, self.std)
  else
    return normalize(x)
  end
end

function KITTIDataset:processSingle(im, cx, cy)
  local pim, cx, cy = random_crop(im, self.opt.cropSize, cx, cy)
  if self.opt.imageSize == self.opt.cropSize then
    pim = self:normalize(pim)
  else
    pim = self:normalize(image.scale(pim, self.opt.imageSize, self.opt.imageSize))
  end
  return pim, cx, cy
end

function KITTIDataset:process(im1, im2)
  local pim1, cx, cy = random_crop(im1, self.opt.cropSize)
  local pim2 = random_crop(im2, self.opt.cropSize, cx, cy)
  if self.opt.imageSize == self.opt.cropSize then
    pim1 = self:normalize(pim1)
    pim2 = self:normalize(pim2)
  else
    pim1 = self:normalize(image.scale(pim1, self.opt.imageSize, self.opt.imageSize))
    pim2 = self:normalize(image.scale(pim2, self.opt.imageSize, self.opt.imageSize))
  end
  return pim1, pim2
end

function KITTIDataset:getBatch(n)
  local x1 = torch.Tensor(n, unpack(self.opt.geometry))
  local x2 = torch.Tensor(n, unpack(self.opt.geometry))
  for i = 1,n do
    local im1, im2 = self:getUnlabelled() 
    im1, im2 = self:process(im1, im2)
    x1[i]:copy(im1)
    x2[i]:copy(im2)
  end 
  return {x1, x2}
end

function KITTIDataset:getLabelledBatch(n)
  local x1 = torch.Tensor(n, unpack(self.opt.geometry))
  local x2 = torch.Tensor(n, unpack(self.opt.geometry))
  local tz = torch.Tensor(n)
  local ty = torch.Tensor(n)
  for i = 1,n do
    local im1, im2, t = self:getLabelled() 
    im1, im2 = self:process(im1, im2)
    x1[i]:copy(im1)
    x2[i]:copy(im2)
    local _, _, iz, iy = self:processTransform(t)
    tz[i] = iz
    ty[i] = iy
  end 
  return x1, x2, {tz, ty}
end

function KITTIDataset:getChunkBatch(n)
  local x1 = torch.Tensor(n, unpack(self.opt.t_geometry))
  local x2 = torch.Tensor(n, unpack(self.opt.t_geometry))
  local x = self:getSeqBatch(n, self.opt.chunkSize*2)
  for i = 1,n do
    for t = 1,self.opt.chunkSize do
      x1[{ i, {}, t }]:copy(x[t][i])
    end
    for t = self.opt.chunkSize+1,self.opt.chunkSize*2 do
      x2[{ i, {}, t-self.opt.chunkSize }]:copy(x[t][i])
    end
  end
  return x1, x2
end

function KITTIDataset:getSeqBatch(n, t)
  local x = torch.Tensor(t, n, unpack(self.opt.geometry))
  for i = 1,n do
    local d = math.random(self.min_d, self.max_d)
    if torch.uniform() > 0.5 then
      d = -d
    end
    local s = math.random(#self.poses)
    local i1 = math.random(math.max(1, -d*t + 1), math.min(self.poses[s].N - d*t, self.poses[s].N))
    local k = 0
    local im, cx, cy
    for j = 1,t do
      local im, _, _ = self:getSingle(s, i1+k) --1, 1000+(i-1)*5+1, 5) --self:get()
      k = k+d
      im, cx, cy = self:processSingle(im, cx, cy)
      x[j][i]:copy(im)
    end
  end
  return x
end

function KITTIDataset:getLabels(s, i1, d)
  local d = d or math.random(self.min_d, self.max_d)
  if torch.uniform() > 0.5 then
    d = -d
  end
  local s = s or math.random(#self.poses)
  local i1 = i1 or math.random(math.max(1, -d + 1), math.min(self.poses[s].N - d, self.poses[s].N))
  local i2 = i1 + d
  local t = self:getRelativeAction(s, i1, i2)
  return t
end

function KITTIDataset:getSingle(s, i)
  local d = d or math.random(self.min_d, self.max_d)
  if torch.uniform() > 0.5 then
    d = -d
  end
  local s = s or math.random(#self.poses)
  local i = i or math.random(math.max(1, -d + 1), math.min(self.poses[s].N - d, self.poses[s].N))
  -- subtract 1 since 0-indexed
  local path = ('%s/sequences%s/%02d/image_2/%06d.png'):format(self.dir, self.imgtype, self.indices[s], i-1)
  local im = image.load(path)
  return im
end

function KITTIDataset:getLabelled(s, i1, d)
  local d = d or math.random(self.min_d, self.max_d)
  if torch.uniform() > 0.5 then
    d = -d
  end
  local s = s or math.random(#self.poses)
  local i1 = i1 or math.random(math.max(1, -d + 1), math.min(self.poses[s].labelledN - d, self.poses[s].labelledN))
  local i2 = i1 + d
  -- subtract 1 since 0-indexed
  local path1 = ('%s/sequences%s/%02d/image_2/%06d.png'):format(self.dir, self.imgtype, self.indices[s], i1-1)
  local path2 = ('%s/sequences%s/%02d/image_2/%06d.png'):format(self.dir, self.imgtype, self.indices[s], i2-1)
  local im1 = image.load(path1)
  local im2 = image.load(path2)
  local t = self:getRelativeAction(s, i1, i2)
  return im1, im2, t
end

function KITTIDataset:getUnlabelled(s, i1, d)
  local d = d or math.random(self.min_d, self.max_d)
  if torch.uniform() > 0.5 then
    d = -d
  end
  local s = s or math.random(#self.poses)
  local i1 = i1 or math.random(math.max(1, -d + 1), math.min(self.poses[s].N - d, self.poses[s].N))
  local i2 = i1 + d
  -- subtract 1 since 0-indexed
  local path1 = ('%s/sequences%s/%02d/image_2/%06d.png'):format(self.dir, self.imgtype, self.indices[s], i1-1)
  local path2 = ('%s/sequences%s/%02d/image_2/%06d.png'):format(self.dir, self.imgtype, self.indices[s], i2-1)
  local im1 = image.load(path1)
  local im2 = image.load(path2)
  return im1, im2, t
end

function KITTIDataset:getAction(s, i)
  local s = s or math.random(#self.poses)
  local i = i or math.random(self.poses[s].labelledN)
  local t = self.poses[s].translation[i]
  local r = self.poses[s].rotation[i]
  local tx, ty, tz = t[1], t[2], t[3]
  local rx, ry, rz = mat2euler(r)
  return {tx, ty, tz, rx, ry, rz}
end

function KITTIDataset:getRelativeAction(s, i1, i2)
  local t1 = self.poses[s].translation[i1]
  local t2 = self.poses[s].translation[i2]
  local r1 = self.poses[s].rotation[i1]
  local r2 = self.poses[s].rotation[i2]
  local tx, ty, tz = get_translation(r1, r2, t1, t2)
  local rx, ry, rz = get_rotation(r1, r2)
  local t = {tx, ty, tz, rx, ry, rz}
  return t
end

function KITTIDataset:size()
   return self.N
end

function KITTIDataset:computeStats(N)
  assert(false)
  local N = N or 1000
  local tx = torch.Tensor(N)
  local tz = torch.Tensor(N)
  local ry = torch.Tensor(N)
  for n = 1,N do
    xlua.progress(n, N)
    local t = self:getLabels()
    tx[n] = t[1]
    tz[n] = t[2]
    ry[n] = t[3]
  end
  print(('tx: mean = %.3f, std = %.5f, min = %.3f, max = %.3f, scale = %.3f'):format(tx:mean(), tx:std(), tx:min(), tx:max(), tx:std()/tx:max()))
  print(('tz: mean = %.3f, std = %.5f, min = %.3f, max = %.3f'):format(tz:mean(), tz:std(), tz:min(), tz:max(), tz:std()/tz:max()))
  print(('ry: mean = %.3f, std = %.5f, min = %.3f, max = %.3f'):format(ry:mean(), ry:std(), ry:min(), ry:max(), ry:std()/ry:max()))
  local max_std = torch.Tensor({tx:std(), tz:std(), ry:std()}):max()
  self.stats = {{mu=tx:mean(), std=tx:std(), scale=tx:std()/max_std},
                {mu=tz:mean(), std=tz:std(), scale=tz:std()/max_std},
                {mu=ry:mean(), std=ry:std(), scale=ry:std()/max_std}}
  return tx, tx, ry
end

function KITTIDataset:plotChunks(fname)
  if not self.t_geometry then return end
  print('plotting chunks: ' .. fname)
  local to_plot = {}
  local n = 20
  local x1, x2 = self:getChunkBatch(n)
  for i = 1,n do
    for j = 1,self.opt.chunkSize do
      table.insert(to_plot, x1[{ i, {}, j }]:float())
    end
    for j = 1,self.opt.chunkSize do
      table.insert(to_plot, x2[{ i, {}, j }]:float())
    end
  end 
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=self.opt.chunkSize*2})
end

function KITTIDataset:plotSeq(fname)
  print('plotting sequence: ' .. fname)
  local to_plot = {}
  local t = 10
  local n = 20
  local x = self:getSeqBatch(n, t)
  for i = 1,n do
    for j = 1,t do
      table.insert(to_plot, x[j][i])
    end
  end 
  image.save(fname, image.toDisplayTensor{input=to_plot, scaleeach=true, nrow=t})
end

function KITTIDataset:plotActions(fname)
  print('plotting absolute actions: ' .. fname)
  for s = 1,#self.poses do
    local tx = torch.FloatTensor(self.poses[s].labelledN)
    local tz = torch.FloatTensor(self.poses[s].labelledN)
    local ry = torch.FloatTensor(self.poses[s].labelledN)
    for i = 1,self.poses[s].labelledN do
      local t = self:getAction(s, i)
      tx[i] = t[1]
      tz[i] = t[3]
      ry[i] = t[5]
    end 
    scatter(tx, tz, fname .. 'abs_translation_' .. s) 
    scatter(torch.linspace(1, self.poses[s].labelledN, self.poses[s].labelledN), ry, fname .. 'abs_rotation_' .. s) 
  end
end

function KITTIDataset:plotRelativeActions(fname)
  print('plotting relative actions: ' .. fname)
  for s = 1,#self.poses do
    local tx = torch.FloatTensor(self.poses[s].labelledN-1)
    local tz = torch.FloatTensor(self.poses[s].labelledN-1)
    local ry = torch.FloatTensor(self.poses[s].labelledN-1)
    for i = 1,self.poses[s].labelledN-1 do
      local t = self:getRelativeAction(s, i, i+1)
      tx[i] = t[1]
      tz[i] = t[3]
      ry[i] = t[5]
    end 
    scatter(tx, tz, fname .. 'rel_translation_' .. s) 
    scatter(torch.linspace(1, self.poses[s].labelledN-1, self.poses[s].labelledN-1), ry, fname .. 'rel_rotation_' .. s) 
  end
end

function KITTIDataset:plotTargetHist(fname)
  print('plotting target hist: ' .. fname)
  local y, z = {}, {}, {}
  local K = 10 --500
  for i = 1,K do
    xlua.progress(i, K)
    local _, _, t = self:getLabelledBatch(1)
    table.insert(z, t[1][1])
    table.insert(y, t[2][1])
  end
  hist(torch.FloatTensor(y), fname .. 'y_target_hist') 
  hist(torch.FloatTensor(z), fname .. 'z_target_hist') 
end

function KITTIDataset:plotMajorActions(fname)
  print('plotting major actions: ' .. fname)
  local ry, tz = {}, {}
  for s = 1,#self.poses do
    for d = -7, 7 do
      for i = 10,self.poses[s].labelledN-10,5 do
        local t = self:getRelativeAction(s, i, i+d)
        table.insert(tz, t[3])
        table.insert(ry, t[5])
      end 
    end
  end
  scatter(torch.FloatTensor(tz), torch.FloatTensor(ry), fname .. 'major_transforms') 
end

function KITTIDataset:plotActionHist(fname)
  print('plotting action hist: ' .. fname)
  local tx, ty, tz = {}, {}, {}
  local rx, ry, rz = {}, {}, {}
  for s = 1,#self.poses do
    for d = self.min_d, self.max_d do
      for i = d+1,self.poses[s].labelledN-d do
        local mul; if torch.uniform() > 0.5 then mul = 1 else mul = -1 end
        local t = self:getRelativeAction(s, i, i+(d*mul))
        table.insert(tx, t[1])
        table.insert(ty, t[2])
        table.insert(tz, t[3])
        table.insert(rx, t[4])
        table.insert(ry, t[5])
        table.insert(rz, t[6])
      end
    end 
    hist(torch.FloatTensor(tx), fname .. 'tx_hist') 
    hist(torch.FloatTensor(ty), fname .. 'ty_hist') 
    hist(torch.FloatTensor(tz), fname .. 'tz_hist') 
    hist(torch.FloatTensor(rx), fname .. 'rx_hist') 
    hist(torch.FloatTensor(ry), fname .. 'ry_hist') 
    hist(torch.FloatTensor(rz), fname .. 'rz_hist') 
  end
end

function KITTIDataset:plot()
  local savedir = self.opt.save or '/home/denton/'
  savedir = savedir .. '/data/'
  os.execute('mkdir -p ' .. savedir)
  self:plotActionHist(savedir)
  self:plotTargetHist(savedir)
  self:plotMajorActions(savedir)
  self:plotSeq(savedir .. '/seq.png')
  self:plotChunks(savedir .. '/chunks.png')
  self:plotActions(savedir)
  self:plotRelativeActions(savedir)
end

trainLoader = KITTILoader(opt_t, true)
valLoader = KITTILoader(opt_t, false)
