function strsplit(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

function mat2euler(r)
  local cy = math.sqrt(r[3][3]*r[3][3] + r[2][3]*r[2][3])
  local z = torch.atan2(-r[1][2],  r[1][1]) -- atan2(cos(y)*sin(z), cos(y)*cos(z))
  local y = math.atan2(r[1][3],  cy) -- atan2(sin(y), cy)
  local x = math.atan2(-r[2][3], r[3][3]) -- atan2(cos(y)*sin(x), cos(x)*cos(y))
  return z*180/math.pi, y*180/math.pi, x*180/math.pi
end 

function get_rotation(r1, r2)
  local r_rel = torch.mm(r2, torch.inverse(r1))
  --local r_rel = torch.mm(torch.inverse(r2), r1)
  return mat2euler(r_rel)
end

function get_translation_old(t1, t2)
  local d = torch.add(t2, -1, t1)
  return d[1], d[2], d[3]
end

function get_translation(r1, r2, t1, t2)
  local m1 = torch.FloatTensor(4, 4)
  m1:sub(1, 3, 1, 3):copy(r1)
  m1:sub(1, 3, 4, 4):copy(torch.mul(t1, -1))
  m1:sub(4, 4, 1, 3):fill(0)
  m1:sub(4, 4, 4, 4):fill(1)

  local x1 = torch.FloatTensor(4, 1):fill(1)
  local x2 = torch.FloatTensor(4, 1):fill(1)
  x1:sub(1,3):copy(t1)
  x2:sub(1,3):copy(t2)

  local d = torch.mm(torch.inverse(m1), x2) - torch.mm(torch.inverse(m1), x1)
  return d[1][1], d[2][1], d[3][1]
end

function center_crop(x, crop)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = math.floor((x:size(2) - crop)/2)
  local sy = math.floor((x:size(3) - crop)/2)
  return image.crop(x, sy, sx, sy+crop, sx+crop)
end

function random_crop(x, crop, sx, sy)
  assert(x:dim() == 3)
  local crop = math.min(crop, math.min(x:size(2), x:size(3)))
  local sx = sx or math.random(0, x:size(2) - crop)
  local sy = sy or math.random(0, x:size(3) - crop)
  return image.crop(x, sy, sx, sy+crop, sx+crop), sx, sy
end

function interval_index(breakpoints, x)
  local idx
  local n = breakpoints:size(1)
  --[[
  if not (x >= breakpoints[1] and x < breakpoints[n]) then
    print(x .. ' outside breakpoint bounds')
  end
  --]]
  if x < breakpoints[1] then
    return 1
  elseif x >= breakpoints[n] then
    return n+1
  else
    for i = 1,n-1 do
      if x >= breakpoints[i] and x < breakpoints[i+1] then
        idx = i
        break
      end
    end
    return idx
  end
end

function adjust_meanstd(x, mean, std)
  for c = 1,3 do
    x[c]:add(-mean[c]):div(std[c])
  end
  return x
end

function normalize(x, min, max)
  local new_min, new_max = -1, 1
  local old_min, old_max = x:min(), x:max()
  local eps = 1e-7
  x:add(-old_min)
  x:mul(new_max - new_min)
  x:div(old_max - old_min + eps)
  x:add(new_min)
  return x
end

-- based on https://github.com/wojzaremba/lstm/blob/master/base.lua
function clone_many(net, T)
  local clones = {}
  local params, grads = net:parameters()
  local mem = torch.MemoryFile('w'):binary()
  mem:writeObject(net)
  for t = 1,T do
    local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
    local clone = reader:readObject()
    reader:close()
    local cloneParams, cloneGrads = clone:parameters()
    for i = 1, #params do
      cloneParams[i]:set(params[i])
      cloneGrads[i]:set(grads[i])
    end 
    clones[t] = clone
    collectgarbage() 
  end 
  mem:close()
  return clones
end

function updateConfusion(confusion, output, targets)
  local correct = 0
  for i = 1,targets:nElement() do
    if targets[i] ~= -1 then
      local _, ind = output[i]:max(1)
      confusion:add(ind[1], targets[i])
      if ind[1] == targets[i] then
        correct = correct+1
      end
    end
  end
  return correct
end

function classResults(outputs, targets)
  local top1, top5, N = 0, 0, 0
  local _, sorted = outputs:float():sort(2, true)
  for i = 1,opt.batchSize do
    if targets[i] > 0 then -- has label
      N = N+1
      if sorted[i][1] == targets[i] then
        top1 = top1 + 1
      end
      for k = 1,5 do
        if sorted[i][k] == targets[i] then
          top5 = top5 + 1
          break
        end
      end
    end
  end
  return top1, top5, N
end

function sanitize(net)
   local list = net:listModules()
   for _,val in ipairs(list) do
      for name,field in pairs(val) do
         if torch.type(field) == 'cdata' then val[name] = nil end
         if name == 'homeGradBuffers' then val[name] = nil end
         if name == 'input_gpu' then val['input_gpu'] = {} end
         if name == 'gradOutput_gpu' then val['gradOutput_gpu'] = {} end
         if name == 'gradInput_gpu' then val['gradInput_gpu'] = {} end
         if (name == 'output' or name == 'gradInput') then
            if torch.isTensor(val[name]) then
               val[name] = field.new()
            end
         end
         if  name == 'buffer' or name == 'buffer2' or name == 'normalized'
         or name == 'centered' or name == 'addBuffer' then
            val[name] = nil
         end
      end
   end
   return net
end

local function weights_init(m)
  local name = torch.type(m)
  if name:find('Convolution') or name:find('Linear') then
    m.weight:normal(0.0, 0.02)
    m.bias:fill(0)
  elseif name:find('BatchNormalization') then
    if m.weight then m.weight:normal(1.0, 0.02) end
    if m.bias then m.bias:fill(0) end
  end
end

function initModel(model)
  for _, m in pairs(model:listModules()) do
    weights_init(m)
  end
end

function scatter(x, y, fname)
  Plot = Plot or require 'itorch.Plot'
  local plot = Plot():circle(x, y)
  plot:save(fname .. '.html')
end

function hist(x, fname)
  Plot = Plot or require 'itorch.Plot'
  local plot = Plot():hist(x)
  plot:save(fname .. '.html')
end

function isNan(x)
  return x:ne(x):sum() > 0 
end

function plotTSNE(h, fname)
  local tsne = dofile('utils/tsne.lua')

  local ydata
  if h:size(2) > 2 then
    ydata = tsne(h:float())
  else
    ydata = h:float()
  end

  if isNan(ydata) then
    print('NaN in tsne results, can\'t plot...')
    return
  end
  Plot = Plot or require 'itorch.Plot'
  local ydata = ydata:t()
  local plot = Plot():circle(ydata[1], ydata[2])
  local epoch = epoch or 1
  plot:title('tsne on h (epoch ' .. epoch - 1 .. ')')
  plot:save(fname .. '.html')
  return ydata
end

function sampleNoise(z)
  if opt.noise == 'normal' then
    z:normal()
  else
    z:uniform(-1, 1)
  end
end
