--[[
   Note that it costs time to switch from set (train/test/valid)
   and change the batch size. If you intend to do it a lot, create
   multiple instances of datasources, with constant set/batchSize
   params:
   nDonkeys [4]
--]]

require 'datasources.datasource'
local threads = require 'threads'

local ThreadedDatasource, parent = torch.class('ThreadedDatasource', 'ClassDatasource')

function ThreadedDatasource:__init(getDatasourceFun, params)
   parent.__init(self)
   self.nDonkeys = params.nDonkeys or 4
   local opt = opt

   self.batch_pool_size = 1000
   self.batch_pool = {}
   --threads.Threads.serialization('threads.sharedserialize') --TODO
   self.donkeys = threads.Threads(self.nDonkeys,
      function(threadid)
         require 'torch'
         require 'math'
         require 'os'
         opt_t = opt
         -- print(opt_t)
         torch.manualSeed(threadid*os.clock())
         math.randomseed(threadid*os.clock()*1.7)
         torch.setnumthreads(1)
         threadid_t = threadid
         datasource_t = getDatasourceFun()
      end)
   self.donkeys:addjob(
      function()
         return datasource_t.nChannels, datasource_t.nClasses, datasource_t.h, datasource_t.w
      end,
      function(nChannels, nClasses, h, w)
         self.nChannels, self.nClasses = nChannels, nClasses
         self.h, self.w = h, w
      end)
   self.donkeys:synchronize()
   self.donkeys:specific(false)
   self.started = false
   self.output, self.labels = self.output_cpu, self.labels_cpu

   -- TODO? does that overrides the parent __gc?:
   if newproxy then
      --lua <= 5.1
      self.__gc__ = newproxy(true)
      getmetatable(self.__gc__).__gc = 
	 function() self.output = nil end
   else
      self.__gc = function() self.output = nil end
   end

   print("Warming up data pool...")
   for i = 1, self.nDonkeys do
      self:addjob()
   end
   self.donkeys:dojob()
end

function ThreadedDatasource:addjob()
   self.donkeys:addjob(
      function()
         collectgarbage()
         collectgarbage()
          local batch = datasource_t:getSeqBatch(opt_t.batchsize, opt_t.seqLength)
          return batch
      end,
      function(batch)
         collectgarbage()
         collectgarbage()
         if #self.batch_pool < self.batch_pool_size then
            table.insert(self.batch_pool, batch)
         else
            local replacement_index = math.random(1, #self.batch_pool)
            self.batch_pool[replacement_index]:copy(batch)
         end
      end
   )
end

-- function ThreadedDatasource:type(typ)
--    parent.type(self, typ)
--    if typ == 'torch.CudaTensor' then
--       self.output, self.labels = self.output_gpu, self.labels_gpu
--    else
--       self.output, self.labels = self.output_cpu, self.labels_cpu
--    end
-- end

function ThreadedDatasource:getSeqBatch(batchSize, seqLength)
   assert(batchSize ~= nil, 'getSeqBatch: must specify batchSize')
   -- if not self.started then
   --    self.donkeys:synchronize()
   --    self.donkeys:specific(false)
   --    for i = 1, self.nDonkeys do
   --    	if self.donkeys:acceptsjob() then
   --    	   addjob()
   --    	end
   --    end
   --    self.started = true
   -- end

   if #self.batch_pool % 100 == 0 then 
      print("batch_pool size: ", #self.batch_pool)
   end
   if self.donkeys:haserror() then
      print("ThreadedDatasource: There is an error in a donkey")
      self.donkeys:terminate()
      os.exit(0)
   end

   -- queue has something for us
   -- dojob to put the newly loaded batch into the pool
   if self.donkeys.mainqueue.isempty == 0 then   
      self.donkeys:dojob()
      self:addjob()
   end
   local batch_to_use = math.random(1, #self.batch_pool)
   -- print("picked batch: ", batch_to_use)
   -- print("batch pool: ", self.batch_pool)
   self.output:resize(self.batch_pool[batch_to_use]:size())
   self.output:copy(self.batch_pool[batch_to_use])

   return self.output
end

-- function ThreadedDatasource:orderedIterator(batchSize, set)
--    -- this one doesn't parallelize on more than one thread
--    -- (this might be a TODO but seems hard)
--    assert(batchSize ~= nil, 'nextBatch: must specify batchSize')
--    assert(set ~= nil, 'nextBatch: must specify set')
--    self.donkeys:synchronize()
--    self.donkeys:specific(true)
--    self.started = false
--    self.donkeys:addjob(
--       1, function()
-- 	 collectgarbage()
-- 	 it_t = datasource_t:orderedIterator(batchSize, set) 
-- 	 end)
--    local finished = false
--    local function addjob()
--       self.donkeys:addjob(
-- 	 1,
-- 	 function()
-- 	    return it_t()
-- 	 end,
-- 	 function(output, labels)
-- 	    if output == nil then
-- 	       finished = true
-- 	    else
-- 	       if self.output ~= nil then --TODO: why is the line useful?
-- 		  self.output:resize(output:size()):copy(output)
-- 		  self.labels:resize(labels:size()):copy(labels)
-- 	       end
-- 	    end
-- 	 end)
--    end
--    return function()
--       self.donkeys:synchronize()
--       if finished then
-- 	 self.donkeys:addjob(1, function() it_t = nil collectgarbage() end)
-- 	 self.donkeys:synchronize()
--       else
-- 	 addjob()
-- 	 return self.output, self.labels
--       end
--    end
-- end