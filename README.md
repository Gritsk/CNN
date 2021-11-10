## CNN Classification 
to get better results: 
increase epoch size
also modified max pool with differnt kernel stride size, for better accuracy
   for example: with kernel size 3 and stide 1 last layer is  
torch.Size([4, 16, 20, 20])
so when applying linearity 
use: self.fc1 = nn.Linear(16 * 20 * 20, 120)

formula:
(W-F + 2p)/S+1
