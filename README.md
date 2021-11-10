## CNN Classification for 
to get better results: 
increase epoch size
also modified max pool with 3 kernel size and 1 stride, better accuracy
    last layer is  
torch.Size([4, 16, 20, 20])
so when applying linearity 
use: self.fc1 = nn.Linear(16 * 20 * 20, 120)

formula:
(W-F + 2p)/S+1
