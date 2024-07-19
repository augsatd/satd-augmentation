# Configuration settings
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 10
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
labels = ["documentation_debt", "requirement_debt", "test_debt", "code-design_debt"]
