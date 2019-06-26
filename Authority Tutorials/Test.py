import torch

bach_size, n_in, n_hidden, n_out = 64, 1000, 100, 10
dtype = torch.float

x = torch.randn(bach_size, n_in, dtype=dtype)
y = torch.randn(bach_size, n_out, dtype=dtype)

model = torch.nn.Sequential(
    torch.nn.Linear(n_in, n_hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(n_hidden, n_out)
)
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



